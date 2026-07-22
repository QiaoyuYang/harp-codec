import gc
import yaml
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from audiotools import AudioSignal

from harp.data.jamendo import Jamendo
from harp.models.dac import DAC
from harp.models.discriminator import Discriminator
from harp.models.components.loss import GANLoss, MultiScaleSTFTLoss, MelSpectrogramLoss, L1Loss
from harp.models.components.audio_metrics import LogSpectralDistance, SignalToNoiseRatio, ScaleInvariantSDR


class DACTrainer(L.LightningModule):
    
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters(configs)
        
        self.sample_rate = configs["data"]["sample_rate"]
        
        # Model
        model_configs = configs["model"]
        dac_configs = model_configs["dac"]
        self.model = DAC(**dac_configs)
        discriminator_configs = model_configs["discriminator"]
        self.discriminator = Discriminator(**discriminator_configs)
        
        # Losses
        loss_configs = configs["losses"]
        stft_loss_configs = loss_configs["stft_loss"]
        mel_loss_configs = loss_configs["mel_loss"]
        self.stft_loss = MultiScaleSTFTLoss(**stft_loss_configs)
        self.mel_loss = MelSpectrogramLoss(**mel_loss_configs)
        self.waveform_loss = L1Loss()
        self.gan_loss = GANLoss(self.discriminator)
        self.lambdas = loss_configs["lambdas"]
        
        # Validation metrics
        self.lsd = LogSpectralDistance(n_fft=2048, reduction='mean').eval()
        self.snr = SignalToNoiseRatio(reduction='mean').eval()
        self.si_sdr = ScaleInvariantSDR(reduction='mean').eval()
        
        self.automatic_optimization = False
        self.optimizer_configs = configs["optimizer"]
        self.grad_clip_gen = configs.get("train", {}).get("grad_clip_gen", 1000.0)
        self.grad_clip_disc = configs.get("train", {}).get("grad_clip_disc", 1.0)
    
    def configure_optimizers(self):

        opt_configs = self.optimizer_configs

        opt_g = optim.AdamW(
            self.model.parameters(),
            lr=opt_configs["lr"],
            betas=tuple(opt_configs["betas"]),
            weight_decay=opt_configs["weight_decay"],
        )
        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=opt_configs["lr"] / 2,
            betas=tuple(opt_configs["betas"]),
            weight_decay=opt_configs["weight_decay"],
        )
        
        sched_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=opt_configs["gamma"])
        sched_d = optim.lr_scheduler.ExponentialLR(opt_d, gamma=opt_configs["gamma"])
        
        return [opt_d, opt_g], [sched_d, sched_g]
    
    def training_step(self, batch, batch_idx):
        if batch_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        opt_d, opt_g = self.optimizers()
        
        audio = batch
        output = self.model(audio, self.sample_rate)
        recons = output["audio"]
        
        input_signal = AudioSignal(audio, self.sample_rate)
        recons_signal = AudioSignal(recons, self.sample_rate)
        
        # Train Discriminator every step
        if batch_idx % 2 == 0:
            disc_loss = self.gan_loss.discriminator_loss(recons_signal.detach(), input_signal)
            
            opt_d.zero_grad()
            self.manual_backward(disc_loss)
            self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip_disc, gradient_clip_algorithm="norm")
            opt_d.step()
            
            self.log("train/disc", disc_loss, sync_dist=True)
        
        # Train Generator
        stft_loss = self.stft_loss(recons_signal, input_signal)
        mel_loss = self.mel_loss(recons_signal, input_signal)
        waveform_loss = self.waveform_loss(recons_signal, input_signal)
        gen_loss, feat_loss = self.gan_loss.generator_loss(recons_signal, input_signal)
        commitment_loss = output["vq/commitment_loss"]
        codebook_loss = output["vq/codebook_loss"]
        
        total_loss = (
            self.lambdas.get("stft/loss", 1.0) * stft_loss +
            self.lambdas.get("mel/loss", 15.0) * mel_loss +
            self.lambdas.get("waveform/loss", 0.0) * waveform_loss +
            self.lambdas.get("adv/gen_loss", 1.0) * gen_loss +
            self.lambdas.get("adv/feat_loss", 5.0) * feat_loss +
            self.lambdas.get("vq/commitment_loss", 0.25) * commitment_loss +
            self.lambdas.get("vq/codebook_loss", 1.0) * codebook_loss
        )
        
        opt_g.zero_grad()
        self.manual_backward(total_loss)
        self.clip_gradients(opt_g, gradient_clip_val=self.grad_clip_gen, gradient_clip_algorithm="norm")
        opt_g.step()
        
        # Logging
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/mel", mel_loss, sync_dist=True)
        self.log("train/gen", gen_loss, sync_dist=True)
        self.log("train/feat", feat_loss, sync_dist=True)
        self.log("train/commitment", commitment_loss, sync_dist=True)
        self.log("train/codebook", codebook_loss, sync_dist=True)

    def on_train_epoch_end(self):
        # Step schedulers once per epoch, not per batch
        sched_d, sched_g = self.lr_schedulers()
        sched_d.step()
        sched_g.step()
        
    def validation_step(self, batch, batch_idx):
        if batch_idx % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        audio = batch
        
        with torch.no_grad():
            output = self.model(audio, self.sample_rate)
            recons = output["audio"]
            
            input_signal = AudioSignal(audio, self.sample_rate)
            recons_signal = AudioSignal(recons, self.sample_rate)
            
            mel_loss = self.mel_loss(recons_signal, input_signal)
            si_sdr = self.si_sdr(recons, audio)
            lsd = self.lsd(recons, audio)
            snr = self.snr(recons, audio)
        
        # Logging (aligned with train_harp.py style)
        self.log("val/mel", mel_loss, prog_bar=True, sync_dist=True)
        self.log("val/si_sdr", si_sdr, prog_bar=True, sync_dist=True)
        self.log("val/lsd", lsd, sync_dist=True)
        self.log("val/snr", snr, sync_dist=True)
        self.log("val/commitment", output["vq/commitment_loss"], sync_dist=True)
        self.log("val/codebook", output["vq/codebook_loss"], sync_dist=True)


def main(config_path: str = "harp/configs/train_dac.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    
    train_configs = configs["train"]
    data_configs = configs["data"]

    model = DACTrainer(configs)

    # Dataset
    dataset_type = data_configs.get("dataset_type", "Jamendo")
    if dataset_type == "Jamendo":
        train_dataset = Jamendo(split="train", data_configs=data_configs)
        val_dataset = Jamendo(split="val", data_configs=data_configs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_configs["batch_size"],
        shuffle=True,
        num_workers=train_configs["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_configs["batch_size"],
        shuffle=False,
        num_workers=train_configs["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename="dac-{epoch:02d}",
        monitor="val/si_sdr",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
    )
    
    logger = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        save_dir=train_configs["logdir"],
        name="dac",
    )
    
    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        accelerator="gpu",
        devices=train_configs.get("devices", 1),
        num_sanity_val_steps=0,
        max_epochs=train_configs["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        precision=32,
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()