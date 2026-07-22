"""
Training pipeline for Harp model with 4 bands × 9 codebooks configuration.
Supports variable bitrate tiers with 3-2-2-2 codebook distribution.
"""
import gc
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from audiotools import AudioSignal
from harp.data.jamendo import Jamendo
from harp.models.harp import Harp, HarpConfig
from harp.models.discriminator import Discriminator
from harp.models.components.loss import GANLoss, MultiScaleSTFTLoss, MelSpectrogramLoss
from harp.models.components.audio_metrics import ScaleInvariantSDR, LogSpectralDistance, SignalToNoiseRatio


class HarpTrainer(L.LightningModule):
    def __init__(self, configs: dict):
        super().__init__()
        self.save_hyperparameters(configs)
        self.sample_rate = configs["data"]["sample_rate"]

        # Build Harp model
        model_configs = configs["model"]
        harp_config = HarpConfig.from_dict(model_configs)
        self.model = Harp(harp_config)
        self.n_groups = len(harp_config.stage_groups)
        self.n_codebooks = model_configs["dac"]["n_codebooks"]

        # Discriminator
        self.discriminator = Discriminator(**model_configs["discriminator"])

        # Losses
        loss_configs = configs["losses"]
        self.stft_loss = MultiScaleSTFTLoss(**loss_configs["stft_loss"])
        self.mel_loss = MelSpectrogramLoss(**loss_configs["mel_loss"])
        self.gan_loss = GANLoss(self.discriminator)
        self.lambdas = loss_configs["lambdas"]

        # Validation metrics
        self.si_sdr = ScaleInvariantSDR(reduction='mean').eval()
        self.lsd = LogSpectralDistance(n_fft=2048, reduction='mean').eval()
        self.snr = SignalToNoiseRatio(reduction='mean').eval()

        self.automatic_optimization = False
        self.optimizer_configs = configs["optimizer"]
        self.grad_clip_gen = configs.get("train", {}).get("grad_clip_gen", 1000.0)
        self.grad_clip_disc = configs.get("train", {}).get("grad_clip_disc", 10.0)

        self.bitrate_tiers = self.model.get_bitrate_tiers()

    def configure_optimizers(self):
        opt_configs = self.optimizer_configs

        opt_g = optim.AdamW(
            [
                {'params': self.model.dac.parameters(), 'lr': opt_configs["lr"]},
                {'params': self.model.band_prioritizer.parameters(), 'lr': opt_configs["lr"]},
            ],
            betas=tuple(opt_configs["betas"]),
            weight_decay=opt_configs["weight_decay"],
        )

        opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=opt_configs["lr"]/2,
            betas=tuple(opt_configs["betas"]),
            weight_decay=opt_configs["weight_decay"],
        )

        sched_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=opt_configs["gamma"])
        sched_d = optim.lr_scheduler.ExponentialLR(opt_d, gamma=opt_configs["gamma"])

        return [opt_d, opt_g], [sched_d, sched_g]

    def _compute_generator_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        preprocessed_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all generator losses."""
        recons = outputs['audio']

        input_signal = AudioSignal(preprocessed_audio, self.sample_rate)
        recons_signal = AudioSignal(recons, self.sample_rate)

        # Reconstruction losses
        stft_loss = self.stft_loss(recons_signal, input_signal)
        mel_loss = self.mel_loss(recons_signal, input_signal)

        # Adversarial losses
        gen_loss, feat_loss = self.gan_loss.generator_loss(recons_signal, input_signal)

        # Band losses
        band_losses = self.model.compute_band_losses(outputs, preprocessed_audio)

        # Total loss
        total = (
            self.lambdas.get("stft/loss", 1.0) * stft_loss +
            self.lambdas.get("mel/loss", 15.0) * mel_loss +
            self.lambdas.get("adv/gen_loss", 1.0) * gen_loss +
            self.lambdas.get("adv/feat_loss", 5.0) * feat_loss +
            self.lambdas.get("vq/commitment_loss", 0.25) * outputs['commitment_loss'] +
            self.lambdas.get("vq/codebook_loss", 1.0) * outputs['codebook_loss'] +
            self.lambdas.get("band/total", 5.0) * band_losses['band/total']
        )

        return {
            'total': total,
            'mel': mel_loss,
            'gen': gen_loss,
            'feat': feat_loss,
            'band': band_losses['band/total'],
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if batch_idx % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        opt_d, opt_g = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()

        # Forward pass
        outputs = self.model.forward_with_bands(batch, self.sample_rate, apply_dropout=True)
        recons = outputs['audio']
        preprocessed_audio = outputs['preprocessed_audio']

        # Train Discriminator
        if batch_idx % 2 == 0:
            input_signal = AudioSignal(preprocessed_audio, self.sample_rate)
            recons_signal = AudioSignal(recons.detach(), self.sample_rate)

            disc_loss = self.gan_loss.discriminator_loss(recons_signal, input_signal)

            opt_d.zero_grad()
            self.manual_backward(disc_loss)
            self.clip_gradients(opt_d, gradient_clip_val=self.grad_clip_disc, gradient_clip_algorithm="norm")
            opt_d.step()
            sched_d.step()

            self.log("train/disc", disc_loss, sync_dist=True)

        # Train Generator
        losses = self._compute_generator_loss(outputs, preprocessed_audio)

        opt_g.zero_grad()
        self.manual_backward(losses['total'])
        self.clip_gradients(opt_g, gradient_clip_val=self.grad_clip_gen, gradient_clip_algorithm="norm")
        opt_g.step()
        sched_g.step()

        # Logging
        self.log("train/loss", losses['total'], prog_bar=True, sync_dist=True)
        self.log("train/mel", losses['mel'], sync_dist=True)
        self.log("train/band", losses['band'], sync_dist=True)
        self.log("train/gen", losses['gen'], sync_dist=True)
        self.log("train/feat", losses['feat'], sync_dist=True)
        self.log("train/commitment", outputs['commitment_loss'], sync_dist=True)
        self.log("train/codebook", outputs['codebook_loss'], sync_dist=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if batch_idx % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Full-rate validation
            outputs = self.model.forward_with_bands(
                batch, self.sample_rate, n_groups=self.n_groups, apply_dropout=False
            )
            recons = outputs['audio']
            preprocessed_audio = outputs['preprocessed_audio']

            # Metrics
            input_signal = AudioSignal(preprocessed_audio, self.sample_rate)
            recons_signal = AudioSignal(recons, self.sample_rate)

            mel_loss = self.mel_loss(recons_signal, input_signal)
            si_sdr = self.si_sdr(recons, preprocessed_audio)
            lsd = self.lsd(recons, preprocessed_audio)
            snr = self.snr(recons, preprocessed_audio)

            self.log("val/mel", mel_loss, prog_bar=True, sync_dist=True)
            self.log("val/si_sdr", si_sdr, prog_bar=True, sync_dist=True)
            self.log("val/lsd", lsd, sync_dist=True)
            self.log("val/snr", snr, sync_dist=True)
            self.log("val/commitment", outputs['commitment_loss'], sync_dist=True)
            self.log("val/codebook", outputs['codebook_loss'], sync_dist=True)

            # Multi-bitrate SI-SDR (evaluate all lower tiers)
            for n_groups in range(1, self.n_groups):
                outputs_tier = self.model.forward_with_bands(
                    batch, self.sample_rate, n_groups=n_groups, apply_dropout=False
                )
                si_sdr_tier = self.si_sdr(outputs_tier['audio'], preprocessed_audio)
                self.log(f"val/si_sdr_{n_groups}g", si_sdr_tier, sync_dist=True)


def main(config_path: str = "harp/configs/train_harp.yaml"):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    train_configs = configs["train"]
    data_configs = configs["data"]

    model = HarpTrainer(configs)

    # Print model configuration
    print("=" * 50)
    print("HARP Model Configuration:")
    print(f"  Codebooks: {model.n_codebooks}")
    print(f"  Groups: {model.n_groups}")
    print("=" * 50)
    print("HARP Bitrate Tiers:")
    for n_groups, bitrate in model.bitrate_tiers.items():
        print(f"  {n_groups} group(s): {bitrate:.2f} kbps")
    print("=" * 50)

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
        filename="harp-{epoch:02d}",
        monitor="val/si_sdr",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
    )

    logger = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        save_dir=train_configs["logdir"],
        name="harp",
    )

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        num_nodes=train_configs.get("num_nodes", 1),
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