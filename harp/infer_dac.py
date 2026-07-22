"""
DAC Inference Script - With Dataset Evaluation Support

Supports variable codebook counts for different bitrates:
    - 1 codebook:  ~0.86 kbps
    - 3 codebooks: ~2.58 kbps
    - 5 codebooks: ~4.30 kbps
    - 7 codebooks: ~6.02 kbps
    - 9 codebooks: ~7.74 kbps (full quality)
"""
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiotools import AudioSignal
from harp.data.jamendo import Jamendo
from harp.data.libritts import LibriTTS
from harp.data.musdb18 import MUSDB18
from harp.models.dac import DAC
from harp.models.components.loss import MelSpectrogramLoss
from harp.models.components.audio_metrics import LogSpectralDistance, SignalToNoiseRatio, ScaleInvariantSDR


# Codebook tiers matching HARP's bitrate tiers
CODEBOOK_TIERS = [3, 5, 7, 9]


class DACInference:

    DATASET_ROOTS = {
        "Jamendo": "/data/Jamendo",
        "LibriTTS": "/data/LibriTTS",
        "MUSDB18": "/data/musdb18hq",
    }

    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None,
                 device: str = "cuda", chunk_duration: float = 10.0):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_duration = chunk_duration

        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.configs = yaml.safe_load(f)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.configs = checkpoint.get("hyper_parameters", {})

        self.sample_rate = self.configs["data"]["sample_rate"]

        # Initialize model
        model_configs = self.configs["model"]["dac"]
        self.model = DAC(**model_configs)
        self.n_codebooks = model_configs.get("n_codebooks", 9)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device).eval()

        # Metrics
        loss_cfg = self.configs.get("losses", {}).get("mel_loss", {})
        self.mel_loss = MelSpectrogramLoss(**loss_cfg).to(self.device).eval()
        self.lsd = LogSpectralDistance(n_fft=2048, reduction='mean').to(self.device).eval()
        self.snr = SignalToNoiseRatio(reduction='mean').to(self.device).eval()
        self.si_sdr = ScaleInvariantSDR(reduction='mean').to(self.device).eval()

        # Print model info
        print("=" * 60)
        print(f"Loaded: {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Sample rate: {self.sample_rate}")
        print(f"Codebooks: {self.n_codebooks}")
        print("-" * 60)
        print("Bitrate tiers:")
        for n_cb in range(1, self.n_codebooks + 1):
            bitrate = self.model.get_bitrate(n_cb)
            print(f"  {n_cb} codebook(s): {bitrate:.2f} kbps")
        print("=" * 60)

    def _load_audio(self, path: Union[str, Path]) -> AudioSignal:
        signal = AudioSignal(path)
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)
        if signal.num_channels > 1:
            signal = signal.to_mono()
        return signal

    @torch.no_grad()
    def _forward(self, audio: torch.Tensor, n_codebooks: Optional[int] = None) -> dict:
        audio = audio.to(self.device)
        return self.model(audio, self.sample_rate, n_quantizers=n_codebooks)

    @torch.no_grad()
    def _compute_metrics(self, recons: torch.Tensor, original: torch.Tensor) -> dict:
        recons, original = recons.to(self.device), original.to(self.device)
        return {
            'mel_loss': self.mel_loss(AudioSignal(recons, self.sample_rate),
                                       AudioSignal(original, self.sample_rate)).item(),
            'si_sdr': self.si_sdr(recons, original).item(),
            'lsd': self.lsd(recons, original).item(),
            'snr': self.snr(recons, original).item(),
        }

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     n_codebooks: Optional[int] = None) -> tuple:
        signal = self._load_audio(input_path)
        audio = signal.audio_data
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        n_cb = n_codebooks or self.n_codebooks
        bitrate = self.model.get_bitrate(n_cb)
        print(f"Processing: {Path(input_path).name} ({signal.signal_duration:.2f}s)")
        print(f"Using {n_cb} codebook(s), ~{bitrate:.2f} kbps")

        if audio.shape[-1] <= chunk_samples:
            out = self._forward(audio, n_codebooks)
            recons, preproc = out['audio'].cpu(), out['preprocessed_audio'].cpu()
        else:
            recons_chunks, preproc_chunks = [], []
            num_chunks = (audio.shape[-1] + chunk_samples - 1) // chunk_samples

            for i in tqdm(range(num_chunks), desc="Processing"):
                start, end = i * chunk_samples, min((i + 1) * chunk_samples, audio.shape[-1])
                chunk = audio[:, :, start:end]
                if chunk.shape[-1] < chunk_samples:
                    chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

                out = self._forward(chunk, n_codebooks)
                actual_len = end - start
                recons_chunks.append(out['audio'][:, :, :actual_len].cpu())
                preproc_chunks.append(out['preprocessed_audio'][:, :, :actual_len].cpu())

                if i % 10 == 0: gc.collect(); torch.cuda.empty_cache()

            recons = torch.cat(recons_chunks, dim=-1)
            preproc = torch.cat(preproc_chunks, dim=-1)

        metrics = self._compute_metrics(recons, preproc)
        print(f"SI-SDR={metrics['si_sdr']:.2f}dB, Mel={metrics['mel_loss']:.4f}, "
              f"LSD={metrics['lsd']:.4f}, SNR={metrics['snr']:.2f}dB")

        out_signal = AudioSignal(recons, self.sample_rate).normalize(-1)
        out_signal.audio_data = torch.clamp(out_signal.audio_data, -1.0, 1.0)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_signal.write(output_path)
            print(f"Saved: {output_path}")

        return out_signal, metrics

    def evaluate_codebooks(self, input_path: str, output_dir: Optional[str] = None,
                           codebook_list: Optional[List[int]] = None) -> dict:
        """Evaluate specified codebook configurations on a single file."""
        signal = self._load_audio(input_path)
        max_samples = int(min(signal.signal_duration, self.chunk_duration) * self.sample_rate)
        audio = signal.audio_data[:, :, :max_samples].to(self.device)
        preproc = self._forward(audio)['preprocessed_audio']

        codebooks = codebook_list or list(range(1, self.n_codebooks + 1))

        print(f"\n{'Codebooks':<12}{'Bitrate':<12}{'SI-SDR':<12}{'Mel':<12}{'LSD':<12}{'SNR':<12}")
        print("-" * 72)

        results = {}
        for n_cb in codebooks:
            m = self._compute_metrics(self._forward(audio, n_cb)['audio'], preproc)
            m['bitrate_kbps'] = self.model.get_bitrate(n_cb)
            results[n_cb] = m
            print(f"{n_cb:<12}{m['bitrate_kbps']:<12.2f}{m['si_sdr']:<12.2f}"
                  f"{m['mel_loss']:<12.4f}{m['lsd']:<12.4f}{m['snr']:<12.2f}")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for n_cb in codebooks:
                out = AudioSignal(self._forward(signal.audio_data, n_cb)['audio'].cpu(), self.sample_rate).normalize(-1)
                out.audio_data = torch.clamp(out.audio_data, -1.0, 1.0)
                bitrate = self.model.get_bitrate(n_cb)
                out.write(Path(output_dir) / f"{Path(input_path).stem}_{n_cb}cb_{bitrate:.0f}kbps.wav")

        return results

    def evaluate_dataset(self, split: str = "val", max_batches: Optional[int] = None,
                         eval_all_codebooks: bool = False, output_path: Optional[str] = None,
                         dataset_type: Optional[str] = None,
                         n_codebooks: Optional[int] = None,
                         codebook_tiers: bool = False) -> dict:
        """Evaluate DAC on a dataset.
        
        Args:
            split: Dataset split to evaluate.
            max_batches: Maximum number of batches to evaluate.
            eval_all_codebooks: If True, evaluate all codebook counts (1-9).
            output_path: Path to save results JSON.
            dataset_type: Dataset type (Jamendo, LibriTTS, MUSDB18).
            n_codebooks: Specific number of codebooks to evaluate.
            codebook_tiers: If True, evaluate only [3, 5, 7, 9] codebooks (matching HARP tiers).
        """
        data_cfg = self.configs.get("data", {}).copy()
        dataset_type = dataset_type or data_cfg.get("dataset_type", "Jamendo")

        config_roots = data_cfg.get("dataset_roots", {})
        all_roots = {**self.DATASET_ROOTS, **config_roots}
        if dataset_type in all_roots:
            data_cfg["dataset_root"] = all_roots[dataset_type]

        Dataset = {"Jamendo": Jamendo, "LibriTTS": LibriTTS, "MUSDB18": MUSDB18}[dataset_type]
        dataset = Dataset(split=split, data_configs=data_cfg)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        total = min(max_batches, len(loader)) if max_batches else len(loader)

        print(f"\nEvaluating {dataset_type} {split}: {len(dataset)} samples")
        print(f"Model: DAC with {self.n_codebooks} codebooks")

        # Determine which codebook counts to evaluate
        if codebook_tiers:
            codebooks = [cb for cb in CODEBOOK_TIERS if cb <= self.n_codebooks]
            print(f"Evaluating codebook tiers: {codebooks}")
        elif eval_all_codebooks:
            codebooks = list(range(1, self.n_codebooks + 1))
        else:
            codebooks = [n_codebooks or self.n_codebooks]

        metrics = {cb: {k: [] for k in ['mel_loss', 'si_sdr', 'lsd', 'snr']} for cb in codebooks}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader, total=total)):
                if max_batches and idx >= max_batches: break
                if idx % 200 == 0: gc.collect(); torch.cuda.empty_cache()

                batch = batch.to(self.device)
                preproc = self._forward(batch)['preprocessed_audio']

                for n_cb in codebooks:
                    m = self._compute_metrics(self._forward(batch, n_cb)['audio'], preproc)
                    for k, v in m.items(): metrics[n_cb][k].append(v)

        results = {}
        show_table = eval_all_codebooks or codebook_tiers

        if show_table:
            print(f"\n{'Codebooks':<12}{'Bitrate':<12}{'SI-SDR':<12}{'Mel':<12}{'LSD':<12}{'SNR':<12}")
            print("-" * 72)

        for n_cb in codebooks:
            avg = {k: sum(v)/len(v) for k, v in metrics[n_cb].items()}
            avg['bitrate_kbps'] = self.model.get_bitrate(n_cb)
            avg['n_samples'] = len(metrics[n_cb]['mel_loss'])
            results[f'{n_cb}_codebooks' if show_table else 'full'] = avg

            if show_table:
                print(f"{n_cb:<12}{avg['bitrate_kbps']:<12.2f}{avg['si_sdr']:<12.2f}"
                      f"{avg['mel_loss']:<12.4f}{avg['lsd']:<12.4f}{avg['snr']:<12.2f}")
            else:
                print(f"SI-SDR={avg['si_sdr']:.2f}dB, Mel={avg['mel_loss']:.4f}, "
                      f"LSD={avg['lsd']:.4f}, SNR={avg['snr']:.2f}dB")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    'dataset': dataset_type,
                    'split': split,
                    'n_codebooks': self.n_codebooks,
                    'results': results
                }, f, indent=2)
            print(f"Saved: {output_path}")

        return results

    def generate_examples(self, split: str = "val", n_samples: int = 5,
                          output_dir: str = "./examples", dataset_type: Optional[str] = None,
                          n_codebooks: Optional[int] = None, save_original: bool = True,
                          all_tiers: bool = False) -> None:
        """Generate example reconstructions.
        
        Args:
            split: Dataset split to use.
            n_samples: Number of samples to generate.
            output_dir: Output directory.
            dataset_type: Dataset type.
            n_codebooks: Number of codebooks (ignored if all_tiers=True).
            save_original: Whether to save original audio.
            all_tiers: If True, generate all tiers [3, 5, 7, 9] for each sample.
        """
        data_cfg = self.configs.get("data", {}).copy()
        dataset_type = dataset_type or data_cfg.get("dataset_type", "Jamendo")

        config_roots = data_cfg.get("dataset_roots", {})
        all_roots = {**self.DATASET_ROOTS, **config_roots}
        if dataset_type in all_roots:
            data_cfg["dataset_root"] = all_roots[dataset_type]

        Dataset = {"Jamendo": Jamendo, "LibriTTS": LibriTTS, "MUSDB18": MUSDB18}[dataset_type]
        dataset = Dataset(split=split, data_configs=data_cfg)

        if len(dataset) == 0:
            print(f"No samples found in {dataset_type} {split}")
            return

        n_samples = min(n_samples, len(dataset))
        
        # Determine codebook tiers to generate
        if all_tiers:
            tiers = [cb for cb in CODEBOOK_TIERS if cb <= self.n_codebooks]
            output_dir = Path(output_dir) / dataset_type / split
        else:
            n_cb = n_codebooks or self.n_codebooks
            tiers = [n_cb]
            bitrate = self.model.get_bitrate(n_cb)
            if n_codebooks and n_codebooks < self.n_codebooks:
                output_dir = Path(output_dir) / dataset_type / f"{split}_{n_cb}cb_{bitrate:.0f}kbps"
            else:
                output_dir = Path(output_dir) / dataset_type / split
        
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {n_samples} examples from {dataset_type} {split}")
        print(f"Codebook tiers: {tiers}")
        print(f"Save original: {save_original}")
        print(f"Output: {output_dir}")
        print("-" * 60)

        chunk_samples = int(self.chunk_duration * self.sample_rate)

        with torch.no_grad():
            for i in tqdm(range(n_samples), desc="Generating"):
                track_index = dataset.tracks[i]
                audio_path = dataset.get_audio_path(track_index)
                signal = self._load_audio(audio_path)
                audio = signal.audio_data

                # Get preprocessed audio first (using full codebooks)
                if audio.shape[-1] <= chunk_samples:
                    out_full = self._forward(audio, self.n_codebooks)
                    preproc = out_full['preprocessed_audio'].cpu()
                else:
                    preproc_chunks = []
                    num_chunks = (audio.shape[-1] + chunk_samples - 1) // chunk_samples

                    for j in range(num_chunks):
                        start, end = j * chunk_samples, min((j + 1) * chunk_samples, audio.shape[-1])
                        chunk = audio[:, :, start:end]
                        if chunk.shape[-1] < chunk_samples:
                            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

                        out = self._forward(chunk, self.n_codebooks)
                        actual_len = end - start
                        preproc_chunks.append(out['preprocessed_audio'][:, :, :actual_len].cpu())

                    preproc = torch.cat(preproc_chunks, dim=-1)

                # Save original if requested
                if save_original:
                    AudioSignal(preproc, self.sample_rate).write(output_dir / f"sample_{i:03d}_original.wav")

                # Generate each tier
                for n_cb in tiers:
                    if audio.shape[-1] <= chunk_samples:
                        out = self._forward(audio, n_cb)
                        recons = out['audio'].cpu()
                    else:
                        recons_chunks = []
                        num_chunks = (audio.shape[-1] + chunk_samples - 1) // chunk_samples

                        for j in range(num_chunks):
                            start, end = j * chunk_samples, min((j + 1) * chunk_samples, audio.shape[-1])
                            chunk = audio[:, :, start:end]
                            if chunk.shape[-1] < chunk_samples:
                                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

                            out = self._forward(chunk, n_cb)
                            actual_len = end - start
                            recons_chunks.append(out['audio'][:, :, :actual_len].cpu())

                        recons = torch.cat(recons_chunks, dim=-1)

                    metrics = self._compute_metrics(recons, preproc)
                    bitrate = self.model.get_bitrate(n_cb)

                    # Save reconstructed audio
                    if all_tiers:
                        filename = f"sample_{i:03d}_{n_cb}cb_{bitrate:.0f}kbps.wav"
                    else:
                        filename = f"sample_{i:03d}_reconstructed.wav"
                    AudioSignal(recons, self.sample_rate).write(output_dir / filename)

                    print(f"Sample {i}, {n_cb}cb ({bitrate:.1f}kbps): SI-SDR={metrics['si_sdr']:.2f}dB, "
                          f"duration={signal.signal_duration:.2f}s")

                gc.collect(); torch.cuda.empty_cache()

        print("-" * 60)
        print(f"Saved examples to {output_dir}")


def main(args=None, config_path=None):
    p = argparse.ArgumentParser(description="DAC Inference - Neural Audio Codec")
    p.add_argument("--ckpt", "--checkpoint", type=str, dest="checkpoint",
                   help="Checkpoint path (or set checkpoint_path in config)")
    p.add_argument("--config", "-c", type=str, default="harp/configs/train_dac.yaml",
                   help="Config file path")
    p.add_argument("-i", "--input", type=str, help="Input audio file")
    p.add_argument("-o", "--output", type=str, help="Output path (file or directory)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk-duration", type=float, default=10.0)

    # Codebook control
    p.add_argument("--n-codebooks", type=int,
                   help="Number of codebooks to use (fewer=lower bitrate)")
    
    # Modes
    p.add_argument("--eval-dataset", action="store_true", help="Evaluate on dataset")
    p.add_argument("--eval-codebooks", action="store_true", help="Evaluate all codebook counts on single file")
    p.add_argument("--generate-examples", action="store_true", help="Generate example reconstructions")

    # Dataset options
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--dataset", type=str, choices=["Jamendo", "LibriTTS", "MUSDB18"])
    p.add_argument("--max-batches", type=int, help="Max batches for quick eval")
    p.add_argument("--all-codebooks", action="store_true", help="Evaluate all codebook counts (1-9) on dataset")
    p.add_argument("--codebook-tiers", action="store_true", 
                   help="Evaluate only [3, 5, 7, 9] codebooks (matching HARP bitrate tiers)")
    p.add_argument("--all-tiers", action="store_true",
                   help="Generate examples for all tiers [3, 5, 7, 9]")
    p.add_argument("--n-samples", type=int, default=5, help="Number of examples to generate")
    p.add_argument("--no-original", action="store_true", help="Don't save original audio when generating examples")

    args = p.parse_args(args)

    cfg_path = config_path or args.config

    ckpt = args.checkpoint
    if not ckpt:
        if cfg_path and Path(cfg_path).exists():
            with open(cfg_path) as f:
                ckpt = yaml.safe_load(f).get("checkpoint_path")
    if not ckpt and Path("checkpoints/dac.ckpt").exists():
        ckpt = "checkpoints/dac.ckpt"
    if not ckpt:
        p.error("no checkpoint found; pass --ckpt, place one at "
                "checkpoints/dac.ckpt, or set checkpoint_path in config")

    dac = DACInference(
        checkpoint_path=ckpt,
        config_path=cfg_path,
        device=args.device,
        chunk_duration=args.chunk_duration,
    )

    if args.eval_dataset:
        dac.evaluate_dataset(
            split=args.split,
            max_batches=args.max_batches,
            eval_all_codebooks=args.all_codebooks,
            output_path=args.output,
            dataset_type=args.dataset,
            n_codebooks=args.n_codebooks,
            codebook_tiers=args.codebook_tiers,
        )
    elif args.eval_codebooks:
        if not args.input: p.error("--input required")
        # Use tiers if specified, otherwise all codebooks
        codebook_list = CODEBOOK_TIERS if args.codebook_tiers else None
        dac.evaluate_codebooks(args.input, args.output, codebook_list=codebook_list)
    elif args.generate_examples:
        dac.generate_examples(
            split=args.split,
            n_samples=args.n_samples,
            output_dir=args.output or "./examples",
            dataset_type=args.dataset,
            n_codebooks=args.n_codebooks,
            save_original=not args.no_original,
            all_tiers=args.all_tiers,
        )
    else:
        if not args.input: p.error("--input required")
        output = args.output or f"{Path(args.input).stem}_reconstructed.wav"
        dac.process_file(args.input, output, args.n_codebooks)

    print("Done!")


if __name__ == "__main__":
    main()