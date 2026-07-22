"""
Harp Inference Script - Aligned with train_harp.py

Supports 9 codebooks with 4 bands (3-2-2-2 distribution):
    - 1 group (3 codebooks): ~2.6 kbps
    - 2 groups (5 codebooks): ~4.3 kbps
    - 3 groups (7 codebooks): ~6.0 kbps
    - 4 groups (9 codebooks): ~7.7 kbps

Modes:
    (default)           Reconstruct a single audio file at a chosen bitrate tier.
    --eval-tiers        Report metrics for every bitrate tier on a single file.
    --analyze           Analyze learned band specialization on a single file.
    --eval-dataset      Evaluate metrics over a dataset split.
    --generate-examples Render reconstructions for dataset samples.
"""
import argparse
import gc
import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from audiotools import AudioSignal
from harp.data.jamendo import Jamendo
from harp.data.libritts import LibriTTS
from harp.data.musdb18 import MUSDB18
from harp.models.harp import Harp, HarpConfig
from harp.models.components.loss import MelSpectrogramLoss
from harp.models.components.audio_metrics import (
    LogSpectralDistance,
    SignalToNoiseRatio,
    ScaleInvariantSDR,
)


class HarpInference:

    DATASET_ROOTS = {
        "Jamendo": "/data/Jamendo",
        "LibriTTS": "/data/LibriTTS",
        "MUSDB18": "/data/musdb18hq",
    }

    def __init__(self, checkpoint_path: str, config_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_duration = 10.0

        # Load config file (required for data paths)
        if not config_path or not Path(config_path).exists():
            raise ValueError("Config file required")

        with open(config_path, "r") as f:
            self.configs = yaml.safe_load(f)

        # Load checkpoint for model weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_configs = checkpoint.get("hyper_parameters", {})

        # Use model config from checkpoint (architecture must match)
        if "model" in ckpt_configs:
            self.configs["model"] = ckpt_configs["model"]
        if "losses" in ckpt_configs:
            self.configs["losses"] = ckpt_configs["losses"]

        self.sample_rate = self.configs["data"]["sample_rate"]

        # Initialize model
        harp_config = HarpConfig.from_dict(self.configs["model"])
        self.model = Harp(harp_config)
        self.n_groups = len(harp_config.stage_groups)
        self.n_codebooks = self.configs["model"]["dac"]["n_codebooks"]

        # Load weights
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        self.bitrate_tiers = self.model.get_bitrate_tiers()

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
        print(f"Groups: {self.n_groups}")
        print("-" * 60)
        print("Bitrate tiers:")
        for ng, bitrate in self.bitrate_tiers.items():
            n_cb = self.model.groups_to_quantizers(ng)
            print(f"  {ng} group(s) ({n_cb} codebooks): {bitrate:.2f} kbps")
        print("=" * 60)

    def _load_audio(self, path: Union[str, Path]) -> AudioSignal:
        signal = AudioSignal(path)
        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)
        if signal.num_channels > 1:
            signal = signal.to_mono()
        return signal

    @torch.no_grad()
    def _forward(self, audio: torch.Tensor, n_groups: Optional[int] = None) -> dict:
        return self.model.forward_with_bands(
            audio.to(self.device), self.sample_rate,
            n_groups=n_groups or self.n_groups, apply_dropout=False
        )

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
                     n_groups: Optional[int] = None) -> tuple:
        """Reconstruct a single arbitrary audio file at the requested bitrate tier."""
        signal = self._load_audio(input_path)
        audio = signal.audio_data
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        n_cb = self.model.groups_to_quantizers(n_groups or self.n_groups)
        bitrate = self.bitrate_tiers[n_groups or self.n_groups]
        print(f"Processing: {Path(input_path).name} ({signal.signal_duration:.2f}s)")
        print(f"Using {n_groups or self.n_groups} group(s), {n_cb} codebooks, ~{bitrate:.2f} kbps")

        if audio.shape[-1] <= chunk_samples:
            out = self._forward(audio, n_groups)
            recons, preproc = out['audio'].cpu(), out['preprocessed_audio'].cpu()
        else:
            recons_chunks, preproc_chunks = [], []
            num_chunks = (audio.shape[-1] + chunk_samples - 1) // chunk_samples

            for i in tqdm(range(num_chunks), desc="Processing"):
                start, end = i * chunk_samples, min((i + 1) * chunk_samples, audio.shape[-1])
                chunk = audio[:, :, start:end]
                if chunk.shape[-1] < chunk_samples:
                    chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

                out = self._forward(chunk, n_groups)
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

    def evaluate_tiers(self, input_path: str, output_dir: Optional[str] = None) -> dict:
        """Report metrics for every bitrate tier on a single file."""
        signal = self._load_audio(input_path)
        max_samples = int(min(signal.signal_duration, self.chunk_duration) * self.sample_rate)
        audio = signal.audio_data[:, :, :max_samples].to(self.device)
        preproc = self._forward(audio)['preprocessed_audio']

        print(f"\n{'Groups':<8}{'Codebooks':<12}{'Bitrate':<12}{'SI-SDR':<12}{'Mel':<12}{'LSD':<12}{'SNR':<12}")
        print("-" * 80)

        results = {}
        for ng in range(1, self.n_groups + 1):
            m = self._compute_metrics(self._forward(audio, ng)['audio'], preproc)
            m['bitrate_kbps'] = self.bitrate_tiers[ng]
            m['n_codebooks'] = self.model.groups_to_quantizers(ng)
            results[ng] = m
            print(f"{ng:<8}{m['n_codebooks']:<12}{m['bitrate_kbps']:<12.2f}{m['si_sdr']:<12.2f}{m['mel_loss']:<12.4f}{m['lsd']:<12.4f}{m['snr']:<12.2f}")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            for ng in range(1, self.n_groups + 1):
                out = AudioSignal(self._forward(signal.audio_data, ng)['audio'].cpu(), self.sample_rate).normalize(-1)
                out.audio_data = torch.clamp(out.audio_data, -1.0, 1.0)
                n_cb = self.model.groups_to_quantizers(ng)
                out.write(Path(output_dir) / f"{Path(input_path).stem}_{ng}g_{n_cb}cb_{self.bitrate_tiers[ng]:.0f}kbps.wav")

        return results

    def evaluate_dataset(self, split: str = "val", max_batches: Optional[int] = None,
                         eval_all_tiers: bool = False, output_path: Optional[str] = None,
                         dataset_type: Optional[str] = None) -> dict:
        data_cfg = self.configs.get("data", {}).copy()
        dataset_type = dataset_type or data_cfg.get("dataset_type", "Jamendo")

        # Override dataset_root if we have a specific path for this dataset
        config_roots = data_cfg.get("dataset_roots", {})
        all_roots = {**self.DATASET_ROOTS, **config_roots}  # Config overrides defaults

        if dataset_type in all_roots:
            data_cfg["dataset_root"] = all_roots[dataset_type]
            print(f"Using dataset root: {data_cfg['dataset_root']}")

        Dataset = {"Jamendo": Jamendo, "LibriTTS": LibriTTS, "MUSDB18": MUSDB18}[dataset_type]
        dataset = Dataset(split=split, data_configs=data_cfg)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        total = min(max_batches, len(loader)) if max_batches else len(loader)

        print(f"\nEvaluating {dataset_type} {split}: {len(dataset)} samples")
        print(f"Model: {self.n_codebooks} codebooks, {self.n_groups} groups")

        tiers = range(1, self.n_groups + 1) if eval_all_tiers else [self.n_groups]
        metrics = {t: {k: [] for k in ['mel_loss', 'si_sdr', 'lsd', 'snr']} for t in tiers}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader, total=total)):
                if max_batches and idx >= max_batches: break
                if idx % 200 == 0: gc.collect(); torch.cuda.empty_cache()

                batch = batch.to(self.device)
                preproc = self._forward(batch)['preprocessed_audio']

                for ng in tiers:
                    recons = self._forward(batch, ng)['audio']
                    m = self._compute_metrics(recons, preproc)
                    for k, v in m.items(): metrics[ng][k].append(v)

        results = {}
        if eval_all_tiers:
            header = f"{'Groups':<8}{'Codebooks':<12}{'Bitrate':<12}{'SI-SDR':<12}{'Mel':<12}{'LSD':<12}{'SNR':<12}"
            print(f"\n{header}")
            print("-" * 80)

        for ng in tiers:
            avg = {k: sum(v)/len(v) for k, v in metrics[ng].items()}
            avg['bitrate_kbps'] = self.bitrate_tiers[ng]
            avg['n_codebooks'] = self.model.groups_to_quantizers(ng)

            results[f'{ng}_groups' if eval_all_tiers else 'full'] = avg

            if eval_all_tiers:
                line = f"{ng:<8}{avg['n_codebooks']:<12}{avg['bitrate_kbps']:<12.2f}{avg['si_sdr']:<12.2f}{avg['mel_loss']:<12.4f}{avg['lsd']:<12.4f}{avg['snr']:<12.2f}"
                print(line)
            else:
                msg = f"SI-SDR={avg['si_sdr']:.2f}dB, Mel={avg['mel_loss']:.4f}, LSD={avg['lsd']:.4f}, SNR={avg['snr']:.2f}dB"
                print(msg)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    'dataset': dataset_type,
                    'split': split,
                    'n_codebooks': self.n_codebooks,
                    'n_groups': self.n_groups,
                    'results': results
                }, f, indent=2)
            print(f"Saved: {output_path}")

        return results

    def generate_examples(
        self,
        split: str = "val",
        n_samples: int = 5,
        output_dir: str = "./examples",
        dataset_type: Optional[str] = None,
        n_groups: Optional[int] = None,
        all_tiers: bool = False,
        save_original: bool = True
    ) -> None:
        """Generate example reconstructions from dataset samples (full length).

        Args:
            split: Dataset split to use.
            n_samples: Number of samples to generate.
            output_dir: Output directory for generated files.
            dataset_type: Dataset type (Jamendo, LibriTTS, MUSDB18).
            n_groups: Number of groups for bitrate control.
            all_tiers: If True, generate all bitrate tiers.
            save_original: If True, save the original/preprocessed audio files.
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
        output_dir = Path(output_dir) / dataset_type / split
        output_dir.mkdir(parents=True, exist_ok=True)

        tiers = list(range(1, self.n_groups + 1)) if all_tiers else [n_groups or self.n_groups]

        print(f"\nGenerating {n_samples} examples from {dataset_type} {split}")
        print(f"Tiers: {tiers}")
        print(f"Output directory: {output_dir}")
        print(f"Save original: {save_original}")
        print("-" * 60)

        with torch.no_grad():
            for i in tqdm(range(n_samples), desc="Generating examples"):
                # Get full audio path and load complete file
                track_index = dataset.tracks[i]
                audio_path = dataset.get_audio_path(track_index)
                signal = self._load_audio(audio_path)
                audio = signal.audio_data

                # Process full length audio with chunking
                chunk_samples = int(self.chunk_duration * self.sample_rate)

                if audio.shape[-1] <= chunk_samples:
                    out_full = self._forward(audio, self.n_groups)
                    preproc = out_full['preprocessed_audio'].cpu()
                    num_chunks = 1
                else:
                    # Get preprocessed audio by processing in chunks
                    preproc_chunks = []
                    num_chunks = (audio.shape[-1] + chunk_samples - 1) // chunk_samples
                    for j in range(num_chunks):
                        start, end = j * chunk_samples, min((j + 1) * chunk_samples, audio.shape[-1])
                        chunk = audio[:, :, start:end]
                        if chunk.shape[-1] < chunk_samples:
                            chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
                        out = self._forward(chunk, self.n_groups)
                        actual_len = end - start
                        preproc_chunks.append(out['preprocessed_audio'][:, :, :actual_len].cpu())
                    preproc = torch.cat(preproc_chunks, dim=-1)

                # Save original (preprocessed) - only if save_original is True
                if save_original:
                    original_signal = AudioSignal(preproc, self.sample_rate)
                    original_path = output_dir / f"sample_{i:03d}_original.wav"
                    original_signal.write(original_path)

                # Generate each tier
                for ng in tiers:
                    if audio.shape[-1] <= chunk_samples:
                        out = self._forward(audio, ng)
                        recons = out['audio'].cpu()
                    else:
                        # Process in chunks
                        recons_chunks = []
                        for j in range(num_chunks):
                            start, end = j * chunk_samples, min((j + 1) * chunk_samples, audio.shape[-1])
                            chunk = audio[:, :, start:end]
                            if chunk.shape[-1] < chunk_samples:
                                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
                            out = self._forward(chunk, ng)
                            actual_len = end - start
                            recons_chunks.append(out['audio'][:, :, :actual_len].cpu())
                        recons = torch.cat(recons_chunks, dim=-1)

                    metrics = self._compute_metrics(recons, preproc)

                    n_cb = self.model.groups_to_quantizers(ng)
                    bitrate = self.bitrate_tiers[ng]

                    # Save reconstructed audio
                    recons_signal = AudioSignal(recons, self.sample_rate)
                    recons_path = output_dir / f"sample_{i:03d}_{ng}g_{n_cb}cb_{bitrate:.0f}kbps.wav"
                    recons_signal.write(recons_path)

                    print(f"Sample {i}, {ng}g ({bitrate:.1f}kbps): SI-SDR={metrics['si_sdr']:.2f}dB, duration={signal.signal_duration:.2f}s")

                gc.collect()
                torch.cuda.empty_cache()

        print("-" * 60)
        print(f"Saved examples to {output_dir}")

    def analyze_bands(self, input_path: str, output_dir: Optional[str] = None) -> dict:
        signal = self._load_audio(input_path)
        audio = signal.audio_data[:, :, :int(self.chunk_duration * self.sample_rate)].to(self.device)

        analysis = self.model.analyze_bands(audio)
        info = analysis['band_info']

        print(f"\nBand Analysis: {Path(input_path).name}")
        print(f"Model: {self.n_codebooks} codebooks, {self.n_groups} groups")
        print(f"Baseline: {info['baseline']:.3f}")
        print("-" * 60)

        for i in range(self.n_groups):
            s, e = self.model.config.stage_groups[i]
            n_cb = e - s
            stats = analysis['group_stats'][i]
            print(f"Group {i} (codebooks {s+1}-{e}, {n_cb} cb): "
                  f"center={info['centers'][i]:.1f}, width={info['widths'][i]:.1f}, "
                  f"centroid={stats['centroid_normalized']:.3f}")

        if output_dir:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np

                Path(output_dir).mkdir(parents=True, exist_ok=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                x = np.arange(self.model.config.n_mels)

                for i in range(self.n_groups):
                    s, e = self.model.config.stage_groups[i]
                    label = f'Group {i} ({e-s} cb)'
                    ax1.plot(x, self.model.band_prioritizer.get_weights(i).detach().cpu().numpy(), label=label)
                    e_dist = analysis['energy_distributions'][i]
                    ax2.plot(x, e_dist / (e_dist.max() + 1e-8), label=label)

                ax1.set(xlabel='Mel bin', ylabel='Weight', title='Learned Band Priorities')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.set(xlabel='Mel bin', ylabel='Energy (normalized)', title='Frequency Contributions')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(Path(output_dir) / f"{Path(input_path).stem}_bands.png", dpi=150)
                plt.close()
                print(f"Saved plot: {Path(output_dir) / f'{Path(input_path).stem}_bands.png'}")

            except ImportError:
                print("matplotlib not available, skipping plot")

        return analysis


def main(args=None, config_path=None):
    p = argparse.ArgumentParser(description="Harp Inference - Variable Bitrate Audio Codec")
    p.add_argument("--ckpt", "--checkpoint", type=str, dest="checkpoint", help="Checkpoint path (or set checkpoint_path in config)")
    p.add_argument("--config", type=str, help="Config file path")
    p.add_argument("-i", "--input", type=str, help="Input audio file")
    p.add_argument("-o", "--output", type=str, help="Output path (file or directory)")
    p.add_argument("--n-groups", type=int, help="Number of groups for bitrate control (1-4)")

    # Modes
    p.add_argument("--eval-dataset", action="store_true", help="Evaluate on dataset")
    p.add_argument("--eval-tiers", action="store_true", help="Evaluate all bitrate tiers on single file")
    p.add_argument("--analyze", action="store_true", help="Analyze band specialization")
    p.add_argument("--generate-examples", action="store_true", help="Generate example reconstructions")
    p.add_argument("--n-samples", type=int, default=5, help="Number of examples to generate")
    p.add_argument("--no-original", action="store_true", help="Skip saving original/preprocessed audio files")

    # Dataset options
    p.add_argument("--split", type=str, default="val", help="Dataset split (e.g., train, val, test, test-clean, dev-other)")
    p.add_argument("--dataset", type=str, choices=["Jamendo", "LibriTTS", "MUSDB18"],
               help="Dataset type (overrides config)")
    p.add_argument("--max-batches", type=int, help="Max batches for quick eval")
    p.add_argument("--all-tiers", action="store_true", help="Evaluate all tiers on dataset")

    args = p.parse_args(args)

    # Config: prefer parameter from entry.py, then CLI arg
    cfg_path = config_path or args.config
    if not cfg_path or not Path(cfg_path).exists():
        p.error("--config required")

    # Resolve checkpoint: --ckpt, then config, then the default location.
    ckpt = args.checkpoint
    if not ckpt:
        with open(cfg_path) as f:
            ckpt = yaml.safe_load(f).get("checkpoint_path")
    if not ckpt and Path("checkpoints/harp.ckpt").exists():
        ckpt = "checkpoints/harp.ckpt"
    if not ckpt:
        p.error("no checkpoint found; pass --ckpt, place one at "
                "checkpoints/harp.ckpt, or set checkpoint_path in config")

    harp = HarpInference(ckpt, cfg_path)

    if args.eval_dataset:
        harp.evaluate_dataset(args.split, args.max_batches, args.all_tiers, args.output,
                              dataset_type=args.dataset)
    elif args.eval_tiers:
        if not args.input: p.error("--input required")
        harp.evaluate_tiers(args.input, args.output)
    elif args.analyze:
        if not args.input: p.error("--input required")
        harp.analyze_bands(args.input, args.output)
    elif args.generate_examples:
        harp.generate_examples(
            split=args.split,
            n_samples=args.n_samples,
            output_dir=args.output or "./examples",
            dataset_type=args.dataset,
            n_groups=args.n_groups,
            all_tiers=args.all_tiers,
            save_original=not args.no_original
        )
    else:
        if not args.input: p.error("--input required")
        output = args.output or f"{Path(args.input).stem}_reconstructed.wav"
        harp.process_file(args.input, output, args.n_groups)

    print("Done!")


if __name__ == "__main__":
    main()
