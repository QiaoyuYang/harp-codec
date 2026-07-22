<p align="center">
  <img src="docs/assets/hero.webp" width="720"
       alt="A harp with sound waves streaming from its strings; highlighted orange strands represent HARP's prioritized frequency bands">
</p>

# HARP: Harmonic-Aware Residual Partitioning for Neural Audio Codecs

[![arXiv](https://img.shields.io/badge/arXiv-2607.16657-b31b1b.svg)](https://arxiv.org/abs/2607.16657)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffcc4d.svg)](https://huggingface.co/KelvinYang/harp-codec)
[![Demo](https://img.shields.io/badge/%F0%9F%94%8A%20Audio-Samples-brightgreen.svg)](https://qiaoyuyang.github.io/harp-codec/)

HARP is a neural audio codec that partitions residual vector quantization across
harmonically meaningful frequency bands to achieve high-quality, variable-bitrate
audio compression. The key innovation is a harmonic-aware partitioning that guides
the model to distribute codebook capacity across perceptually meaningful frequency
bands, so a single model serves multiple bitrates by decoding a growing number of
codebook groups.

This repository provides the official code for training and evaluating HARP.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and FFmpeg (for audio decoding):

- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- FFmpeg: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)

```bash
git clone https://github.com/QiaoyuYang/harp-codec.git && cd harp-codec
uv sync                    # create .venv and install dependencies
source .venv/bin/activate
```

## Pretrained models

Pretrained HARP weights are on the [Hugging Face Hub](https://huggingface.co/KelvinYang/harp-codec).
Download the checkpoint to the default location:

```bash
uv pip install "huggingface_hub[cli]"
hf download KelvinYang/harp-codec harp.ckpt --local-dir checkpoints
```

## Usage

### Inference on a single audio file

Reconstruct any audio file (wav/mp3/flac/...):

```bash
# full-rate reconstruction
python entry.py -i --input path/to/audio.wav --output recon.wav

# lower bitrate: fewer groups (1..4)
python entry.py -i --input path/to/audio.wav --n-groups 2 --output recon_2g.wav
```

Each run prints SI-SDR, multi-scale mel loss, LSD, and SNR against the input, and
writes the reconstructed audio to `--output`.

### Evaluate all bitrate tiers on one file

```bash
# metrics for every group tier
python entry.py -i --input audio.wav --eval-tiers
```

### Evaluate on a dataset

Point the config's `data.dataset_root` (or `--dataset`) at a prepared dataset split:

```bash
# across all tiers, saving a JSON summary
python entry.py -i --eval-dataset --all-tiers \
  --dataset Jamendo --split val --output results_harp.json
```

Supported dataset types: `Jamendo`, `LibriTTS`, `MUSDB18`. Reported metrics are
SI-SDR, multi-scale mel loss, LSD, and SNR.

### Training

```bash
python entry.py -t --model harp --config harp/configs/train_harp.yaml
```

Before training, edit the config and set `data.dataset_root` to your audio directory
and `train.logdir` to where checkpoints/logs should go. Training uses PyTorch Lightning;
checkpoints and TensorBoard logs are written under `logdir`.

After training, export an inference checkpoint with `scripts/export_weights.py`. It
keeps only the model weights, dropping the optimizer, discriminator, and callback state:

```bash
python scripts/export_weights.py path/to/harp-epoch=NN.ckpt checkpoints/harp.ckpt
```

## Command-line arguments

| Argument | Description |
|----------|-------------|
| `-t`, `--train` | Run the training pipeline |
| `-i`, `--infer` | Run the inference pipeline |
| `--model` | `harp` (default) or `dac` |
| `-c`, `--config` | Path to a YAML config |
| `--ckpt` / `--checkpoint` | Trained checkpoint (or set `checkpoint_path` in the config) |
| `--input` | Input audio file for single-file inference |
| `--output` | Output path (file or directory) |
| `--n-groups` | HARP bitrate control: number of groups (1–4) |

## Bitrate tiers

With 9 codebooks @ 1024 entries and a ~86 Hz frame rate:

| HARP groups | Codebooks | Approx. bitrate |
|-------------|-----------|-----------------|
| 1 | 3 | ~2.6 kbps |
| 2 | 5 | ~4.3 kbps |
| 3 | 7 | ~6.0 kbps |
| 4 | 9 | ~7.7 kbps (full) |

## Citation

```bibtex
@inproceedings{harp2026,
  title     = {HARP: Harmonic-Aware Residual Partitioning for Neural Audio Codecs},
  author    = {Yang, Qiaoyu and He, Lixing and Deng, Binyue and Zhao, Weifeng},
  booktitle = {Interspeech},
  year      = {2026}
}
```

## Acknowledgments

This project builds upon the
[Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec)
and uses the [`audiotools`](https://github.com/descriptinc/audiotools) library for
audio processing.

## License

MIT
