#!/usr/bin/env python
"""
Export a slim, inference-only checkpoint from a Lightning training checkpoint.

Training checkpoints saved by `entry.py -t` bundle optimizer state, LR
schedulers, the discriminator, and callback state, which makes them large
(~1.4 GB). Inference only needs the generator weights (the ``model.*`` keys)
plus the ``hyper_parameters`` used to rebuild the model. This script keeps just
those, shrinking the checkpoint to a few hundred MB.

The result stays compatible with the inference scripts, which read
``checkpoint["state_dict"]`` (filtering the ``model.`` prefix) and, for HARP,
``checkpoint["hyper_parameters"]``.

Usage:
    python scripts/export_weights.py in.ckpt out.ckpt
    python scripts/export_weights.py in.ckpt out.ckpt --half   # store fp16 weights
"""
import argparse
from pathlib import Path

import torch


def export(in_path: str, out_path: str, half: bool = False) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError(
            f"{in_path} has no 'state_dict'; is this a Lightning training checkpoint?"
        )

    # Keep only the generator weights (drop discriminator / optimizer / etc.).
    model_sd = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    if not model_sd:
        raise ValueError("No 'model.*' weights found in the checkpoint state_dict.")

    if half:
        model_sd = {
            k: (v.half() if torch.is_floating_point(v) else v)
            for k, v in model_sd.items()
        }

    slim = {
        "state_dict": model_sd,
        # Required by HARP inference to rebuild the model architecture.
        "hyper_parameters": ckpt.get("hyper_parameters", {}),
    }
    if "pytorch-lightning_version" in ckpt:
        slim["pytorch-lightning_version"] = ckpt["pytorch-lightning_version"]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_path)

    in_mb = Path(in_path).stat().st_size / 1e6
    out_mb = Path(out_path).stat().st_size / 1e6
    print(f"Exported {len(model_sd)} weight tensors")
    print(f"  {in_path}  ({in_mb:.0f} MB)")
    print(f"  -> {out_path}  ({out_mb:.0f} MB){'  [fp16]' if half else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export inference-only checkpoint")
    p.add_argument("input", help="Path to a Lightning training .ckpt")
    p.add_argument("output", help="Path for the slim inference .ckpt")
    p.add_argument("--half", action="store_true",
                   help="Store floating-point weights in fp16 (halves size)")
    args = p.parse_args()
    export(args.input, args.output, half=args.half)


if __name__ == "__main__":
    main()
