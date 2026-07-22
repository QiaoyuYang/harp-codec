#!/usr/bin/env python
"""
Generate audio samples for the demo page (docs/).

For each input audio file, copies the original and writes HARP reconstructions at
each bitrate tier into docs/audio/<name>/, then updates docs/audio/manifest.json
which the demo page reads to build the comparison players.

Usage:
    python scripts/build_demo.py path/to/*.wav
    python scripts/build_demo.py a.wav b.mp3 --groups 1 2 3 4
    python scripts/build_demo.py samples/*.wav --ckpt checkpoints/harp.ckpt
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

# Run from anywhere: make the repo root importable so `import harp` works.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harp.infer_harp import HarpInference

# Approximate bitrate per group tier (9 codebooks @ 1024 entries, ~86 Hz).
TIER_KBPS = {1: "~2.6 kbps", 2: "~4.3 kbps", 3: "~6.0 kbps", 4: "~7.7 kbps"}


def main() -> None:
    p = argparse.ArgumentParser(description="Build demo audio samples for docs/")
    p.add_argument("inputs", nargs="+", help="Input audio files (wav/mp3/flac/...)")
    p.add_argument("--ckpt", default="checkpoints/harp.ckpt", help="HARP checkpoint")
    p.add_argument("--config", default="harp/configs/train_harp.yaml", help="Config file")
    p.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Group tiers to render (1..4)")
    p.add_argument("--out", default="docs/audio", help="Output directory")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    harp = HarpInference(args.ckpt, args.config)

    tiers = [{"key": "orig", "label": "Original"}]
    for g in args.groups:
        tiers.append({"key": f"g{g}",
                      "label": f"HARP · {g} group{'s' if g > 1 else ''} · {TIER_KBPS.get(g, '')}".strip(" ·")})

    samples = []
    for inp in args.inputs:
        inp = Path(inp)
        if not inp.exists():
            print(f"skip (not found): {inp}")
            continue
        stem = inp.stem
        sdir = out_root / stem
        sdir.mkdir(parents=True, exist_ok=True)

        files = {}
        orig_name = f"original{inp.suffix}"
        shutil.copy2(inp, sdir / orig_name)
        files["orig"] = orig_name

        for g in args.groups:
            out_wav = sdir / f"harp_{g}g.wav"
            harp.process_file(str(inp), str(out_wav), g)
            files[f"g{g}"] = out_wav.name

        samples.append({"name": stem, "dir": stem, "files": files})
        print(f"  built {len(files)} tracks for {stem}")

    manifest = {"tiers": tiers, "samples": samples}
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_root / 'manifest.json'} ({len(samples)} sample(s))")


if __name__ == "__main__":
    main()
