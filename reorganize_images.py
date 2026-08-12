"""
TrustLens - Reorganize an already-split real/fake image dataset (e.g. Kaggle's
"140k Real and Fake Faces") into the train/{real,fake} + val/{real,fake}
structure that train.py / HF imagefolder expects.

Use this INSTEAD of prepare_data.py when you already have real/fake images
(not raw videos) — no frame extraction needed.

Usage:
    python reorganize_images.py \
        --real_dir data_raw/real_vs_fake/real-vs-fake/train/real \
        --fake_dir data_raw/real_vs_fake/real-vs-fake/train/fake \
        --out_dir data \
        --val_split 0.15 \
        --max_per_class 3000   # optional cap, keeps training fast for a hackathon
"""
import argparse
import random
import shutil
from pathlib import Path


def process_class(src_dir, label, out_root, val_split, max_per_class, exts=(".jpg", ".jpeg", ".png")):
    files = [p for p in Path(src_dir).rglob("*") if p.suffix.lower() in exts]
    random.shuffle(files)
    if max_per_class:
        files = files[:max_per_class]

    n_val = int(len(files) * val_split)
    val_files = set(files[:n_val])

    for f in files:
        split = "val" if f in val_files else "train"
        out_dir = out_root / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out_dir / f.name)

    print(f"[{label}] {len(files)} images copied ({n_val} to val)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--max_per_class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_dir)

    process_class(args.real_dir, "real", out_root, args.val_split, args.max_per_class)
    process_class(args.fake_dir, "fake", out_root, args.val_split, args.max_per_class)

    print(f"\nDone. Dataset ready at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
