"""
TrustLens - Dataset preparation.

Converts video-based deepfake datasets (FaceForensics++, Celeb-DF, DFDC) into
a flat image-folder structure that train.py / HuggingFace ImageFolder can
consume directly:

    data/
      train/
        real/  *.jpg
        fake/  *.jpg
      val/
        real/  *.jpg
        fake/  *.jpg

Usage:
    python prepare_data.py \
        --real_videos_dir /path/to/FaceForensics++/original_sequences \
        --fake_videos_dir /path/to/FaceForensics++/manipulated_sequences \
        --out_dir data \
        --frames_per_video 10 \
        --val_split 0.15
"""
import argparse
import os
import random
from pathlib import Path

import cv2


def extract_frames(video_path: str, out_dir: Path, prefix: str, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    step = max(1, total // n_frames)
    saved = 0
    idx = 0
    while cap.isOpened() and saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = out_dir / f"{prefix}_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def process_class(videos_dir: str, label: str, out_root: Path,
                   frames_per_video: int, val_split: float, exts=(".mp4", ".avi", ".mov")):
    videos = [str(p) for p in Path(videos_dir).rglob("*") if p.suffix.lower() in exts]
    random.shuffle(videos)
    n_val = int(len(videos) * val_split)
    val_videos = set(videos[:n_val])

    total_frames = 0
    for v in videos:
        split = "val" if v in val_videos else "train"
        out_dir = out_root / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = Path(v).stem
        total_frames += extract_frames(v, out_dir, prefix, frames_per_video)

    print(f"[{label}] {len(videos)} videos -> {total_frames} frames "
          f"({n_val} videos held out for val)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_videos_dir", required=True)
    ap.add_argument("--fake_videos_dir", required=True)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--frames_per_video", type=int, default=10)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_dir)

    process_class(args.real_videos_dir, "real", out_root, args.frames_per_video, args.val_split)
    process_class(args.fake_videos_dir, "fake", out_root, args.frames_per_video, args.val_split)

    print(f"\nDone. Dataset ready at: {out_root.resolve()}")
    print("Folder structure:")
    print(f"  {out_root}/train/real, {out_root}/train/fake")
    print(f"  {out_root}/val/real,   {out_root}/val/fake")


if __name__ == "__main__":
    main()
