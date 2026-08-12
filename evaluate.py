"""
TrustLens - Evaluate a (fine-tuned or off-the-shelf) classifier on your val set.
Also useful for picking between the 2-3 candidate checkpoints listed in
classifier.py before you commit to one.

Usage:
    python evaluate.py --model_path checkpoints/trustlens-vit --data_dir data/val
"""
import argparse
from pathlib import Path

from PIL import Image
from transformers import pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                     help="local checkpoint dir or HF hub model id")
    ap.add_argument("--data_dir", required=True,
                     help="expects data_dir/real/*.jpg and data_dir/fake/*.jpg")
    args = ap.parse_args()

    clf = pipeline("image-classification", model=args.model_path)

    y_true, y_score, y_pred = [], [], []

    for label, true_val in [("real", 0), ("fake", 1)]:
        folder = Path(args.data_dir) / label
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            results = clf(img)
            fake_score = 0.0
            for r in results:
                if "fake" in r["label"].lower() or "ai" in r["label"].lower():
                    fake_score = max(fake_score, r["score"])
            y_true.append(true_val)
            y_score.append(fake_score)
            y_pred.append(1 if fake_score >= 0.5 else 0)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    print(f"\nEvaluated on {len(y_true)} images from {args.data_dir}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
