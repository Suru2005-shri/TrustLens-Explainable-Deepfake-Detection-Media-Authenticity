"""
TrustLens - Fine-tune a vision transformer for deepfake classification.

Fine-tunes a pretrained ViT on your real/fake image-folder dataset
(see prepare_data.py). Uses HuggingFace Trainer, so checkpointing, logging,
and mixed precision are handled for you.

Usage:
    python train.py \
        --data_dir data \
        --base_model google/vit-base-patch16-224 \
        --output_dir checkpoints/trustlens-vit \
        --epochs 5 \
        --batch_size 16

Needs a GPU for reasonable speed (Colab T4 / Kaggle GPU is enough for a
hackathon-scale dataset of a few thousand frames). CPU-only will work but
is slow.
"""
import argparse

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data",
                     help="expects data_dir/train/{real,fake} and data_dir/val/{real,fake}")
    ap.add_argument("--base_model", default="google/vit-base-patch16-224")
    ap.add_argument("--output_dir", default="checkpoints/trustlens-vit")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    # ImageFolder auto-labels from subfolder names (fake=0, real=1 or similar,
    # check dataset.features["label"].names after load to confirm).
    dataset = load_dataset(
        "imagefolder",
        data_dir=args.data_dir,
    )
    print("Classes:", dataset["train"].features["label"].names)

    processor = AutoImageProcessor.from_pretrained(args.base_model)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    dataset = dataset.with_transform(transform)

    labels = dataset["train"].features["label"].names
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=refs)
        f1 = f1_metric.compute(predictions=preds, references=refs, average="binary")
        return {**acc, **f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=True,  # set False if no GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nDone. Fine-tuned model saved to: {args.output_dir}")
    print("Point classifier.py's MODEL_ID at this local path to use it in the app.")


if __name__ == "__main__":
    main()
