"""
TrustLens - Pretrained deepfake classifier wrapper.

Uses a HuggingFace image-classification checkpoint fine-tuned for
deepfake/AI-generated image detection. Requires internet access + torch +
transformers in YOUR environment (this sandbox has no network, so this
module is untested here but is standard `transformers` usage).

Swap MODEL_ID for whichever checkpoint you validate best on FaceForensics++ /
Celeb-DF / DFDC. A few solid public options as of early 2026:
  - "prithivMLmods/Deepfake-Detect-Siglip2"
  - "dima806/deepfake_vs_real_image_detection"
  - "Wvolf/ViT_Deepfake_Detection"

Try 2-3 on your validation split and pick the best AUC before the deadline.
"""
from PIL import Image

MODEL_ID = "dima806/deepfake_vs_real_image_detection"

_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("image-classification", model=MODEL_ID)
    return _pipeline


def classify_image(image: Image.Image) -> dict:
    """
    Returns suspicion_score in [0,1] where 1 = confidently AI-generated/fake.
    Falls back gracefully with a clear error if transformers/torch aren't
    installed, so the rest of the pipeline still runs in dev/demo mode.
    """
    try:
        clf = _load_pipeline()
    except Exception as e:
        return {
            "signal": "pretrained_classifier",
            "suspicion_score": None,
            "error": f"classifier unavailable: {e}",
            "note": "pip install torch transformers, then re-run",
        }

    results = clf(image.convert("RGB"))
    # results like [{'label': 'Fake', 'score': 0.93}, {'label': 'Real', 'score': 0.07}]
    fake_score = 0.0
    for r in results:
        label = r["label"].lower()
        if "fake" in label or "ai" in label or "synthetic" in label:
            fake_score = max(fake_score, r["score"])

    return {
        "signal": "pretrained_classifier",
        "suspicion_score": round(float(fake_score), 4),
        "raw_results": results,
    }
