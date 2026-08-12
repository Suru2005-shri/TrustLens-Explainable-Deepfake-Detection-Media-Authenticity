# TrustLens — Project Documentation

**Brainwave 2026 · Problem Statement 1: Deepfake & Digital Media Authenticity Verification**

---

## 1. Problem

Generative AI has made synthetic images, videos, and voices cheap to produce
and hard to distinguish from real media. This threatens journalism, elections,
public safety, and everyday digital trust. Most existing detectors give a
single opaque score with no explanation — which makes them hard to trust and
easy to game as generators improve.

## 2. Our approach

TrustLens verifies images and videos using **four independent forensic
signals** instead of one black-box model, then fuses them into a single
confidence score with a visible breakdown:

| # | Signal | Detects | Mechanism |
|---|---|---|---|
| 1 | Pretrained deepfake classifier | Learned generative artifacts | Fine-tuned ViT / HF vision transformer |
| 2 | Error Level Analysis (ELA) | Splicing, recompression mismatches | JPEG re-save difference map |
| 3 | Frequency-domain (FFT) analysis | GAN/diffusion upsampling artifacts | Spectral energy + angular periodicity |
| 4 | Temporal consistency (video) | Frame-to-frame flicker | Optical flow + frame-diff jitter |

**Why four signals, not one:** a generator architecture absent from the
classifier's training data will fool it — but it still leaves frequency and
compression artifacts a forensic signal can catch, and vice versa. No single
signal is trusted to carry the whole verdict.

### Fusion

Each signal outputs a suspicion score in [0, 1]. A weighted ensemble
(weights renormalized to whatever signals are actually available — e.g. no
temporal score for a still image) produces:

- A final **authenticity confidence %**
- A verdict: `LIKELY AUTHENTIC` / `UNCERTAIN — REVIEW RECOMMENDED` / `LIKELY MANIPULATED`
- A **per-signal breakdown** so the user (and judges) can see *why*, not just *what*

This explainability is the core differentiator — see Section 5.

## 3. System architecture

```
                    ┌─────────────────┐
   Image / Video →  │   Streamlit UI   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┬───────────────┐
              ▼              ▼              ▼               ▼
      ┌───────────┐  ┌──────────────┐ ┌───────────┐ ┌────────────────┐
      │ Pretrained │  │     ELA      │ │    FFT    │ │ Temporal (video)│
      │ classifier │  │ (forensics.py)│ │(forensics)│ │  (forensics.py) │
      └─────┬──────┘  └──────┬───────┘ └─────┬─────┘ └────────┬────────┘
            │                │               │                │
            └────────────────┴───────┬───────┴────────────────┘
                                      ▼
                            ┌───────────────────┐
                            │   Fusion (fusion.py) │
                            │  weighted ensemble  │
                            └─────────┬───────────┘
                                      ▼
                     Verdict + Confidence % + Signal Breakdown
```

## 4. Tech stack

- **Model:** Vision Transformer (ViT-base), fine-tuned via HuggingFace `transformers`
- **Forensics:** NumPy, SciPy, OpenCV, PIL (all offline, no model weights needed)
- **Training:** HF `Trainer` API, `datasets` for imagefolder loading, `evaluate` for metrics
- **UI:** Streamlit
- **Data:** Kaggle "140k Real and Fake Faces" / FaceForensics++ / Celeb-DF (any real/fake image or video corpus works via the provided prep scripts)

## 5. What makes this defensible in a viva / Q&A

- **Explainability over black-box scoring.** Judges can ask "why did it flag
  this?" and get an actual answer (ELA hotspot map, FFT periodicity, flow
  jitter) — not just "the model said so."
- **Architecture-agnostic fallback.** ELA and FFT don't depend on having seen
  a given generator during training, unlike the classifier alone.
- **Honest limitation, stated upfront:** no detector — ours included —
  generalizes perfectly to generator architectures absent from
  training/validation data. We treat the confidence score as decision
  support, not ground truth. This is *more* credible to technical judges
  than an unqualified accuracy claim.
- **Reproducible pipeline:** `prepare_data.py` / `reorganize_images.py` →
  `train.py` → `evaluate.py` is a clean, scripted path from raw data to a
  validated checkpoint — not a notebook that only ran once.

## 6. Results (fill in after calibration — see README.md step 6)

| Signal | Accuracy | AUC | Notes |
|---|---|---|---|
| Pretrained classifier | _pending_ | _pending_ | run `evaluate.py` |
| ELA (calibrated) | _pending_ | — | threshold-based, not probabilistic |
| FFT (calibrated) | _pending_ | — | threshold-based, not probabilistic |
| Fused ensemble | _pending_ | _pending_ | on held-out val set |

*(Run the evaluation step and paste real numbers here before submitting —
an empty results table is a weaker slide than a small honest one.)*

## 7. Roadmap

- **Phase 1 (this submission):** image + video upload, 4-signal fusion, explainable UI — solo-executable in the hackathon window
- **Phase 2:** audio deepfake detection (ASVspoof2019 + RawNet2 architecture)
- **Phase 3:** browser extension for pre-share checks; C2PA content-provenance chain verification for end-to-end media trust

## 8. Repository structure

```
trustlens/
├── app.py                  # Streamlit demo UI
├── forensics.py             # ELA, FFT, temporal signal extractors (offline)
├── classifier.py             # Pretrained/fine-tuned HF classifier wrapper
├── fusion.py                 # Weighted ensemble + verdict logic
├── prepare_data.py           # Video → labeled image-folder dataset
├── reorganize_images.py      # Pre-split image datasets → train/val structure
├── train.py                  # Fine-tuning script (HF Trainer, ViT)
├── evaluate.py                # Accuracy/AUC/report on a val set
├── tests/
│   └── test_forensics.py     # Smoke tests for offline signals
├── requirements.txt
├── README.md                  # Setup + full training walkthrough
└── PROJECT_DOCUMENTATION.md   # This file
```

## 9. Team / submission notes

Solo submission. Problem Statement 1. All code in this repository is
original; the pretrained classifier backbone is a public HuggingFace
checkpoint, fine-tuned on our own train/val split as described in README.md.
