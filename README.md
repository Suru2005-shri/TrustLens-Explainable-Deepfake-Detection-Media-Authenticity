# TrustLens — Multi-Signal Deepfake & Media Authenticity Verification

Built for Brainwave 2026, Problem Statement 1 (Deepfake & Digital Media Authenticity Verification).

## Why this design

Most hackathon deepfake detectors ship a single black-box classifier score.
TrustLens fuses **4 independent signals** so a failure in one (e.g. a
generator architecture the classifier was never trained on) doesn't sink the
whole verdict — and every prediction comes with an explainable breakdown,
which is what actually earns trust from judges and end users.

| Signal | What it catches | Cost |
|---|---|---|
| Pretrained classifier (ViT/Xception, HF) | Learned deepfake artifacts | needs torch+transformers+download |
| Error Level Analysis (ELA) | Splicing / recompression mismatches | free, offline, instant |
| Frequency-domain (FFT) analysis | GAN/diffusion upsampling checkerboard artifacts | free, offline, instant |
| Temporal consistency (video) | Frame-to-frame flicker, unnatural optical flow | free, offline, instant |

Fusion layer: weighted ensemble → single 0–100% authenticity confidence +
per-signal breakdown card.

## Quickstart (using off-the-shelf pretrained classifier, no training needed)

```bash
pip install -r requirements.txt
streamlit run app.py
```

First image classification will download the HF checkpoint (~90MB) — do this
**before** your demo, not live in front of judges.

---

## Full build + training, step by step

Do this if you want your own fine-tuned checkpoint instead of (or in
addition to) the off-the-shelf model — stronger for judges, since "we
fine-tuned on FaceForensics++ ourselves" beats "we called a HF pipeline."
Needs a GPU (Colab T4 free tier is enough); CPU works but is slow.

**1. Get a dataset.**
Download [FaceForensics++](https://github.com/ondyari/FaceForensics) (request
access, academic use) or [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
(open download). You want the raw video files split into real ("original
sequences") and fake ("manipulated sequences") folders. For a hackathon-scale
run, ~200-500 videos per class is plenty.

**2. Extract frames into an image-folder dataset.**
```bash
python prepare_data.py \
    --real_videos_dir /path/to/original_sequences \
    --fake_videos_dir /path/to/manipulated_sequences \
    --out_dir data \
    --frames_per_video 10 \
    --val_split 0.15
```
This gives you `data/train/{real,fake}` and `data/val/{real,fake}` — a few
thousand JPEGs, ready for HF's `imagefolder` loader.

**3. Fine-tune the classifier.**
```bash
python train.py \
    --data_dir data \
    --base_model google/vit-base-patch16-224 \
    --output_dir checkpoints/trustlens-vit \
    --epochs 5 \
    --batch_size 16
```
5 epochs on ~5-10k frames takes roughly 20-40 min on a Colab T4. Watch the
eval F1 in the logs — if it's not climbing past epoch 2-3, drop `--lr` to
`2e-5` or check your dataset labels loaded correctly (printed at startup).

**4. Evaluate it properly (don't just trust training-loop numbers).**
```bash
python evaluate.py --model_path checkpoints/trustlens-vit --data_dir data/val
```
Gives you accuracy, AUC, and a full precision/recall report. This is also
how you A/B the off-the-shelf checkpoint vs. your fine-tuned one — run both
through this script on the same val set and keep whichever wins.

**5. Point the app at your fine-tuned model.**
In `classifier.py`, change:
```python
MODEL_ID = "checkpoints/trustlens-vit"   # local path instead of HF hub id
```

**6. Calibrate the offline forensic signals (ELA + FFT) on the same data.**
Run `error_level_analysis()` / `frequency_domain_analysis()` from
`forensics.py` against `data/val/real` and `data/val/fake`, and adjust the
`suspicion = ...` formulas until real images score consistently low. This
step matters as much as training the classifier — it's what makes the
"multi-signal" story in your deck actually true rather than decorative.

**7. Re-run the full Streamlit app end-to-end** with real images/videos
before your demo slot, including the video tab (temporal signal needs an
actual multi-frame clip to test, not a still image).

## Validating the classifier before submission

`classifier.py` defaults to `dima806/deepfake_vs_real_image_detection`. Two
backup checkpoints are listed in that file's docstring — pull FaceForensics++
or Celeb-DF sample images, run all 3, and keep whichever gets best accuracy
on a quick 20-image manual spot check. Don't skip this; a mismatched label
scheme (e.g. "0"/"1" instead of "Real"/"Fake") will silently break
`classify_image`'s label matching.

## Calibration note

The forensic thresholds in `forensics.py` (hotspot_ratio, hf_ratio,
angular_variance cutoffs) were smoke-tested on synthetic noise, **not**
tuned on real photos. Before the demo, run both signals against ~10 known
real photos and ~10 known FaceForensics++ fakes, and adjust the suspicion
formulas in `error_level_analysis()` / `frequency_domain_analysis()` so real
photos score low. This is the single highest-leverage thing to do with
remaining time.

## Honest limitation (state this explicitly in your deck)

No detector generalizes perfectly to generator architectures absent from
training/validation data. TrustLens' forensic signals (ELA, FFT) are
architecture-agnostic and help cover this gap, but the confidence score
should be framed as decision support, not ground truth — this honesty is a
stronger judge-facing position than an overclaimed "99% accuracy" slide.

## Phase 2 roadmap (for your deck, not required for submission)

- Audio deepfake detection (ASVspoof2019 + RawNet2)
- Browser extension for pre-share checks
- C2PA content-provenance verification chain

## Files

- `forensics.py` — ELA, FFT, temporal signals (pure numpy/PIL/cv2, no network needed)
- `classifier.py` — pretrained/fine-tuned HF classifier wrapper
- `prepare_data.py` — video → labeled image-folder dataset extraction
- `train.py` — fine-tuning script (HF Trainer, ViT base)
- `evaluate.py` — accuracy/AUC/report on a val set, for model selection
- `fusion.py` — weighted ensemble + verdict logic
- `app.py` — Streamlit demo UI
- `sample_data/` — synthetic smoke-test images (replace with real FaceForensics++ samples for the actual demo)
