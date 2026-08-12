"""
TrustLens - Core forensic signal extractors
Signals 2 & 3 (frequency-domain + ELA) run fully offline, no model weights needed.
Signal 4 (temporal) needs a video (multi-frame) input.
"""
import io
import numpy as np
from PIL import Image, ImageChops
import cv2


# ---------------------------------------------------------------------------
# Signal A: Error Level Analysis (ELA)
# Detects recompression / splicing inconsistencies. Real photos compress
# uniformly; edited/spliced regions re-compress differently and light up.
# ---------------------------------------------------------------------------
def error_level_analysis(image: Image.Image, quality: int = 90) -> dict:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    ela_image = ImageChops.difference(image, resaved)
    ela_array = np.array(ela_image).astype(np.float32)

    max_diff = ela_array.max() if ela_array.max() > 0 else 1.0
    scale = 255.0 / max_diff
    ela_array = (ela_array * scale).clip(0, 255).astype(np.uint8)

    mean_diff = float(ela_array.mean())
    std_diff = float(ela_array.std())
    # High local variance / hot patches = suspicious splice regions
    hotspot_ratio = float((ela_array > 60).mean())

    # Heuristic score: uniform low-level noise = likely authentic,
    # patchy high-contrast regions = likely manipulated.
    suspicion = min(1.0, (hotspot_ratio * 3.0) + (std_diff / 120.0))

    return {
        "signal": "error_level_analysis",
        "suspicion_score": round(suspicion, 4),
        "mean_diff": round(mean_diff, 3),
        "std_diff": round(std_diff, 3),
        "hotspot_ratio": round(hotspot_ratio, 4),
        "ela_image": Image.fromarray(ela_array),
    }


# ---------------------------------------------------------------------------
# Signal B: Frequency-domain (FFT) analysis
# GAN/diffusion upsampling leaves periodic checkerboard artifacts in the
# high-frequency spectrum that real camera sensors don't produce.
# ---------------------------------------------------------------------------
def frequency_domain_analysis(image: Image.Image) -> dict:
    gray = np.array(image.convert("L")).astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius_low = min(h, w) // 8
    radius_high = min(h, w) // 3

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    low_band = magnitude[dist <= radius_low]
    mid_band = magnitude[(dist > radius_low) & (dist <= radius_high)]
    high_band = magnitude[dist > radius_high]

    low_energy = float(low_band.mean())
    high_energy = float(high_band.mean()) if high_band.size else 0.0
    hf_ratio = high_energy / (low_energy + 1e-6)

    # Periodicity check: GAN upsampling creates regular grid peaks.
    # Look at azimuthal variance in the mid band — real photos are
    # roughly isotropic, synthetic upsampling shows angular peaks.
    angles = np.arctan2(Y - cy, X - cx)
    mid_mask = (dist > radius_low) & (dist <= radius_high)
    angle_bins = np.linspace(-np.pi, np.pi, 36)
    bin_idx = np.digitize(angles[mid_mask], angle_bins)
    bin_means = [magnitude[mid_mask][bin_idx == i].mean()
                 for i in range(1, len(angle_bins)) if (bin_idx == i).any()]
    angular_variance = float(np.var(bin_means)) if bin_means else 0.0

    suspicion = min(1.0, (hf_ratio * 1.5) + (angular_variance / 4.0))

    return {
        "signal": "frequency_domain_analysis",
        "suspicion_score": round(suspicion, 4),
        "high_freq_ratio": round(hf_ratio, 4),
        "angular_variance": round(angular_variance, 4),
    }


# ---------------------------------------------------------------------------
# Signal C: Temporal consistency (video only)
# Real faces blink/move smoothly; GAN-generated frames flicker slightly
# frame-to-frame in ways optical flow + frame-diff can pick up.
# ---------------------------------------------------------------------------
def temporal_consistency_analysis(frames: list) -> dict:
    if len(frames) < 3:
        return {
            "signal": "temporal_consistency",
            "suspicion_score": 0.0,
            "note": "insufficient frames (need >=3)",
        }

    grays = [cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2GRAY) for f in frames]
    flow_mags = []
    frame_diffs = []

    for i in range(1, len(grays)):
        prev, curr = grays[i - 1], grays[i]
        prev = cv2.resize(prev, (256, 256))
        curr = cv2.resize(curr, (256, 256))

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(float(mag.mean()))

        diff = cv2.absdiff(prev, curr)
        frame_diffs.append(float(diff.mean()))

    flow_std = float(np.std(flow_mags))
    diff_jitter = float(np.std(frame_diffs)) / (float(np.mean(frame_diffs)) + 1e-6)

    # High frame-to-frame jitter relative to overall motion = suspicious flicker
    suspicion = min(1.0, diff_jitter / 2.0 + flow_std / 10.0)

    return {
        "signal": "temporal_consistency",
        "suspicion_score": round(suspicion, 4),
        "flow_std": round(flow_std, 4),
        "diff_jitter": round(diff_jitter, 4),
        "frames_analyzed": len(frames),
    }
