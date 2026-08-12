"""
TrustLens - Signal fusion.
Weighted ensemble -> single authenticity confidence score + explainable breakdown.
"""

# Weights sum to 1.0. Pretrained classifier gets the most trust once it's
# wired up and validated; forensic signals catch cases the classifier misses
# (out-of-distribution generators) and vice versa.
DEFAULT_WEIGHTS = {
    "pretrained_classifier": 0.45,
    "error_level_analysis": 0.20,
    "frequency_domain_analysis": 0.20,
    "temporal_consistency": 0.15,  # only applies to video; renormalized if absent
}


def fuse_signals(signal_results: list, weights: dict = None) -> dict:
    weights = dict(weights or DEFAULT_WEIGHTS)

    usable = [r for r in signal_results if r.get("suspicion_score") is not None]
    if not usable:
        return {
            "final_verdict": "UNKNOWN",
            "confidence_authentic_pct": None,
            "error": "no usable signals",
        }

    # Renormalize weights over signals actually present (e.g. no video -> no temporal)
    present = {r["signal"]: r["suspicion_score"] for r in usable}
    active_weights = {k: v for k, v in weights.items() if k in present}
    total_weight = sum(active_weights.values()) or 1.0
    active_weights = {k: v / total_weight for k, v in active_weights.items()}

    weighted_suspicion = sum(present[k] * active_weights[k] for k in active_weights)
    confidence_authentic_pct = round((1.0 - weighted_suspicion) * 100, 1)

    if confidence_authentic_pct >= 80:
        verdict = "LIKELY AUTHENTIC"
    elif confidence_authentic_pct >= 55:
        verdict = "UNCERTAIN — REVIEW RECOMMENDED"
    else:
        verdict = "LIKELY MANIPULATED / AI-GENERATED"

    breakdown = [
        {
            "signal": r["signal"],
            "suspicion_score": r["suspicion_score"],
            "weight_used": round(active_weights.get(r["signal"], 0.0), 3),
        }
        for r in usable
    ]

    return {
        "final_verdict": verdict,
        "confidence_authentic_pct": confidence_authentic_pct,
        "weighted_suspicion": round(weighted_suspicion, 4),
        "signal_breakdown": breakdown,
    }
