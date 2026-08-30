"""
Stage 5 metric utilities: calibration curve / ECE and stratified bootstrap.
Nothing here fits or tunes anything — pure evaluation on already-scored
predictions.
"""

import numpy as np


def calibration_curve_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Equal-width probability bins over [0, 1]. Returns (bins, ece)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                "count": 0, "predicted_mean": None, "observed_rate": None,
            })
            continue
        pred_mean = float(y_prob[mask].mean())
        obs_rate = float(y_true[mask].mean())
        bins.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "count": count, "predicted_mean": pred_mean, "observed_rate": obs_rate,
        })
        ece += (count / n) * abs(pred_mean - obs_rate)

    return bins, float(ece)


def stratified_bootstrap_indices(y_true: np.ndarray, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Each resample draws, with replacement, from the positive and negative
    index pools separately, at their original sizes — preserving the class
    balance of the observed data in every resample."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    resamples = []
    for _ in range(n_resamples):
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        resamples.append(np.concatenate([pos_sample, neg_sample]))
    return resamples


def percentile_interval(values: list[float], lo: float = 2.5, hi: float = 97.5):
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))
