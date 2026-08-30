"""
Stage 6 computation: drift measurement (Section 10, D1-D4) and global
explainability (Section 11, E1).

NOTHING HERE FITS ANYTHING. No model is refit, no threshold recomputed, no
calibration adjusted, no drift corrected. The frozen Stage 4 preprocessors and
models are loaded and applied; .fit() is never called on any of them. Test
predictions are read from reports/stage5_test_predictions.parquet rather than
recomputed, and the model is touched only to obtain SHAP contributions, which
are checked against those stored predictions before use.
"""

import numpy as np
import pandas as pd

# PSI is a sum of terms p_i * ln(p_i / q_i). A bin that is empty on one side
# sends that term to infinity, which reports the arrival of a single unseen
# row as an unbounded drift. The industry convention is to substitute a small
# constant for the empty proportion, bounding the term instead. 1e-4 is the
# usual choice: one hundredth of one percent of the population, below any
# proportion these splits can actually resolve (the smallest split here has
# ~6.5k rows, so one row is ~1.5e-4).
PSI_EPSILON = 1e-4
PSI_N_BINS = 10
NULL_TOKEN = "__NULL__"


def psi_terms(train_props: np.ndarray, test_props: np.ndarray,
              epsilon: float = PSI_EPSILON) -> np.ndarray:
    """Per-bin PSI contributions, with the empty-bin substitution applied."""
    p = np.where(test_props <= 0, epsilon, test_props)
    q = np.where(train_props <= 0, epsilon, train_props)
    return (p - q) * np.log(p / q)


def asymmetric_empty(train_props: np.ndarray, test_props: np.ndarray) -> int:
    """Bins empty on exactly one side — the only ones where the substitution
    changes the answer. A bin empty on both sides (a null bin for a feature
    that has no nulls anywhere, say) contributes exactly zero whatever
    constant is chosen, and counting it would overstate how often the
    substitution actually fired."""
    return int(((train_props <= 0) ^ (test_props <= 0)).sum())


def psi_from_bins(bins: list[dict], epsilon: float) -> float:
    """Recompute a feature's PSI from its stored bin proportions under a
    different empty-bin constant, for the sensitivity check."""
    train_props = np.array([b["train_prop"] for b in bins])
    test_props = np.array([b["test_prop"] for b in bins])
    return float(psi_terms(train_props, test_props, epsilon=epsilon).sum())


def numeric_psi(train_col: pd.Series, test_col: pd.Series) -> dict:
    """Quantile bins defined on the TRAIN distribution, applied unchanged to
    test. Nulls are carried as their own bin rather than imputed, because
    missingness is itself a thing that can drift."""
    train_vals = train_col.astype(float)
    test_vals = test_col.astype(float)
    train_nn = train_vals.dropna().values
    test_nn = test_vals.dropna().values

    quantiles = np.linspace(0.0, 1.0, PSI_N_BINS + 1)
    raw_edges = np.quantile(train_nn, quantiles) if len(train_nn) else np.array([0.0])
    # Heavily tied features (counts, mostly-zero features) collapse adjacent
    # quantiles onto the same value; the duplicate edges are dropped and the
    # realised bin count is reported rather than claimed to be 10.
    edges = np.unique(raw_edges)
    degenerate = len(edges) < 2
    if degenerate:
        interior = np.array([])
    else:
        # Open the outer edges so test values beyond the train range land in
        # the end bins instead of falling outside the binning entirely.
        interior = edges[1:-1]

    train_bins = np.digitize(train_nn, interior, right=False)
    test_bins = np.digitize(test_nn, interior, right=False)
    n_bins = len(interior) + 1

    n_train, n_test = len(train_vals), len(test_vals)
    train_counts = np.bincount(train_bins, minlength=n_bins).astype(float)
    test_counts = np.bincount(test_bins, minlength=n_bins).astype(float)
    train_counts = np.append(train_counts, float(train_vals.isna().sum()))
    test_counts = np.append(test_counts, float(test_vals.isna().sum()))

    train_props = train_counts / n_train if n_train else train_counts
    test_props = test_counts / n_test if n_test else test_counts
    terms = psi_terms(train_props, test_props)

    labels = []
    for b in range(n_bins):
        lo = "-inf" if b == 0 else f"{interior[b - 1]:.4g}"
        hi = "+inf" if b == n_bins - 1 else f"{interior[b]:.4g}"
        labels.append(f"[{lo}, {hi})")
    labels.append(NULL_TOKEN)

    return {
        "psi": float(terms.sum()),
        "n_bins_used": n_bins,
        "degenerate": degenerate,
        "bins": [
            {"label": labels[i], "train_prop": float(train_props[i]),
             "test_prop": float(test_props[i]), "term": float(terms[i])}
            for i in range(len(labels))
        ],
        "empty_bins": asymmetric_empty(train_props, test_props),
    }


def categorical_psi(train_col: pd.Series, test_col: pd.Series) -> dict:
    """Categories as their own bins, null carried as its own category."""
    train_vals = train_col.fillna(NULL_TOKEN).astype(str)
    test_vals = test_col.fillna(NULL_TOKEN).astype(str)
    categories = sorted(set(train_vals.unique()) | set(test_vals.unique()))

    train_props = np.array([float((train_vals == c).mean()) for c in categories])
    test_props = np.array([float((test_vals == c).mean()) for c in categories])
    terms = psi_terms(train_props, test_props)

    return {
        "psi": float(terms.sum()),
        "n_bins_used": len(categories),
        "degenerate": len(categories) < 2,
        "bins": [
            {"label": categories[i], "train_prop": float(train_props[i]),
             "test_prop": float(test_props[i]), "term": float(terms[i])}
            for i in range(len(categories))
        ],
        "empty_bins": asymmetric_empty(train_props, test_props),
    }


def psi_band(psi: float) -> str:
    """The conventional reading. A convention from industry practice, NOT a
    threshold this protocol pre-registered; Section 10 says as much."""
    if psi < 0.1:
        return "stable"
    if psi <= 0.25:
        return "moderate"
    return "significant"


def group_contributions(contrib: np.ndarray, design_columns: list[str],
                        groups: dict[str, list[int]]) -> dict[str, float]:
    """Mean absolute contribution per protocol feature.

    A protocol feature can span several design columns: a categorical becomes
    one column per level, and a numeric with nulls gains a was_missing_
    indicator. The row's contribution for that feature is the SIGNED sum over
    its columns, and the importance is the mean of the absolute value of that
    sum. Summing absolute values per column instead would inflate wide
    one-hot groups by counting cancelling contributions twice.
    """
    out = {}
    for feature, idx in groups.items():
        out[feature] = float(np.abs(contrib[:, idx].sum(axis=1)).mean())
    return out


def design_groups(design_columns: list[str], numeric_features: list[str],
                  categorical_features: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {f: [] for f in numeric_features + categorical_features}
    for i, col in enumerate(design_columns):
        if col in groups:
            groups[col].append(i)
            continue
        if col.startswith("was_missing_"):
            groups[col[len("was_missing_"):]].append(i)
            continue
        base = col.split("__")[0]
        if base in groups:
            groups[base].append(i)
            continue
        raise KeyError(f"design column {col!r} maps to no protocol feature")
    missing = [f for f, idx in groups.items() if not idx]
    if missing:
        raise KeyError(f"protocol features with no design column: {missing}")
    return groups


def rank_map(values: dict[str, float]) -> dict[str, int]:
    """Rank by descending value, 1 = largest."""
    order = sorted(values, key=lambda k: -values[k])
    return {k: i + 1 for i, k in enumerate(order)}
