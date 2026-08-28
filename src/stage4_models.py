"""
Stage 4 model utilities: expected cost, threshold sweeps, ranking metrics,
and the B0/B1/B2/B3/M1 ladder rungs.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import lightgbm as lgb
from lightgbm.sklearn import LGBMDeprecationWarning

warnings.filterwarnings("ignore", category=LGBMDeprecationWarning)

THRESHOLD_GRID = np.round(np.arange(0.01, 0.99 + 1e-9, 0.01), 2)
RATIO_GRID = list(range(2, 21))
HEADLINE_RATIO = 10

LGBM_GRID = [
    {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
    for nl in (15, 31, 63)
    for lr in (0.01, 0.05, 0.1)
    for mcs in (20, 50, 100)
]
assert len(LGBM_GRID) == 27

B3_C_GRID = [0.001, 0.01, 0.1, 1, 10]


def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, ratio: float) -> float:
    preds = (y_prob >= threshold).astype(int)
    y_true = np.asarray(y_true)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    n = len(y_true)
    return (ratio * fn + fp) / n


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray, ratio: float = HEADLINE_RATIO):
    """Returns (thresholds, costs, best_threshold, best_cost)."""
    costs = np.array([expected_cost(y_true, y_prob, t, ratio) for t in THRESHOLD_GRID])
    best_idx = int(np.argmin(costs))
    return THRESHOLD_GRID, costs, float(THRESHOLD_GRID[best_idx]), float(costs[best_idx])


def cost_ratio_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """For each ratio in RATIO_GRID, the best achievable threshold and cost."""
    curve = []
    for ratio in RATIO_GRID:
        _, _, best_t, best_c = sweep_threshold(y_true, y_prob, ratio=ratio)
        curve.append({"ratio": ratio, "threshold": best_t, "expected_cost": best_c})
    return curve


def recall_at_budget(y_true: np.ndarray, y_prob: np.ndarray, budget_frac: float) -> float:
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = max(1, int(round(budget_frac * n)))
    order = np.argsort(-y_prob)
    flagged = np.zeros(n, dtype=int)
    flagged[order[:k]] = 1
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return float("nan")
    tp = int(((y_true == 1) & (flagged == 1)).sum())
    return tp / total_pos


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        auc_pr = float("nan")
        auc_roc = float("nan")
    else:
        auc_pr = float(average_precision_score(y_true, y_prob))
        auc_roc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "brier": brier,
        "recall_at_5pct": recall_at_budget(y_true, y_prob, 0.05),
        "recall_at_10pct": recall_at_budget(y_true, y_prob, 0.10),
        "recall_at_20pct": recall_at_budget(y_true, y_prob, 0.20),
    }


class B0BaseRate:
    """Predicts the fit-data prevalence for every row."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "B0BaseRate":
        self.p_ = float(np.mean(y))
        return self

    def predict_proba1(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p_)


def fit_logistic(X: pd.DataFrame, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=seed,
    )
    model.fit(X.values, y)
    return model


def fit_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray,
                  X_val: pd.DataFrame, y_val: np.ndarray,
                  params: dict, seed: int) -> lgb.LGBMClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"],
        min_child_samples=params["min_child_samples"],
        n_estimators=1000,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        deterministic=True,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        X_train.values, y_train,
        eval_set=[(X_val.values, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model
