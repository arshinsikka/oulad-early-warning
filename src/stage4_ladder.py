"""
Stage 4: model ladder, validation-only selection, frozen threshold.

THE TEST SPLIT MUST NOT BE READ. Every data load goes through
stage4_guard.load_split(), which raises if 'test' is requested. Stage 4 never
passes it.

Usage:
    .venv/bin/python src/stage4_ladder.py
"""

import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from stage2_cohort import CUTOFFS
from stage3_report import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from stage4_guard import load_split
from stage4_models import (
    B0BaseRate, B3_C_GRID, HEADLINE_RATIO, LGBM_GRID,
    confusion_at_threshold, cost_ratio_curve, expected_cost, fit_lightgbm,
    fit_logistic, ranking_metrics, sweep_threshold,
)
from stage4_preprocess import FeaturePreprocessor, NumericPreprocessor

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
MODELS_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "stage4_ladder.txt"
JSON_PATH = ROOT / "reports" / "frozen_threshold.json"
RESULTS_CACHE_PATH = ROOT / "reports" / "stage4_results.joblib"

BASE_SEED = 42
STABILITY_SEEDS = [42, 7, 123, 2024]

GROUP_A_NUMERIC = ["num_of_prev_attempts", "studied_credits"]
GROUP_A_CATEGORICAL = [
    "gender", "region", "highest_education", "imd_band", "age_band", "disability",
]
RUNG_NAMES = ["B0", "B1", "B2", "B3", "M1"]


def git_commit_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def build_preprocessors(train_df: pd.DataFrame) -> dict:
    preproc_b1 = NumericPreprocessor(["days_since_last_activity"], add_indicators=False).fit(train_df)
    preproc_b2 = FeaturePreprocessor(GROUP_A_NUMERIC, GROUP_A_CATEGORICAL, add_indicators=True).fit(train_df)
    preproc_full = FeaturePreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_indicators=True).fit(train_df)
    return {"B1": preproc_b1, "B2": preproc_b2, "full": preproc_full}


def fit_ladder_at_seed(train_df: pd.DataFrame, validate_df: pd.DataFrame,
                        preprocs: dict, seed: int,
                        b3_C: float | None, m1_params: dict | None):
    """Fits B0..M1 on train, evaluated on validate. If b3_C/m1_params are
    None, performs the selection sweep/grid; otherwise fits directly with
    the given hyperparameters (used for the seed-stability control runs)."""
    y_train = train_df["not_completed"].values
    y_val = validate_df["not_completed"].values

    fitted = {}

    # B0
    b0 = B0BaseRate().fit(None, y_train)
    fitted["B0"] = {"model": b0, "val_prob": b0.predict_proba1(validate_df)}

    # B1
    X_train_b1 = preprocs["B1"].transform(train_df)
    X_val_b1 = preprocs["B1"].transform(validate_df)
    b1 = fit_logistic(X_train_b1, y_train, C=1.0, seed=seed)
    fitted["B1"] = {"model": b1, "val_prob": b1.predict_proba(X_val_b1.values)[:, 1]}

    # B2
    X_train_b2 = preprocs["B2"].transform(train_df)
    X_val_b2 = preprocs["B2"].transform(validate_df)
    b2 = fit_logistic(X_train_b2, y_train, C=1.0, seed=seed)
    fitted["B2"] = {"model": b2, "val_prob": b2.predict_proba(X_val_b2.values)[:, 1]}

    # B3
    X_train_full = preprocs["full"].transform(train_df)
    X_val_full = preprocs["full"].transform(validate_df)
    if b3_C is None:
        b3_candidates = []
        for C in B3_C_GRID:
            m = fit_logistic(X_train_full, y_train, C=C, seed=seed)
            prob = m.predict_proba(X_val_full.values)[:, 1]
            _, _, best_t, best_c = sweep_threshold(y_val, prob, ratio=HEADLINE_RATIO)
            b3_candidates.append((C, m, prob, best_c))
        b3_C, b3_model, b3_prob, _ = min(b3_candidates, key=lambda t: t[3])
    else:
        b3_model = fit_logistic(X_train_full, y_train, C=b3_C, seed=seed)
        b3_prob = b3_model.predict_proba(X_val_full.values)[:, 1]
    fitted["B3"] = {"model": b3_model, "val_prob": b3_prob, "C": b3_C}

    # M1
    if m1_params is None:
        m1_candidates = []
        for params in LGBM_GRID:
            m = fit_lightgbm(X_train_full, y_train, X_val_full, y_val, params, seed=seed)
            prob = m.predict_proba(X_val_full.values)[:, 1]
            _, _, best_t, best_c = sweep_threshold(y_val, prob, ratio=HEADLINE_RATIO)
            m1_candidates.append((params, m, prob, best_c))
        m1_params, m1_model, m1_prob, _ = min(m1_candidates, key=lambda t: t[3])
    else:
        m1_model = fit_lightgbm(X_train_full, y_train, X_val_full, y_val, m1_params, seed=seed)
        m1_prob = m1_model.predict_proba(X_val_full.values)[:, 1]
    fitted["M1"] = {"model": m1_model, "val_prob": m1_prob, "params": m1_params}

    return fitted, y_val


def summarise_model(name: str, fit_result: dict, y_val: np.ndarray) -> dict:
    prob = fit_result["val_prob"]
    thresholds, costs, best_t, best_c = sweep_threshold(y_val, prob, ratio=HEADLINE_RATIO)
    metrics = ranking_metrics(y_val, prob)
    ratio_curve = cost_ratio_curve(y_val, prob)

    t_minus = round(max(0.01, best_t - 0.05), 2)
    t_plus = round(min(0.99, best_t + 0.05), 2)
    cost_minus = expected_cost(y_val, prob, t_minus, HEADLINE_RATIO)
    cost_plus = expected_cost(y_val, prob, t_plus, HEADLINE_RATIO)

    # Confusion counts at the frozen threshold, cached so the report can
    # recompute cost at any ratio (e.g. the trivial-policy comparison)
    # without re-touching the fitted model or its probabilities.
    fn, fp, n = confusion_at_threshold(y_val, prob, best_t)

    summary = {
        "threshold": best_t,
        "expected_cost": best_c,
        "cost_minus_005": cost_minus,
        "cost_plus_005": cost_plus,
        "metrics": metrics,
        "ratio_curve": ratio_curve,
        "fn_at_threshold": fn,
        "fp_at_threshold": fp,
        "n_validate": n,
    }
    if name == "B3":
        summary["C"] = fit_result["C"]
    if name == "M1":
        summary["params"] = fit_result["params"]
        summary["n_estimators"] = fit_result["model"].best_iteration_
    return summary


def refit_on_trainval(trainval_df: pd.DataFrame, b3_C: float, m1_params: dict, m1_n_estimators: int, seed: int):
    y_trainval = trainval_df["not_completed"].values

    preproc_b1 = NumericPreprocessor(["days_since_last_activity"], add_indicators=False).fit(trainval_df)
    preproc_b2 = FeaturePreprocessor(GROUP_A_NUMERIC, GROUP_A_CATEGORICAL, add_indicators=True).fit(trainval_df)
    preproc_full = FeaturePreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, add_indicators=True).fit(trainval_df)

    b0 = B0BaseRate().fit(None, y_trainval)

    X_b1 = preproc_b1.transform(trainval_df)
    b1 = fit_logistic(X_b1, y_trainval, C=1.0, seed=seed)

    X_b2 = preproc_b2.transform(trainval_df)
    b2 = fit_logistic(X_b2, y_trainval, C=1.0, seed=seed)

    X_full = preproc_full.transform(trainval_df)
    b3 = fit_logistic(X_full, y_trainval, C=b3_C, seed=seed)

    n_neg = int((y_trainval == 0).sum())
    n_pos = int((y_trainval == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    import lightgbm as lgb
    m1 = lgb.LGBMClassifier(
        num_leaves=m1_params["num_leaves"],
        learning_rate=m1_params["learning_rate"],
        min_child_samples=m1_params["min_child_samples"],
        n_estimators=m1_n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        deterministic=True,
        n_jobs=1,
        verbosity=-1,
    )
    m1.fit(X_full.values, y_trainval)

    return {
        "preprocessors": {"B1": preproc_b1, "B2": preproc_b2, "full": preproc_full},
        "models": {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "M1": m1},
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)

    all_results = {}
    frozen = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": git_commit_hash(), "cost_ratio": HEADLINE_RATIO, "cutoffs": {}}

    for D in CUTOFFS:
        table = f"features_d{D}"
        train_df = load_split(con, table, ["train"])
        validate_df = load_split(con, table, ["validate"])

        preprocs = build_preprocessors(train_df)

        fitted, y_val = fit_ladder_at_seed(
            train_df, validate_df, preprocs, seed=BASE_SEED, b3_C=None, m1_params=None
        )

        model_summaries = {}
        for rung in RUNG_NAMES:
            model_summaries[rung] = summarise_model(rung, fitted[rung], y_val)

        # B3 coefficients for interpretability.
        b3_model = fitted["B3"]["model"]
        b3_cols = preprocs["full"].output_columns
        coefs = list(zip(b3_cols, b3_model.coef_[0].tolist()))
        coefs.sort(key=lambda t: abs(t[1]), reverse=True)
        model_summaries["B3"]["intercept"] = float(b3_model.intercept_[0])
        model_summaries["B3"]["top_coefficients"] = coefs[:15]

        # Seed stability: refit B3 (fixed C) and M1 (fixed params) at each seed,
        # cost measured at the seed-42 frozen threshold. B0/B1/B2 have no
        # meaningful seed dependency under the lbfgs solver.
        seed_costs = {}
        for seed in STABILITY_SEEDS:
            if seed == BASE_SEED:
                seed_fitted = fitted
                seed_y_val = y_val
            else:
                seed_fitted, seed_y_val = fit_ladder_at_seed(
                    train_df, validate_df, preprocs, seed=seed,
                    b3_C=model_summaries["B3"]["C"], m1_params=model_summaries["M1"]["params"],
                )
            costs_this_seed = {}
            for rung in RUNG_NAMES:
                frozen_t = model_summaries[rung]["threshold"]
                prob = seed_fitted[rung]["val_prob"]
                costs_this_seed[rung] = expected_cost(seed_y_val, prob, frozen_t, HEADLINE_RATIO)
            seed_costs[seed] = costs_this_seed

        seed_rankings = {
            seed: sorted(RUNG_NAMES, key=lambda r: costs[r])
            for seed, costs in seed_costs.items()
        }

        # Refit on train + validate combined, per the Section 4 refit rule.
        trainval_df = pd.concat([train_df, validate_df], ignore_index=True)
        refit_bundle = refit_on_trainval(
            trainval_df,
            b3_C=model_summaries["B3"]["C"],
            m1_params=model_summaries["M1"]["params"],
            m1_n_estimators=model_summaries["M1"]["n_estimators"],
            seed=BASE_SEED,
        )
        joblib.dump(refit_bundle, MODELS_DIR / f"stage4_refit_d{D}.joblib")

        train_n = len(train_df)
        val_n = len(validate_df)
        train_base_rate = float(train_df["not_completed"].mean())
        val_base_rate = float(validate_df["not_completed"].mean())

        all_results[D] = {
            "train_n": train_n, "validate_n": val_n,
            "train_base_rate": train_base_rate, "validate_base_rate": val_base_rate,
            "models": model_summaries,
            "seed_costs": seed_costs,
            "seed_rankings": seed_rankings,
        }

        frozen["cutoffs"][str(D)] = {
            rung: {
                "threshold": model_summaries[rung]["threshold"],
                "validate_expected_cost": model_summaries[rung]["expected_cost"],
                "hyperparameters": (
                    {} if rung in ("B0",) else
                    {"C": 1.0} if rung in ("B1", "B2") else
                    {"C": model_summaries["B3"]["C"]} if rung == "B3" else
                    {**model_summaries["M1"]["params"],
                     "n_estimators": model_summaries["M1"]["n_estimators"]}
                ),
            }
            for rung in RUNG_NAMES
        }

        print(f"D={D}: ladder fit, selected, refit, and persisted.")

    con.close()

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(frozen, indent=2) + "\n")
    joblib.dump(all_results, RESULTS_CACHE_PATH)
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {RESULTS_CACHE_PATH}")


if __name__ == "__main__":
    main()
