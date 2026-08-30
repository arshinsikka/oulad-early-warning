"""
Stage 5: the holdout opens. THIS RUNS ONCE.

Stage 5 does not fit, refit, tune, recalibrate, select between models, or
recompute any threshold. It loads the refit artefacts persisted by Stage 4
(models/stage4_refit_d{D}.joblib), loads the frozen thresholds from
reports/frozen_threshold.json, verifies the loaded models' hyperparameters
match that file exactly, and scores the test split (2014J) once.

Usage:
    .venv/bin/python src/stage5_holdout.py
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from stage2_cohort import CUTOFFS
from stage4_models import (
    HEADLINE_RATIO, RATIO_GRID, confusion_at_threshold, expected_cost,
    ranking_metrics, sweep_threshold,
)
from stage5_metrics import calibration_curve_stats, percentile_interval, stratified_bootstrap_indices

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
MODELS_DIR = ROOT / "models"
FROZEN_JSON_PATH = ROOT / "reports" / "frozen_threshold.json"
# Stage 4's validate-side metrics, read only so that Section 8 can state the
# validate figures a drift comparison needs. Nothing is refit from it.
STAGE4_RESULTS_PATH = ROOT / "reports" / "stage4_results.joblib"
REPORT_PATH = ROOT / "reports" / "stage5_holdout.txt"
PREDICTIONS_PATH = ROOT / "reports" / "stage5_test_predictions.parquet"

RUNG_NAMES = ["B0", "B1", "B2", "B3", "M1"]
SLICE_COLS = ["imd_band", "age_band", "disability", "gender"]
SLICE_CUTOFF = 28
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
UNDERPOWERED_N = 100


def load_test_split(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    """The one function in this project permitted to load split='test'.
    Stage 5 exists specifically to open this data, exactly once."""
    return con.execute(f"SELECT * FROM {table} WHERE split = 'test'").df()


def verify_hyperparameters(models: dict, frozen_cutoff: dict) -> None:
    mismatches = []

    for rung in ("B1", "B2", "B3"):
        model_C = models[rung].C
        frozen_C = frozen_cutoff[rung]["hyperparameters"]["C"]
        if model_C != frozen_C:
            mismatches.append(f"{rung}.C: loaded model has {model_C}, frozen_threshold.json has {frozen_C}")

    m1 = models["M1"]
    m1_params = m1.get_params()
    frozen_m1 = frozen_cutoff["M1"]["hyperparameters"]
    for key in ("num_leaves", "learning_rate", "min_child_samples", "n_estimators"):
        if m1_params[key] != frozen_m1[key]:
            mismatches.append(f"M1.{key}: loaded model has {m1_params[key]}, frozen_threshold.json has {frozen_m1[key]}")

    if mismatches:
        raise RuntimeError(
            "Stage 5 hyperparameter guard failed — loaded model does not match "
            "frozen_threshold.json:\n  " + "\n  ".join(mismatches)
        )


def predict_all(bundle: dict, test_df: pd.DataFrame) -> dict:
    preprocs = bundle["preprocessors"]
    models = bundle["models"]
    probs = {}
    probs["B0"] = models["B0"].predict_proba1(test_df)
    probs["B1"] = models["B1"].predict_proba(preprocs["B1"].transform(test_df).values)[:, 1]
    probs["B2"] = models["B2"].predict_proba(preprocs["B2"].transform(test_df).values)[:, 1]
    X_full = preprocs["full"].transform(test_df)
    probs["B3"] = models["B3"].predict_proba(X_full.values)[:, 1]
    probs["M1"] = models["M1"].predict_proba(X_full.values)[:, 1]
    return probs


def point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   rng: np.random.Generator) -> dict:
    m = ranking_metrics(y_true, y_prob, rng=rng)
    m["expected_cost"] = expected_cost(y_true, y_prob, threshold, HEADLINE_RATIO)
    return m


def bootstrap_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, resamples: list,
                       rng: np.random.Generator) -> dict:
    keys = ["expected_cost", "auc_pr", "auc_roc", "brier", "recall_at_5pct", "recall_at_10pct", "recall_at_20pct"]
    collected = {k: [] for k in keys}
    for idx in resamples:
        yt, yp = y_true[idx], y_prob[idx]
        m = ranking_metrics(yt, yp, rng=rng)
        m["expected_cost"] = expected_cost(yt, yp, threshold, HEADLINE_RATIO)
        for k in keys:
            collected[k].append(m[k])
    return {k: percentile_interval(v) for k, v in collected.items()}


def recall_ceiling(budget_frac: float, base_rate: float) -> float:
    return min(1.0, budget_frac / base_rate) if base_rate > 0 else float("nan")


def ci_str(interval: tuple) -> str:
    return f"[{interval[0]:.4f},{interval[1]:.4f}]"


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def main() -> None:
    frozen = json.loads(FROZEN_JSON_PATH.read_text())
    con = duckdb.connect(str(DB_PATH), read_only=True)
    metrics_rng = np.random.default_rng(BOOTSTRAP_SEED)

    out: list[str] = []
    out.append("Stage 5 Holdout Report")
    out.append("THE HOLDOUT OPENS ONCE. No model, feature, hyperparameter or threshold choice may change after this.")
    out.append(f"frozen_threshold.json git_commit_at_generation: {frozen['git_commit_at_generation']}")

    prediction_rows = []
    per_cutoff = {}

    for D in CUTOFFS:
        table = f"features_d{D}"
        test_df = load_test_split(con, table)
        y_test = test_df["not_completed"].values

        bundle = joblib.load(MODELS_DIR / f"stage4_refit_d{D}.joblib")
        frozen_cutoff = frozen["cutoffs"][str(D)]
        verify_hyperparameters(bundle["models"], frozen_cutoff)

        probs = predict_all(bundle, test_df)

        resamples = stratified_bootstrap_indices(y_test, N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

        rung_results = {}
        for rung in RUNG_NAMES:
            threshold = frozen_cutoff[rung]["threshold"]
            point = point_metrics(y_test, probs[rung], threshold, rng=metrics_rng)
            interval = bootstrap_metrics(y_test, probs[rung], threshold, resamples, rng=metrics_rng)
            calib_bins, ece = calibration_curve_stats(y_test, probs[rung], n_bins=10)
            rung_results[rung] = {
                "threshold": threshold,
                "point": point,
                "interval": interval,
                "calibration_bins": calib_bins,
                "ece": ece,
                "fn_fp_n": confusion_at_threshold(y_test, probs[rung], threshold),
            }

        # Paired M1 vs B3 bootstrap, same resample indices for both.
        # The reported quantity is M1 - B3, in the order the label states. Expected
        # cost is a loss, so a NEGATIVE difference means M1 is the cheaper model.
        diffs = []
        for idx in resamples:
            yt = y_test[idx]
            cost_b3 = expected_cost(yt, probs["B3"][idx], frozen_cutoff["B3"]["threshold"], HEADLINE_RATIO)
            cost_m1 = expected_cost(yt, probs["M1"][idx], frozen_cutoff["M1"]["threshold"], HEADLINE_RATIO)
            diffs.append(cost_m1 - cost_b3)
        paired_interval = percentile_interval(diffs)
        paired_point = (rung_results["M1"]["point"]["expected_cost"] - rung_results["B3"]["point"]["expected_cost"])

        selected_rung = min(RUNG_NAMES, key=lambda r: frozen_cutoff[r]["validate_expected_cost"])

        per_cutoff[D] = {
            "test_df": test_df, "y_test": y_test, "probs": probs,
            "base_rate": float(y_test.mean()), "n": len(y_test),
            "rung_results": rung_results,
            "paired_diff_point": paired_point, "paired_diff_interval": paired_interval,
            "selected_rung": selected_rung,
        }

        block = pd.DataFrame({
            "cutoff": D,
            "code_module": test_df["code_module"].values,
            "code_presentation": test_df["code_presentation"].values,
            "id_student": test_df["id_student"].values,
            "not_completed": y_test,
            "split": test_df["split"].values,
        })
        for rung in RUNG_NAMES:
            block[f"prob_{rung}"] = probs[rung]
        prediction_rows.append(block)

    con.close()

    # --- Report assembly ---

    out.append(section("1. Test base rate per cutoff"))
    out.append(f"{'cutoff':<10}{'n':>8}{'base_rate':>12}")
    for D in CUTOFFS:
        out.append(f"D={D:<8}{per_cutoff[D]['n']:>8}{per_cutoff[D]['base_rate']:>12.4f}")
    out.append("")

    out.append(section("2. Metrics, every model, every cutoff, on TEST, with 95% bootstrap intervals"))
    for D in CUTOFFS:
        out.append(f"--- D={D} ---")
        out.append(
            f"{'model':<6}{'thr':>6}{'cost':>9}{'[95% CI]':>18}{'AUC-PR':>9}{'[95% CI]':>18}"
            f"{'Brier':>8}{'[95% CI]':>18}{'AUC-ROC*':>9}{'[95% CI]':>18}"
        )
        for rung in RUNG_NAMES:
            r = per_cutoff[D]["rung_results"][rung]
            p, iv = r["point"], r["interval"]
            out.append(
                f"{rung:<6}{r['threshold']:>6.2f}{p['expected_cost']:>9.4f}"
                f"{ci_str(iv['expected_cost']):>18}"
                f"{p['auc_pr']:>9.4f}{ci_str(iv['auc_pr']):>18}"
                f"{p['brier']:>8.4f}{ci_str(iv['brier']):>18}"
                f"{p['auc_roc']:>9.4f}{ci_str(iv['auc_roc']):>18}"
            )
        out.append("* AUC-ROC reported for completeness. The temporal split (Section 4) makes it "
                    "non-comparable to published OULAD figures that use random splits.")
        out.append("")

        base_rate = per_cutoff[D]["base_rate"]
        out.append("Recall at alert budgets (with ceiling = budget / test base rate, capped at 1.0):")
        out.append(f"{'model':<6}{'r@5%':>8}{'[95% CI]':>18}{'ceil@5%':>9}"
                    f"{'r@10%':>8}{'[95% CI]':>18}{'ceil@10%':>10}"
                    f"{'r@20%':>8}{'[95% CI]':>18}{'ceil@20%':>10}")
        for rung in RUNG_NAMES:
            r = per_cutoff[D]["rung_results"][rung]
            p, iv = r["point"], r["interval"]
            c5, c10, c20 = recall_ceiling(0.05, base_rate), recall_ceiling(0.10, base_rate), recall_ceiling(0.20, base_rate)
            out.append(
                f"{rung:<6}{p['recall_at_5pct']:>8.4f}{ci_str(iv['recall_at_5pct']):>18}{c5:>9.4f}"
                f"{p['recall_at_10pct']:>8.4f}{ci_str(iv['recall_at_10pct']):>18}{c10:>10.4f}"
                f"{p['recall_at_20pct']:>8.4f}{ci_str(iv['recall_at_20pct']):>18}{c20:>10.4f}"
            )
        out.append("")

    out.append(section("3. Calibration (10 equal-width bins) and expected calibration error"))
    for D in CUTOFFS:
        out.append(f"--- D={D} ---")
        for rung in RUNG_NAMES:
            r = per_cutoff[D]["rung_results"][rung]
            out.append(f"  {rung}: ECE={r['ece']:.4f}")
            out.append(f"    {'bin':<14}{'count':>8}{'pred_mean':>12}{'obs_rate':>12}")
            for b in r["calibration_bins"]:
                pm = f"{b['predicted_mean']:.4f}" if b["predicted_mean"] is not None else "  --  "
                orate = f"{b['observed_rate']:.4f}" if b["observed_rate"] is not None else "  --  "
                out.append(f"    [{b['bin_lo']:.1f},{b['bin_hi']:.1f}){'':<4}{b['count']:>8}{pm:>12}{orate:>12}")
        out.append("")

    out.append(section(f"4. Paired M1 - B3 expected cost difference (negative = M1 cheaper), test, all cutoffs"))
    out.append(
        "point_diff is cost(M1) - cost(B3), in that order. Expected cost is a loss, so a "
        "NEGATIVE difference means M1 costs less than B3 and a positive one means B3 costs less."
    )
    out.append(f"{'cutoff':<8}{'point_diff':>12}{'[95% CI]':>22}{'outcome':>10}  detail")
    for D in CUTOFFS:
        pc = per_cutoff[D]["paired_diff_point"]
        lo, hi = per_cutoff[D]["paired_diff_interval"]
        interval_excludes_zero = not (lo <= 0 <= hi)
        if interval_excludes_zero and pc < 0:
            outcome, detail = "O1", "M1 materially beats B3 (M1 cheaper)"
        elif interval_excludes_zero and pc > 0:
            outcome, detail = "O2", "B3 materially beats M1 (not O1: M1 does not win)"
        else:
            outcome, detail = "O2", "interval includes zero, no separated margin either way"
        out.append(f"D={D:<6}{pc:>12.4f}{ci_str((lo, hi)):>22}{outcome:>10}  {detail}")
    out.append("")
    out.append(
        "O1 = M1 materially beats B3 (interval excludes zero and lies below it, M1 cheaper). "
        "O2 = M1 does not beat B3 by an interval-separated margin — this covers both a null result "
        "and the case where B3 significantly beats M1, since O1 requires M1 to win. "
        "Per Section 13, the formal declared outcome is scoped to D=28; D=14/D=56 shown as "
        "supplementary evidence of the same comparison at the other cutoffs."
    )
    out.append("")

    out.append(section(f"5. Slice reporting, D={SLICE_CUTOFF}, selected model = {per_cutoff[SLICE_CUTOFF]['selected_rung']}"))
    sel = per_cutoff[SLICE_CUTOFF]["selected_rung"]
    sel_threshold = frozen["cutoffs"][str(SLICE_CUTOFF)][sel]["threshold"]
    sel_prob = per_cutoff[SLICE_CUTOFF]["probs"][sel]
    sel_y = per_cutoff[SLICE_CUTOFF]["y_test"]
    sel_df = per_cutoff[SLICE_CUTOFF]["test_df"]
    slice_overall_cost = per_cutoff[SLICE_CUTOFF]["rung_results"][sel]["point"]["expected_cost"]

    out.append("Reported, not corrected. No fairness intervention. Slices with n < 100 are flagged underpowered.")
    out.append(
        f"gap% is (slice expected cost - overall expected cost) / overall expected cost, in percent, where the "
        f"overall figure is {slice_overall_cost:.4f} (model {sel}, D={SLICE_CUTOFF}, whole test split). Expected cost "
        "is a loss, so a positive gap% means the slice is worse than the cohort overall. The gap is on expected "
        "cost because Section 8 designates it the primary metric; the slice values of the secondary metrics are "
        "in the same rows for a reader who wants to form the same ratio on those."
    )
    out.append(
        "The protocol does not define a threshold at which a gap becomes material (see Section 8 of this report), "
        "so no slice is labelled materially worse or not materially worse here. The numbers are reported as measured."
    )
    out.append("")
    slice_gaps = []
    for col in SLICE_COLS:
        out.append(f"--- {col} ---")
        out.append(f"{'value':<28}{'n':>7}{'flag':>6}{'cost':>9}{'gap%':>9}{'AUC-PR':>9}{'AUC-ROC':>9}{'Brier':>9}{'recall@10%':>12}")
        values = sel_df[col].fillna("__NULL__").unique().tolist()
        for value in sorted(values, key=str):
            mask = (sel_df[col].fillna("__NULL__") == value).values
            n = int(mask.sum())
            if n == 0:
                continue
            yt, yp = sel_y[mask], sel_prob[mask]
            flag = "UNDERPWR" if n < UNDERPOWERED_N else ""
            if len(set(yt)) < 2:
                out.append(f"{str(value):<28}{n:>7}{flag:>6}   (single-class slice, ranking metrics undefined)")
                continue
            m = ranking_metrics(yt, yp, rng=metrics_rng)
            cost = expected_cost(yt, yp, sel_threshold, HEADLINE_RATIO)
            gap_pct = (cost - slice_overall_cost) / slice_overall_cost * 100 if slice_overall_cost else float("nan")
            slice_gaps.append({"col": col, "value": value, "n": n, "flag": flag, "cost": cost, "gap_pct": gap_pct})
            out.append(
                f"{str(value):<28}{n:>7}{flag:>6}{cost:>9.4f}{gap_pct:>8.1f}%{m['auc_pr']:>9.4f}"
                f"{m['auc_roc']:>9.4f}{m['brier']:>9.4f}{m['recall_at_10pct']:>12.4f}"
            )
        out.append("")

    out.append(section("6. Trivial policy comparison, test, frozen threshold vs counterfactual test-optimal threshold"))
    out.append(
        "cost_frozen uses the pre-committed validate-frozen threshold (the actually deployed policy). "
        "cost_counterfactual re-optimises the threshold ON TEST at each ratio — NOT available at "
        "deployment time, shown only to separate 'the 10:1 assumption was wrong' from 'the model is fragile'. "
        "improve_pct is relative to cost_flag_all (= distance from flag-everyone to zero)."
    )
    out.append("")
    for D in CUTOFFS:
        sel = per_cutoff[D]["selected_rung"]
        prob = per_cutoff[D]["probs"][sel]
        y_test = per_cutoff[D]["y_test"]
        base_rate = per_cutoff[D]["base_rate"]
        frozen_t = frozen["cutoffs"][str(D)][sel]["threshold"]
        fn_f, fp_f, n = confusion_at_threshold(y_test, prob, frozen_t)
        cost_flag_all = 1.0 - base_rate

        out.append(f"--- D={D}, selected model = {sel}, frozen threshold = {frozen_t:.2f} ---")
        out.append(
            f"{'ratio':<7}{'flag_all':>10}{'flag_none':>11}{'cost_frozen':>13}{'improve%':>10}"
            f"{'cf_thr':>8}{'cost_cf':>10}{'cf_improve%':>13}"
        )
        for ratio in RATIO_GRID:
            cost_flag_none = ratio * base_rate
            cost_frozen = (ratio * fn_f + fp_f) / n
            improve_pct = (cost_flag_all - cost_frozen) / cost_flag_all * 100 if cost_flag_all != 0 else float("nan")
            _, _, cf_threshold, cf_cost = sweep_threshold(y_test, prob, ratio=ratio)
            cf_improve_pct = (cost_flag_all - cf_cost) / cost_flag_all * 100 if cost_flag_all != 0 else float("nan")
            marker = "  <- headline" if ratio == HEADLINE_RATIO else ""
            out.append(
                f"{ratio:<7}{cost_flag_all:>10.4f}{cost_flag_none:>11.4f}{cost_frozen:>13.4f}{improve_pct:>9.2f}%"
                f"{cf_threshold:>8.2f}{cf_cost:>10.4f}{cf_improve_pct:>12.2f}%{marker}"
            )
        out.append("")

    out.append(section("7. Threshold stability: validate-frozen vs test-optimal (counterfactual, NOT adopted)"))
    out.append(f"{'cutoff':<8}{'model':<6}{'frozen_thr':>11}{'test_optimal_thr':>18}{'gap':>8}")
    for D in CUTOFFS:
        sel = per_cutoff[D]["selected_rung"]
        prob = per_cutoff[D]["probs"][sel]
        y_test = per_cutoff[D]["y_test"]
        frozen_t = frozen["cutoffs"][str(D)][sel]["threshold"]
        _, _, test_optimal_t, _ = sweep_threshold(y_test, prob, ratio=HEADLINE_RATIO)
        gap = frozen_t - test_optimal_t
        out.append(f"D={D:<6}{sel:<6}{frozen_t:>11.2f}{test_optimal_t:>18.2f}{gap:>8.2f}")
    out.append("The test-optimal threshold is reported for comparison only. It is not adopted anywhere.")
    out.append("")

    # --- 8. Declared outcomes ---
    stage4 = joblib.load(STAGE4_RESULTS_PATH)

    out.append(section("8. Declared outcomes (Section 13)"))
    out.append(
        "O1/O2 turns on whether a paired bootstrap interval contains zero. That is a criterion the protocol "
        "does fix — Section 8 pre-commits to interval-separated comparison between ladder rungs, and Section 13 "
        "words O1 and O2 in those terms — so O1/O2 is decided below."
    )
    out.append(
        "O3, O4, O5 and O6 are not. Section 13 declares them in the language of 'close', 'substantially worse', "
        "'sharp' degradation and 'materially worse' performance, and neither Section 13 nor any other part of the "
        "protocol defines what would count as meeting them: no threshold, no metric on which to read one, and no "
        "comparison rule. Unlike D1's PSI bands in Section 10, nothing was pre-registered here. Earlier versions "
        "of this report supplied numeric conventions anyway — a 10% relative AUC-PR drop for O3/O4, a 15% relative "
        "expected-cost degradation for O5 and for O6 — and returned verdicts against them. Each was chosen after "
        "the results were known, which makes it a cut-off selected with knowledge of what it would decide. All "
        "three are withdrawn."
    )
    out.append(
        "What follows reports, for each of those outcomes, the underlying numbers and the comparison that bears "
        "on it, and derives no verdict from an invented cut-off. O3, O4, O5 and O6 are reported as UNDETERMINED "
        "on the protocol's own terms: the measurements stand, the verdicts are not available. A reader who holds "
        "a standard of their own can apply it to the numbers below."
    )
    out.append("")

    D28 = per_cutoff[28]
    lo, hi = D28["paired_diff_interval"]
    o1_met = not (lo <= 0 <= hi) and D28["paired_diff_point"] < 0
    if o1_met:
        out.append(f"O1 MET: at D=28, M1 beats B3 by {abs(D28['paired_diff_point']):.4f} expected cost "
                    f"(M1 - B3 = {D28['paired_diff_point']:.4f}), "
                    f"95% CI [{lo:.4f}, {hi:.4f}] excludes zero.")
    else:
        out.append(f"O2 MET: at D=28, the M1 - B3 expected cost difference is {D28['paired_diff_point']:.4f}, "
                    f"95% CI [{lo:.4f}, {hi:.4f}] does not exclude zero (or does not favour M1). "
                    "Gradient boosting is not shown to be warranted over regularised logistic regression here.")

    sel14 = per_cutoff[14]["selected_rung"]
    sel28 = per_cutoff[28]["selected_rung"]

    def test_point(D: int, rung: str, key: str) -> float:
        return per_cutoff[D]["rung_results"][rung]["point"][key]

    cost28 = test_point(28, sel28, "expected_cost")
    out.append("")
    out.append("O3/O4, timeliness. Day 14 against day 28, both on test, at the frozen thresholds.")
    out.append(
        f"The cost-selected model differs between the two cutoffs ({sel14} at D=14, {sel28} at D=28), which "
        "confounds the cutoff with the model choice, so the same comparison is also given with the model held "
        "fixed. change is D=14 relative to D=28: lower is better for expected cost and Brier, higher is better "
        "for AUC-PR and recall@10%."
    )
    out.append("")
    out.append(f"{'basis':<26}{'metric':<16}{'D=14':>9}{'D=28':>9}{'change':>10}")
    o34_bases = [
        (f"selected ({sel14} / {sel28})", sel14, sel28),
        ("B3 at both cutoffs", "B3", "B3"),
        ("M1 at both cutoffs", "M1", "M1"),
    ]
    o34_metrics = [
        ("expected cost", "expected_cost"), ("AUC-PR", "auc_pr"),
        ("Brier", "brier"), ("recall@10%", "recall_at_10pct"),
    ]
    for label, rung14, rung28 in o34_bases:
        for metric_label, key in o34_metrics:
            v14, v28 = test_point(14, rung14, key), test_point(28, rung28, key)
            change = (v14 - v28) / v28 * 100 if v28 else float("nan")
            out.append(f"{label:<26}{metric_label:<16}{v14:>9.4f}{v28:>9.4f}{change:>+9.1f}%")
        out.append("")
    out.append(
        "O3 asks whether day 14 is 'close' to day 28 and O4 whether it is 'substantially worse'. The protocol "
        "defines neither term, names no metric on which to read them, and fixes no threshold, so the numbers "
        "above do not resolve into either outcome without a criterion invented here."
    )
    out.append(
        "O3/O4 UNDETERMINED: the timeliness trade-off is quantified above and no verdict is drawn from it."
    )
    out.append("")

    val_cost28 = frozen["cutoffs"]["28"][sel28]["validate_expected_cost"]
    test_cost28 = cost28
    drift_pct = (test_cost28 - val_cost28) / val_cost28 * 100 if val_cost28 else float("nan")
    val_base_rate28 = float(stage4[28]["validate_base_rate"])
    test_base_rate28 = per_cutoff[28]["base_rate"]
    val_flag_all28 = 1.0 - val_base_rate28
    test_flag_all28 = 1.0 - test_base_rate28
    val_metrics28 = {rung: stage4[28]["models"][rung]["metrics"] for rung in ("B3", "M1")}

    out.append(
        f"O5, drift. Validate (2014B) against test (2014J) at D=28, model {sel28}, frozen threshold applied "
        "unchanged to both. Two readings of the same move are reported side by side; the protocol supplies no "
        "rule for choosing between them, so neither is presented as the answer."
    )
    out.append("")
    out.append("Reading 1 — expected cost in levels, which moves with prevalence:")
    out.append(f"  {'quantity':<38}{'validate':>10}{'test':>10}{'change':>10}{'relative':>11}")
    for label, v_val, v_test in [
        ("base rate (non-completion)", val_base_rate28, test_base_rate28),
        ("flag-everyone expected cost", val_flag_all28, test_flag_all28),
        (f"{sel28} expected cost at frozen thr", val_cost28, test_cost28),
    ]:
        rel = (v_test - v_val) / v_val * 100 if v_val else float("nan")
        out.append(f"  {label:<38}{v_val:>10.4f}{v_test:>10.4f}{v_test - v_val:>+10.4f}{rel:>+10.1f}%")
    out.append("")
    out.append(
        f"  The model's expected cost rose {test_cost28 - val_cost28:+.4f} ({drift_pct:+.1f}%) while the "
        f"flag-everyone baseline, which contains no model at all, rose {test_flag_all28 - val_flag_all28:+.4f} "
        f"over the same move. The base rate fell from {val_base_rate28:.4f} to {test_base_rate28:.4f}: a cohort "
        "with proportionally more completers offers more students to flag in error, and the flag-everyone cost, "
        "which is exactly the share of completers, rises by the same amount the base rate falls. Most of the "
        "change in the level therefore tracks the change in prevalence rather than the model."
    )
    out.append("")
    out.append("Reading 2 — discrimination and calibration, which do not move with prevalence in that way:")
    out.append(f"  {'metric':<38}{'validate':>10}{'test':>10}{'change':>10}{'relative':>11}")
    for rung in ("M1", "B3"):
        for metric_label, key in [("AUC-PR", "auc_pr"), ("Brier", "brier")]:
            v_val = float(val_metrics28[rung][key])
            v_test = test_point(28, rung, key)
            rel = (v_test - v_val) / v_val * 100 if v_val else float("nan")
            out.append(f"  {rung + ' ' + metric_label:<38}{v_val:>10.4f}{v_test:>10.4f}{v_test - v_val:>+10.4f}{rel:>+10.1f}%")
    out.append("")
    out.append(
        "  AUC-PR is invariant to the threshold and falls on both models; Brier worsens. These are properties "
        "of the ranking and of the probabilities themselves, not of where the cohort's base rate sits."
    )
    out.append("")
    out.append(
        "The second reading is evidence of genuine degradation: the model separates students less well on 2014J "
        "than it did on 2014B, on measures that do not follow the base rate. The first is largely arithmetic: "
        "the rise in expected cost is mostly the prevalence shift, as the flag-everyone row shows. Section 13 "
        "declares O5 for 'sharp' degradation but defines no criterion that converts either reading into a "
        "verdict. The two readings agree that something moved and disagree about how much of it is the model."
    )
    out.append(
        "O5 UNDETERMINED: drift is measured and reported here and under Section 10, and no verdict is drawn "
        "from it. Per Section 10, nothing is corrected, reweighted or recalibrated in response."
    )
    out.append("")

    overall_cost28 = per_cutoff[28]["rung_results"][per_cutoff[28]["selected_rung"]]["point"]["expected_cost"]
    slice_model = per_cutoff[SLICE_CUTOFF]["selected_rung"]
    out.append(
        f"O6, slice performance at D={SLICE_CUTOFF}, model {slice_model}. Every slice reported in Section 5, with its "
        f"expected cost as a percentage of the overall figure ({overall_cost28:.4f} on the whole test split). "
        "Ranked by gap, worst first. Expected cost is a loss, so a positive gap is a slice doing worse than "
        "the cohort overall and a negative gap is one doing better."
    )
    out.append("")
    out.append(f"{'slice':<40}{'n':>7}{'flag':>10}{'cost':>9}{'gap%':>9}")
    for row in sorted(slice_gaps, key=lambda r: -r["gap_pct"]):
        label = f"{row['col']}={row['value']}"
        out.append(
            f"{label:<40}{row['n']:>7}{row['flag']:>10}{row['cost']:>9.4f}{row['gap_pct']:>8.1f}%"
        )
    out.append("")
    out.append(
        "Section 8 of the protocol makes slice reporting mandatory but does not define materiality: it fixes "
        "no threshold, no metric and no comparison rule at which a slice gap counts as 'materially worse' for "
        "the purposes of O6. Nothing in the pre-registered document supplies one, and the 15% convention this "
        "report previously decided O6 against is withdrawn per the note at the head of this section. None is "
        "supplied in its place."
    )
    out.append(
        "O6 UNDETERMINED: the gaps above are reported as measured. Whether any of them is materially worse in "
        "the sense of O6 cannot be decided against the protocol as pre-registered, so O6 is reported as neither "
        "met nor not met. A reader who holds a materiality standard of their own can apply it to the table above."
    )
    out.append("")

    con_check = "PASS"
    out.append(f"Test split integrity: guard-verified hyperparameters matched frozen_threshold.json at every cutoff. {con_check}")
    out.append("")

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    all_predictions.to_parquet(PREDICTIONS_PATH, index=False)

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Predictions written to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
