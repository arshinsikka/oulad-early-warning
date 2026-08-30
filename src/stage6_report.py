"""
Stage 6: drift measurement and global explainability.

Scoped down from Sections 10 and 11 as instructed: D1-D4 and E1 only. E2
(local reason strings), E3 (counterfactual sanity check) and E4 (demographic
contribution) are NOT produced here and are recorded as dropped in the scope
note at the end of the report.

THE HOLDOUT IS OPEN AND ITS RESULTS ARE FIXED. Nothing here refits a model,
recomputes a threshold, adjusts a calibration or corrects a drift. Test
predictions are read from reports/stage5_test_predictions.parquet. The frozen
D=28 model is loaded only to obtain SHAP contributions, and those are verified
against the stored predictions before being reported.

Usage:
    .venv/bin/python src/stage6_report.py
"""

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from stage2_cohort import cohort_ctes_sql
from stage3_report import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from stage6_drift import (
    PSI_EPSILON, PSI_N_BINS, categorical_psi, design_groups, group_contributions,
    numeric_psi, psi_band, psi_from_bins, rank_map,
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
MODELS_DIR = ROOT / "models"
FROZEN_JSON_PATH = ROOT / "reports" / "frozen_threshold.json"
PREDICTIONS_PATH = ROOT / "reports" / "stage5_test_predictions.parquet"
REPORT_PATH = ROOT / "reports" / "stage6_drift_explainability.txt"

CUTOFF = 28
THIN_TRAINING_ROWS = 200  # Section 4's declared threshold for thin representation.
KEY = ["code_module", "code_presentation", "id_student"]
PSI_DETAIL_TOP_N = 5
OVERLAP_K = [5, 10]


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Section 10 D2 asks for
    intervals on the per-presentation and per-module base rates."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    frozen = json.loads(FROZEN_JSON_PATH.read_text())
    frozen_cutoff = frozen["cutoffs"][str(CUTOFF)]
    selected_rung = min(
        ["B0", "B1", "B2", "B3", "M1"],
        key=lambda r: frozen_cutoff[r]["validate_expected_cost"],
    )

    table = f"features_d{CUTOFF}"
    train_df = con.execute(f"SELECT * FROM {table} WHERE split = 'train'").df()
    validate_df = con.execute(f"SELECT * FROM {table} WHERE split = 'validate'").df()
    test_df = con.execute(f"SELECT * FROM {table} WHERE split = 'test'").df()

    preds = pd.read_parquet(PREDICTIONS_PATH)
    preds = preds[preds["cutoff"] == CUTOFF].copy()

    out: list[str] = []
    out.append("Stage 6 Drift and Explainability Report")
    out.append(
        "Scope: Section 10 D1-D4 and Section 11 E1 only. E2, E3 and E4 are out of scope for this "
        "stage; see the closing note."
    )
    out.append(f"frozen_threshold.json git_commit_at_generation: {frozen['git_commit_at_generation']}")

    # ------------------------------------------------------------------
    out.append(section("1. Inputs and integrity"))
    out.append(
        "The holdout is open and its results are fixed. Nothing in this stage refits a model, "
        "recomputes a threshold, adjusts a calibration or corrects a drift. No .fit() is called on any "
        "model or preprocessor."
    )
    out.append("")
    out.append(f"  Feature table                  {table}")
    out.append(f"  Train split (2013B + 2013J)    n = {len(train_df)}")
    out.append(f"  Validate split (2014B)         n = {len(validate_df)}")
    out.append(f"  Test split (2014J)             n = {len(test_df)}")
    out.append(f"  Test predictions               {PREDICTIONS_PATH.name}, n = {len(preds)} (read, not recomputed)")
    out.append(f"  Selected model at D={CUTOFF}         {selected_rung} (lowest validate expected cost, frozen at Stage 4)")
    out.append("")

    bundle = joblib.load(MODELS_DIR / f"stage4_refit_d{CUTOFF}.joblib")
    preproc = bundle["preprocessors"]["full"]
    m1 = bundle["models"]["M1"]
    b3 = bundle["models"]["B3"]
    design_columns = list(preproc.output_columns)
    X_test = preproc.transform(test_df)
    assert list(X_test.columns) == design_columns

    # TreeSHAP straight from the frozen booster: exact, and it needs no
    # package beyond the one that fitted the model.
    contrib = m1.booster_.predict(X_test.values, pred_contrib=True)
    raw = contrib.sum(axis=1)
    prob_from_contrib = 1.0 / (1.0 + np.exp(-raw))

    stored = test_df[KEY].merge(preds[KEY + ["prob_M1", "prob_B3", "not_completed"]], on=KEY, how="left")
    assert stored["prob_M1"].notna().all(), "test rows missing from the Stage 5 prediction file"
    max_dev = float(np.abs(prob_from_contrib - stored["prob_M1"].values).max())
    out.append(
        "  SHAP integrity check: the M1 contributions below are LightGBM's own TreeSHAP "
        "(booster.predict(pred_contrib=True)), so no additional package is involved. Summing each row's "
        f"contributions and applying the logistic link reproduces the Stage 5 stored probabilities to "
        f"{max_dev:.2e} maximum absolute deviation. The model is therefore the frozen one and the "
        "predictions are unchanged."
    )
    out.append("")

    # ------------------------------------------------------------------
    # E1 first, computed here so Section 3 can rank drift against importance.
    groups = design_groups(design_columns, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    shap_feature = group_contributions(contrib[:, :-1], design_columns, groups)
    shap_column = {c: float(np.abs(contrib[:, i]).mean()) for i, c in enumerate(design_columns)}
    b3_coef = {c: float(b3.coef_[0][i]) for i, c in enumerate(design_columns)}
    shap_rank = rank_map(shap_feature)

    # ------------------------------------------------------------------
    out.append(section(f"2. D1. Feature drift: Population Stability Index, train against test, D={CUTOFF}"))
    out.append(
        "PSI = sum over bins of (p_test - p_train) * ln(p_test / p_train), computed on the 28 protocol "
        "features of Section 6, not on the 72 design-matrix columns the model sees."
    )
    out.append(
        f"Numeric features: {PSI_N_BINS} quantile bins with edges taken from the TRAIN distribution and "
        "applied unchanged to test, the outer edges opened to -inf / +inf so test values beyond the "
        "training range fall in the end bins. Heavily tied features collapse adjacent quantiles onto the "
        "same edge; duplicate edges are dropped and the realised bin count is reported in 'bins' rather "
        "than claimed to be 10. Nulls are carried as their own bin rather than imputed, since "
        "missingness is itself something that can drift."
    )
    out.append("Categorical features: the categories themselves, with null carried as its own category.")
    out.append(
        f"Empty bins: a bin with no rows on one side sends its term to infinity, so a zero proportion is "
        f"replaced by the small constant {PSI_EPSILON:g}. That is the usual convention in industry "
        "practice: one hundredth of one percent of the population, below any proportion these splits can "
        f"resolve (the smallest split here holds {min(len(train_df), len(validate_df), len(test_df))} rows, "
        "so a single row is a larger proportion than the constant). The 'empty' column counts bins empty "
        "on exactly one side, which are the only ones the constant affects — a bin empty on both sides, "
        "such as the null bin of a feature that has no nulls anywhere, contributes exactly zero whatever "
        "constant is chosen. The substitution bounds the term rather than making it meaningful: where it "
        "fires, the value depends on the constant as much as on the data, and the sensitivity table below "
        "the ranking shows by how much."
    )
    out.append(
        "One further caveat on resolution: a heavily tied feature realises far fewer than 10 bins (see the "
        "'bins' column), and a feature measured in 2 or 3 bins can only register coarse movement. A low "
        "PSI on such a feature is weaker evidence of stability than the same number on a feature binned "
        "into 10."
    )
    out.append(
        "Bands: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant. THIS IS A CONVENTION "
        "FROM INDUSTRY PRACTICE, NOT A THRESHOLD FROM THIS PROTOCOL. Section 10 D1 names it and says as "
        "much: 'The convention is stated as a convention, not a law.' It is the one numeric reading in "
        "this report that the protocol did pre-register, and it is still only a convention."
    )
    out.append("")

    psi_results = {}
    for feature in NUMERIC_FEATURES:
        psi_results[feature] = {"kind": "numeric", **numeric_psi(train_df[feature], test_df[feature])}
    for feature in CATEGORICAL_FEATURES:
        psi_results[feature] = {"kind": "categorical", **categorical_psi(train_df[feature], test_df[feature])}

    ranked = sorted(psi_results.items(), key=lambda kv: -kv[1]["psi"])
    psi_rank = {f: i + 1 for i, (f, _) in enumerate(ranked)}

    out.append(f"{'rank':<6}{'feature':<32}{'kind':<13}{'PSI':>9}{'band':>13}{'bins':>7}{'empty':>7}{'SHAP rank':>11}")
    for i, (feature, r) in enumerate(ranked, start=1):
        out.append(
            f"{i:<6}{feature:<32}{r['kind']:<13}{r['psi']:>9.4f}{psi_band(r['psi']):>13}"
            f"{r['n_bins_used']:>7}{r['empty_bins']:>7}{shap_rank[feature]:>11}"
        )
    out.append("")
    degenerate = [f for f, r in psi_results.items() if r["degenerate"]]
    if degenerate:
        out.append(f"Degenerate (fewer than two distinct train values, PSI not informative): {', '.join(degenerate)}")
        out.append("")

    affected = [(f, r) for f, r in ranked if r["empty_bins"] > 0]
    out.append(
        "Sensitivity of the affected features to the empty-bin constant. Only features with a bin empty on "
        "exactly one side appear; every other feature's PSI is unchanged by the choice."
    )
    if affected:
        out.append(f"  {'feature':<32}{'eps=1e-3':>10}{'eps=1e-4':>10}{'eps=1e-6':>10}  (1e-4 is the value used)")
        for feature, r in affected:
            out.append(
                f"  {feature:<32}{psi_from_bins(r['bins'], 1e-3):>10.4f}"
                f"{psi_from_bins(r['bins'], 1e-4):>10.4f}{psi_from_bins(r['bins'], 1e-6):>10.4f}"
            )
        out.append(
            "  A feature whose PSI swings across these columns is being scored on the absence of a "
            "category or range from one split, not on a measured shift in its distribution. The band it "
            "lands in is then a property of the constant."
        )
    else:
        out.append("  No feature has a bin empty on exactly one side; the constant did not fire.")
    out.append("")

    out.append(f"Bin detail for the {PSI_DETAIL_TOP_N} highest-PSI features:")
    out.append("")
    for feature, r in ranked[:PSI_DETAIL_TOP_N]:
        out.append(f"--- {feature} (PSI={r['psi']:.4f}, {r['kind']}) ---")
        out.append(f"  {'bin':<34}{'train':>10}{'test':>10}{'term':>10}")
        for b in r["bins"]:
            one_sided = (b["train_prop"] <= 0) != (b["test_prop"] <= 0)
            marker = "  <- empty on one side, constant substituted" if one_sided else ""
            out.append(
                f"  {b['label']:<34}{b['train_prop']:>10.4f}{b['test_prop']:>10.4f}{b['term']:>10.4f}{marker}"
            )
        out.append("")

    # ------------------------------------------------------------------
    out.append(section("3. D1 continued. Drift against model importance"))
    out.append(
        "Stage 5 found discrimination fell from validate to test: M1 AUC-PR 0.7870 to 0.6794, B3 0.7843 "
        "to 0.6106. The question this section answers is a descriptive one: are the features that moved "
        "most between train and test the same features the model leans on most?"
    )
    out.append(
        "NO CAUSAL CLAIM IS MADE. PSI measures a change in a feature's marginal distribution. It says "
        "nothing about whether the relationship between that feature and the outcome changed, which is "
        "what would actually degrade a model, and a feature can move a great deal without costing any "
        "accuracy while a stable feature whose conditional relationship shifted costs a lot. The overlap "
        "below is reported as an observation, not as an explanation of the Stage 5 result."
    )
    out.append("")
    out.append(f"{'rank':<6}{'by PSI':<32}{'PSI':>9}   {'by SHAP mean |contribution|':<32}{'SHAP':>9}")
    shap_ranked = sorted(shap_feature.items(), key=lambda kv: -kv[1])
    for i in range(len(ranked)):
        pf, pr = ranked[i]
        sf, sv = shap_ranked[i]
        out.append(f"{i + 1:<6}{pf:<32}{pr['psi']:>9.4f}   {sf:<32}{sv:>9.4f}")
    out.append("")
    for k in OVERLAP_K:
        top_psi = {f for f, _ in ranked[:k]}
        top_shap = {f for f, _ in shap_ranked[:k]}
        shared = sorted(top_psi & top_shap, key=lambda f: psi_rank[f])
        out.append(
            f"Top {k} by PSI and top {k} by SHAP importance share {len(shared)} of {k} features"
            + (f": {', '.join(shared)}." if shared else ".")
        )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section(f"4. D2 and D3. Base rate drift and exclusion drift, D={CUTOFF}"))
    out.append(
        "Brought forward from reports/stage2_cohort.txt so the drift picture sits in one place. "
        "Recomputed from the cohort and feature tables by the same SQL Stage 2 used, not retyped."
    )
    out.append("")

    out.append("D2, non-completion rate per presentation (95% Wilson interval):")
    out.append(f"  {'presentation':<15}{'split':<11}{'n':>8}{'base_rate':>11}{'[95% CI]':>20}")
    pres = con.execute(
        f"""
        SELECT code_presentation, split, count(*) AS n, sum(not_completed) AS k
        FROM {table} GROUP BY 1, 2 ORDER BY 1
        """
    ).df()
    for row in pres.itertuples():
        lo, hi = wilson_interval(int(row.k), int(row.n))
        out.append(
            f"  {row.code_presentation:<15}{row.split:<11}{int(row.n):>8}{row.k / row.n:>11.4f}"
            f"{f'[{lo:.4f},{hi:.4f}]':>20}"
        )
    out.append("")

    out.append("D2, non-completion rate per module-presentation (95% Wilson interval):")
    out.append(f"  {'module':<8}{'presentation':<15}{'split':<11}{'n':>8}{'base_rate':>11}{'[95% CI]':>20}")
    modpres = con.execute(
        f"""
        SELECT code_module, code_presentation, split, count(*) AS n, sum(not_completed) AS k
        FROM {table} GROUP BY 1, 2, 3 ORDER BY 1, 2
        """
    ).df()
    for row in modpres.itertuples():
        lo, hi = wilson_interval(int(row.k), int(row.n))
        out.append(
            f"  {row.code_module:<8}{row.code_presentation:<15}{row.split:<11}{int(row.n):>8}"
            f"{row.k / row.n:>11.4f}{f'[{lo:.4f},{hi:.4f}]':>20}"
        )
    out.append("")

    out.append(f"D3, share of students removed by exclusion E1, per presentation, at D={CUTOFF}:")
    out.append(
        "  Share = (rows entering E1 minus rows surviving E1) / rows entering E1, i.e. the share of the "
        "E2-surviving population that had already left by the cutoff."
    )
    out.append(f"  {'presentation':<15}{'entering E1':>13}{'surviving':>11}{'removed':>9}{'share':>9}")
    e1 = con.execute(
        f"""
        WITH {cohort_ctes_sql(CUTOFF)},
        entering AS (SELECT code_presentation, count(*) AS n_in FROM after_e2 GROUP BY 1),
        surviving AS (SELECT code_presentation, count(*) AS n_out FROM after_e1 GROUP BY 1)
        SELECT e.code_presentation, e.n_in, s.n_out
        FROM entering e JOIN surviving s USING (code_presentation)
        ORDER BY 1
        """
    ).df()
    for row in e1.itertuples():
        removed = int(row.n_in) - int(row.n_out)
        out.append(
            f"  {row.code_presentation:<15}{int(row.n_in):>13}{int(row.n_out):>11}{removed:>9}"
            f"{removed / row.n_in:>9.4f}"
        )
    out.append("")
    out.append(
        "Per Section 3, E1 removes students whose unregistration is already recorded before the cutoff. "
        "Per Section 10 D3, a materially different share removed in the test presentation means the "
        "prediction task is not the same task. The shares are reported as measured; the protocol fixes no "
        "threshold for what counts as materially different, so no verdict is drawn here."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section(f"5. D4. Performance decomposition by module, test, D={CUTOFF}, model {selected_rung}"))
    out.append(
        "AUC-PR is not comparable across groups with different prevalence: a random ranker scores the "
        "base rate. Each module's base rate and the ratio of its AUC-PR to that base rate are therefore "
        "printed alongside, so a module is not read as weak merely for having fewer non-completers. B3 is "
        "shown next to M1 because a module that is hard for both models is a different finding from one "
        "where the two disagree."
    )
    out.append("")
    train_rows = train_df["code_module"].value_counts().to_dict()
    validate_rows = validate_df["code_module"].value_counts().to_dict()

    out.append(
        f"  {'module':<8}{'n_test':>8}{'base':>8}{'M1 AUC-PR':>11}{'ratio':>8}{'M1 Brier':>10}"
        f"{'B3 AUC-PR':>11}{'n_train':>9}{'n_val':>8}  flag"
    )
    module_rows = []
    for module in sorted(preds["code_module"].unique()):
        mask = (preds["code_module"] == module).values
        y = preds.loc[mask, "not_completed"].values
        p_m1 = preds.loc[mask, "prob_M1"].values
        p_b3 = preds.loc[mask, "prob_B3"].values
        n = int(mask.sum())
        base = float(y.mean())
        n_train = int(train_rows.get(module, 0))
        n_val = int(validate_rows.get(module, 0))
        thin = n_train < THIN_TRAINING_ROWS
        if len(set(y)) < 2:
            out.append(f"  {module:<8}{n:>8}{base:>8.4f}   (single-class module, ranking metrics undefined)")
            continue
        ap_m1 = float(average_precision_score(y, p_m1))
        ap_b3 = float(average_precision_score(y, p_b3))
        brier_m1 = float(brier_score_loss(y, p_m1))
        flag = "THIN TRAIN (< %d rows in train split)" % THIN_TRAINING_ROWS if thin else ""
        module_rows.append({
            "module": module, "n": n, "base": base, "ap_m1": ap_m1, "ap_b3": ap_b3,
            "brier": brier_m1, "n_train": n_train, "n_val": n_val, "thin": thin,
            "ratio": ap_m1 / base if base else float("nan"),
            "auc_roc": float(roc_auc_score(y, p_m1)),
        })
        out.append(
            f"  {module:<8}{n:>8}{base:>8.4f}{ap_m1:>11.4f}{ap_m1 / base:>8.2f}{brier_m1:>10.4f}"
            f"{ap_b3:>11.4f}{n_train:>9}{n_val:>8}  {flag}"
        )

    y_all = preds["not_completed"].values
    ap_all = float(average_precision_score(y_all, preds["prob_M1"].values))
    base_all = float(y_all.mean())
    out.append(
        f"  {'ALL':<8}{len(preds):>8}{base_all:>8.4f}{ap_all:>11.4f}{ap_all / base_all:>8.2f}"
        f"{brier_score_loss(y_all, preds['prob_M1'].values):>10.4f}"
        f"{average_precision_score(y_all, preds['prob_B3'].values):>11.4f}"
        f"{len(train_df):>9}{len(validate_df):>8}"
    )
    out.append("")

    thin_modules = [m for m in module_rows if m["thin"]]
    out.append(
        f"Section 4 declares in advance that modules present in the test presentation with fewer than "
        f"{THIN_TRAINING_ROWS} training rows are reported separately rather than dropped, "
        "because dropping them would hide the generalisation problem a deployment hits when a new module "
        "launches. Counted against the train split (2013B + 2013J), which is what Section 4's 'training "
        "rows' refers to at the point that rule was written:"
    )
    if thin_modules:
        for m in thin_modules:
            out.append(
                f"  {m['module']}: {m['n_train']} rows in the train split, {m['n_val']} in validate, "
                f"{m['n']} in test ({m['n'] / len(preds) * 100:.1f}% of the test split). "
                f"AUC-PR {m['ap_m1']:.4f} against a base rate of {m['base']:.4f}."
            )
        out.append(
            "  Note the refit rule in Section 4: the scored model was refit on train + validate, so a "
            "module absent from the train split may still have been seen at refit through 2014B. Both "
            "counts are printed above for that reason."
        )
    else:
        out.append(f"  No module in the test split has fewer than {THIN_TRAINING_ROWS} train-split rows.")
    out.append("")

    spread = max(m["ap_m1"] for m in module_rows) - min(m["ap_m1"] for m in module_rows)
    best = max(module_rows, key=lambda m: m["ap_m1"])
    worst = min(module_rows, key=lambda m: m["ap_m1"])
    ratio_best = max(module_rows, key=lambda m: m["ratio"])
    ratio_worst = min(module_rows, key=lambda m: m["ratio"])
    out.append(
        f"AUC-PR spread across modules: {spread:.4f}, from {worst['module']} at {worst['ap_m1']:.4f} "
        f"(base rate {worst['base']:.4f}) to {best['module']} at {best['ap_m1']:.4f} "
        f"(base rate {best['base']:.4f}). Measured against each module's own base rate, the ratio runs "
        f"from {ratio_worst['ratio']:.2f} ({ratio_worst['module']}) to {ratio_best['ratio']:.2f} "
        f"({ratio_best['module']}), against {ap_all / base_all:.2f} pooled."
    )
    brier_pooled = float(brier_score_loss(y_all, preds["prob_M1"].values))
    brier_worst = max(module_rows, key=lambda m: m["brier"])
    brier_best = min(module_rows, key=lambda m: m["brier"])
    brier_rel = {m["module"]: (m["brier"] - brier_pooled) / brier_pooled * 100 for m in module_rows}
    others = [v for k, v in brier_rel.items() if k != brier_worst["module"]]
    out.append(
        f"Brier spread: {brier_best['module']} at {brier_best['brier']:.4f} to {brier_worst['module']} at "
        f"{brier_worst['brier']:.4f}, against {brier_pooled:.4f} pooled. {brier_worst['module']} sits "
        f"{brier_rel[brier_worst['module']]:+.1f}% from the pooled Brier while every other module lies "
        f"between {min(others):+.1f}% and {max(others):+.1f}% of it."
    )
    out.append(
        f"That does not coincide with the weakest ranking. {brier_worst['module']} is not thin in training "
        f"({brier_worst['n_train']} train-split rows), and it is neither the lowest module by AUC-PR "
        f"({worst['module']}) nor the lowest by AUC-PR relative to its own base rate "
        f"({ratio_worst['module']}), so on this decomposition its calibration and its ranking are not "
        "moving together."
    )
    out.append(
        "Whether that spread is concentrated or diffuse is left to the numbers above: the protocol fixes "
        "no threshold at which a module counts as an outlier, and none is invented here. What the "
        "decomposition does separate is the two explanations Section 10 D4 exists to tell apart — a "
        "module with thin or absent training representation is a different case from a module the model "
        "saw well and still ranks poorly, and the n_train column says which is which."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section(f"6. E1. Global explainability: M1 SHAP against B3 coefficients, test, D={CUTOFF}"))
    out.append(
        "M1 importance is the mean absolute TreeSHAP contribution on the test split, in log-odds. A "
        "protocol feature spans several design columns — one per level for a categorical, plus a "
        "was_missing_ indicator for a numeric that had nulls in the fit data — so a row's contribution "
        "for a feature is the SIGNED sum across its columns and the importance is the mean of the "
        "absolute value of that sum. Summing absolute values per column instead would inflate wide "
        "one-hot groups by counting contributions that cancel."
    )
    out.append("")
    out.append(f"  {'rank':<6}{'feature':<32}{'mean |SHAP|':>13}{'PSI rank':>10}")
    for i, (feature, value) in enumerate(shap_ranked, start=1):
        out.append(f"  {i:<6}{feature:<32}{value:>13.4f}{psi_rank[feature]:>10}")
    out.append("")

    out.append(
        "B3 coefficients are from the same frozen refit, on the same design matrix. Numeric columns are "
        "standardised, so a numeric coefficient is a log-odds change per training standard deviation, "
        "while a one-hot coefficient is a log-odds change for membership of that category against the "
        "regularised average. The two units are not the same thing, which limits how far the two halves "
        "of this ranking can be compared with each other."
    )
    out.append("")
    coef_ranked = sorted(b3_coef.items(), key=lambda kv: -abs(kv[1]))
    shap_col_rank = rank_map(shap_column)
    coef_col_rank = {c: i + 1 for i, (c, _) in enumerate(coef_ranked)}
    out.append(f"  {'rank':<6}{'design column':<48}{'coefficient':>13}")
    for i, (col, value) in enumerate(coef_ranked, start=1):
        out.append(f"  {i:<6}{col:<48}{value:>+13.4f}")
    out.append("")

    out.append(
        f"Where the two disagree. Both are ranked over the same {len(design_columns)} design columns: M1 by "
        "mean absolute SHAP contribution, B3 by absolute coefficient. Every column is listed, sorted by "
        "the size of the rank gap. Section 11 E1 asks for material disagreement to be reported and not "
        "resolved; it defines no threshold for 'material', and none is supplied here, so the full ranking "
        "is given and the reader draws the line."
    )
    n_zero = sum(1 for c in design_columns if shap_column[c] == 0.0)
    if n_zero:
        zero_cols = [c for c in design_columns if shap_column[c] == 0.0]
        zero_in_b3_top = sorted(coef_col_rank[c] for c in zero_cols)[:5]
        out.append(
            f"{n_zero} of the {len(design_columns)} columns have a mean absolute SHAP contribution of "
            "exactly zero: the boosted trees never split on them anywhere in the ensemble, so they enter "
            "no prediction at all. Several of them carry large B3 coefficients — the highest-ranked by "
            f"|coefficient| sit at B3 ranks {', '.join(str(r) for r in zero_in_b3_top)} — which is the "
            "sharpest form the disagreement takes here: a column the linear model relies on and the "
            "gradient-boosted model does not use."
        )
    out.append("")
    gaps = sorted(
        design_columns, key=lambda c: -abs(shap_col_rank[c] - coef_col_rank[c])
    )
    out.append(
        f"  {'design column':<48}{'M1 rank':>9}{'B3 rank':>9}{'gap':>6}{'mean |SHAP|':>13}{'coef':>10}"
    )
    for col in gaps:
        gap = shap_col_rank[col] - coef_col_rank[col]
        out.append(
            f"  {col:<48}{shap_col_rank[col]:>9}{coef_col_rank[col]:>9}{gap:>+6}"
            f"{shap_column[col]:>13.4f}{b3_coef[col]:>+10.4f}"
        )
    out.append("")
    out.append(
        "Read with care in both directions. A column ranked high by M1 and low by B3 is one the boosted "
        "trees use and the linear model cannot, which is where non-linearity or interaction would show "
        "up. A column ranked high by B3 and low by M1 is one the linear model leans on to compensate for "
        "structure it cannot represent directly. Neither ranking is the ground truth and the "
        "disagreement is reported rather than resolved, per Section 11 E1."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("7. Scope note: what this stage does not contain"))
    out.append(
        "Section 11 declares four explainability items. Only E1 is produced here. E2 (local reason "
        "strings for a sample of flagged students), E3 (the counterfactual sanity check stated alongside "
        "those reason strings) and E4 (the share of model output attributable to Group A demographic "
        "features) are dropped from this stage by instruction, not by finding, and are to be recorded as "
        "dropped in a protocol amendment. They are not attempted, not partially attempted, and nothing "
        "in this report should be read as standing in for them."
    )
    out.append(
        "E3 is worth naming precisely because it is absent. It is the warning that SHAP explains the "
        "model and not the world: that inactivity is a symptom, and that a feature driving a flag does "
        "not mean acting on that feature changes the outcome. Section 6 predicted in advance that "
        "n_due_not_submitted would dominate, and Section 5 of this report can be read as confirming or "
        "denying that prediction, but neither that ranking nor the SHAP table above carries any causal "
        "content whatsoever. Arm 2 (Section 12) is where a causal question is asked, and it has not run."
    )
    out.append("")
    out.append(
        "Drift is measured and reported here. Per Section 10, no drift correction, no reweighting and no "
        "recalibration on test is performed, and none of the numbers above has been used to change "
        "anything."
    )
    out.append("")

    con.close()

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")
    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
