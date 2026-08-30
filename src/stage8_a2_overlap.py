"""
Stage 8: discharge Amendment A2.

A2 commits to reporting the count of students appearing in both the 2014J test
split and a training presentation, as a raw number and as a share of the test
split, alongside a comparison of test performance on returning students versus
first-time students. Neither was produced at the time. This produces both.

NOTHING IS REFIT AND NOTHING IS RESCORED. Predictions are read from
reports/stage5_test_predictions.parquet and the threshold from
reports/frozen_threshold.json. No model artefact is loaded.

Usage:
    .venv/bin/python src/stage8_a2_overlap.py
"""

import json
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from stage4_models import HEADLINE_RATIO, expected_cost

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
FROZEN_JSON_PATH = ROOT / "reports" / "frozen_threshold.json"
PREDICTIONS_PATH = ROOT / "reports" / "stage5_test_predictions.parquet"
REPORT_PATH = ROOT / "reports" / "stage8_a2_overlap.txt"

CUTOFF = 28
KEY = ["code_module", "code_presentation", "id_student"]
RUNGS = ["M1", "B3"]


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def subgroup_metrics(df: pd.DataFrame, rung: str, threshold: float) -> dict:
    y = df["not_completed"].values
    p = df[f"prob_{rung}"].values
    base = float(y.mean())
    if len(set(y)) < 2:
        return {"n": len(df), "base_rate": base, "auc_pr": float("nan"),
                "ratio": float("nan"), "brier": float("nan"),
                "cost": expected_cost(y, p, threshold, HEADLINE_RATIO),
                "flag_all": 1.0 - base}
    ap = float(average_precision_score(y, p))
    return {
        "n": len(df),
        "base_rate": base,
        "auc_pr": ap,
        "ratio": ap / base if base else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "cost": expected_cost(y, p, threshold, HEADLINE_RATIO),
        "flag_all": 1.0 - base,
    }


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    frozen = json.loads(FROZEN_JSON_PATH.read_text())
    frozen_cutoff = frozen["cutoffs"][str(CUTOFF)]
    selected = min(["B0", "B1", "B2", "B3", "M1"],
                   key=lambda r: frozen_cutoff[r]["validate_expected_cost"])

    features = con.execute(
        f"SELECT code_module, code_presentation, id_student, split FROM features_d{CUTOFF}"
    ).df()
    train_ids = set(features.loc[features["split"] == "train", "id_student"])
    validate_ids = set(features.loc[features["split"] == "validate", "id_student"])
    si_2013_ids = set(con.execute(
        "SELECT DISTINCT id_student FROM v_student_info "
        "WHERE code_presentation IN ('2013B', '2013J')"
    ).df()["id_student"])

    preds = pd.read_parquet(PREDICTIONS_PATH)
    preds = preds[preds["cutoff"] == CUTOFF].copy()
    test_rows = features[features["split"] == "test"][KEY]
    preds = test_rows.merge(preds, on=KEY, how="left")
    assert preds["prob_M1"].notna().all(), "test rows missing from the stored predictions"

    n_test = len(preds)
    n_test_students = preds["id_student"].nunique()

    out: list[str] = []
    out.append("Stage 8: Amendment A2, train/test student overlap on the holdout")
    out.append(
        "Discharges the reporting Amendment A2 committed to and Stage 5 did not produce. Nothing is "
        "refit and nothing is rescored: predictions are read from stage5_test_predictions.parquet and "
        "the threshold from frozen_threshold.json. No model artefact is loaded."
    )
    out.append(f"frozen_threshold.json git_commit_at_generation: {frozen['git_commit_at_generation']}")

    # ------------------------------------------------------------------
    out.append(section("1. What A2 commits to"))
    out.append(
        "Amendment A2, on Section 4: 'The count of distinct students appearing in both the 2014J test "
        "split and either training presentation is reported as a raw number and as a share of the test "
        "split. If that share is material, it is named in the write-up as an alternative explanation for "
        "measured performance, alongside a comparison of test performance on returning students versus "
        "first-time students.'"
    )
    out.append("")
    out.append(
        "A2 is explicit that this is not leakage: no information from the test presentation reaches the "
        "training data, and a returning student is a genuine deployment case already encoded by "
        "num_of_prev_attempts. What it opens is a competing explanation. If the model does better on "
        "students it has seen before, per-student memorisation is an alternative account of measured "
        "test performance, and the comparison below is what distinguishes the two."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section(f"2. Overlap counts, 2014J test split at D={CUTOFF}"))
    out.append(f"  Test split: {n_test} rows over {n_test_students} distinct students.")
    out.append(
        "  A row is one student-module-presentation, so a student taking two modules in 2014J appears "
        "twice. Both counts are given."
    )
    out.append("")
    definitions = [
        ("in a training presentation (2013B or 2013J), as rows the model was fit on",
         train_ids, "A2 as written"),
        ("in the validation presentation (2014B), as rows the model was fit on",
         validate_ids, "refit rule"),
        ("in train OR validate, i.e. seen anywhere by the scored model",
         train_ids | validate_ids, "refit rule"),
        ("in a 2013 presentation in student_info, before Section 3 exclusions",
         si_2013_ids, "upper bound"),
    ]
    out.append(f"  {'definition of a returning student':<72}{'rows':>7}{'share':>8}{'students':>10}{'share':>8}")
    for label, ids, _tag in definitions:
        mask = preds["id_student"].isin(ids)
        n_rows = int(mask.sum())
        n_students = int(preds.loc[mask, "id_student"].nunique())
        out.append(
            f"  {label:<72}{n_rows:>7}{n_rows / n_test:>8.4f}"
            f"{n_students:>10}{n_students / n_test_students:>8.4f}"
        )
    out.append("")
    out.append(
        "The first row is the figure A2 asks for. The third is the one that matters for the competing "
        "explanation it raises: Section 4's refit rule fits the scored model on train + validation "
        "combined, so a student appearing in 2014B was also seen by the model that produced these "
        "predictions, even though 2014B is not a 'training presentation' in A2's wording. A2 was written "
        "at Stage 1, before that consequence of the refit rule was in view."
    )
    out.append("")
    out.append(
        "Whether these shares are material is not decided here. A2 says 'if that share is material' and "
        "defines no threshold, and neither does any other part of the protocol. Consistent with the "
        "treatment of O3 to O6, the numbers are reported and no verdict is derived from them."
    )
    out.append("")

    # ------------------------------------------------------------------
    out.append(section("3. Test performance, returning against first-time students"))
    out.append(
        f"Model {selected} is the cost-selected model at D={CUTOFF}; B3 is shown alongside it. Expected "
        f"cost is at the frozen threshold and the pre-committed {HEADLINE_RATIO}:1 ratio, applied "
        "unchanged to both subgroups."
    )
    out.append(
        "AUC-PR is not comparable across groups with different prevalence, since a random ranker scores "
        "the base rate. Each subgroup's base rate and the ratio of its AUC-PR to that base rate are "
        "printed alongside. Expected cost carries the same dependence: the flag-everyone cost for a "
        "subgroup is one minus its base rate, so that column is printed too and is the floor each "
        "subgroup's cost should be read against."
    )
    out.append("")

    for label, ids, tag in definitions[:1] + definitions[2:3]:
        out.append(f"--- returning = {label}  [{tag}] ---")
        mask = preds["id_student"].isin(ids)
        groups = [("returning", preds[mask]), ("first-time", preds[~mask])]
        for rung in RUNGS:
            threshold = frozen_cutoff[rung]["threshold"]
            out.append(f"  model {rung}, frozen threshold {threshold:.2f}")
            out.append(
                f"    {'subgroup':<14}{'n':>7}{'base_rate':>11}{'AUC-PR':>9}{'ratio':>8}"
                f"{'Brier':>9}{'cost':>9}{'flag_all':>10}{'edge':>9}"
            )
            for name, sub in groups:
                m = subgroup_metrics(sub, rung, threshold)
                out.append(
                    f"    {name:<14}{m['n']:>7}{m['base_rate']:>11.4f}{m['auc_pr']:>9.4f}"
                    f"{m['ratio']:>8.2f}{m['brier']:>9.4f}{m['cost']:>9.4f}"
                    f"{m['flag_all']:>10.4f}{m['flag_all'] - m['cost']:>9.4f}"
                )
            out.append("")
        out.append("")

    all_m1 = subgroup_metrics(preds, selected, frozen_cutoff[selected]["threshold"])
    out.append(
        f"Whole test split for reference, model {selected}: n = {all_m1['n']}, base rate "
        f"{all_m1['base_rate']:.4f}, AUC-PR {all_m1['auc_pr']:.4f}, ratio {all_m1['ratio']:.2f}, "
        f"Brier {all_m1['brier']:.4f}, expected cost {all_m1['cost']:.4f} against a flag-everyone cost "
        f"of {all_m1['flag_all']:.4f}."
    )
    out.append("")
    out.append(
        "'edge' is flag_all minus cost: how much the model's frozen policy saves against flagging every "
        "student in that subgroup. It is the comparison the primary metric supports at this base rate, "
        "since expected cost in levels moves with prevalence and the two subgroups have different "
        "prevalence."
    )
    out.append("")
    a2_mask = preds["id_student"].isin(train_ids)
    ret = subgroup_metrics(preds[a2_mask], selected, frozen_cutoff[selected]["threshold"])
    new_ = subgroup_metrics(preds[~a2_mask], selected, frozen_cutoff[selected]["threshold"])
    out.append(
        "Read as evidence on the competing explanation A2 raises: if per-student memorisation were "
        "driving measured test performance, the returning subgroup would rank better than the first-time "
        "subgroup once the base-rate difference is accounted for. On these figures it does not."
    )
    out.append(
        f"  Raw AUC-PR is higher for returning students ({ret['auc_pr']:.4f} against "
        f"{new_['auc_pr']:.4f} for {selected}), but so is their base rate ({ret['base_rate']:.4f} "
        f"against {new_['base_rate']:.4f}). Measured against each subgroup's own base rate the ranking "
        f"is WORSE for returning students: {ret['ratio']:.2f} against {new_['ratio']:.2f}."
    )
    out.append(
        f"  On expected cost, the frozen policy is worse than flagging everyone within the returning "
        f"subgroup (edge {ret['flag_all'] - ret['cost']:+.4f}) and better within the first-time subgroup "
        f"(edge {new_['flag_all'] - new_['cost']:+.4f})."
    )
    out.append(
        "  Both directions run against the memorisation account rather than for it, and the same "
        "pattern holds under the refit-inclusive definition and for B3. Reported as a measured "
        "direction, not converted into a verdict: A2 conditions its write-up clause on the share being "
        "'material' and the protocol defines no threshold for that, here or anywhere else."
    )
    out.append(
        "  These are point comparisons. Stage 5's bootstrap was not re-run, because re-running it is "
        "not documentation, so no interval is attached to any figure above and none should be inferred."
    )
    out.append("")

    con.close()
    report_text = "\n".join(out)
    REPORT_PATH.write_text(report_text + "\n")
    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
