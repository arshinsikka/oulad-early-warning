"""
Stage 4 report: formats reports/stage4_ladder.txt from the artefacts written
by stage4_ladder.py (reports/stage4_results.joblib and
reports/frozen_threshold.json). Does not refit anything.

Usage:
    .venv/bin/python src/stage4_report.py
"""

from pathlib import Path

import joblib

from stage2_cohort import CUTOFFS
from stage4_models import HEADLINE_RATIO, RATIO_GRID

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CACHE_PATH = ROOT / "reports" / "stage4_results.joblib"
REPORT_PATH = ROOT / "reports" / "stage4_ladder.txt"

HEADLINE_CUTOFF = 28
RUNG_NAMES = ["B0", "B1", "B2", "B3", "M1"]


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def best_rung(models: dict) -> str:
    return min(RUNG_NAMES, key=lambda r: models[r]["expected_cost"])


def main() -> None:
    results = joblib.load(RESULTS_CACHE_PATH)
    out: list[str] = []

    out.append("Stage 4 Model Ladder Report")
    out.append("Selection performed on VALIDATE only. Test split never read (guard-enforced).")

    # --- 1. Row counts and base rates ---
    out.append(section("1. Row counts and base rates"))
    out.append(f"{'cutoff':<10}{'train_n':>10}{'val_n':>10}{'train_base_rate':>18}{'val_base_rate':>16}")
    for D in CUTOFFS:
        r = results[D]
        out.append(
            f"D={D:<8}{r['train_n']:>10}{r['validate_n']:>10}"
            f"{r['train_base_rate']:>18.4f}{r['validate_base_rate']:>16.4f}"
        )
    out.append("")

    # --- 2. Metrics table, every model, every cutoff, on validate ---
    out.append(section("2. Validate metrics, every model, every cutoff"))
    out.append(
        f"{'cutoff':<8}{'model':<6}{'AUC-PR':>9}{'AUC-ROC':>9}{'Brier':>9}"
        f"{'cost@thr':>10}{'recall@5%':>11}{'recall@10%':>12}{'recall@20%':>12}"
    )
    for D in CUTOFFS:
        for rung in RUNG_NAMES:
            m = results[D]["models"][rung]["metrics"]
            cost = results[D]["models"][rung]["expected_cost"]
            out.append(
                f"D={D:<6}{rung:<6}{m['auc_pr']:>9.4f}{m['auc_roc']:>9.4f}{m['brier']:>9.4f}"
                f"{cost:>10.4f}{m['recall_at_5pct']:>11.4f}{m['recall_at_10pct']:>12.4f}{m['recall_at_20pct']:>12.4f}"
            )
        out.append("")

    # --- 3. Threshold and sensitivity ---
    out.append(section("3. Chosen threshold and cost sensitivity (+/- 0.05)"))
    out.append(
        f"{'cutoff':<8}{'model':<6}{'threshold':>10}{'cost@thr':>10}"
        f"{'cost@-0.05':>11}{'cost@+0.05':>11}"
    )
    for D in CUTOFFS:
        for rung in RUNG_NAMES:
            s = results[D]["models"][rung]
            out.append(
                f"D={D:<6}{rung:<6}{s['threshold']:>10.2f}{s['expected_cost']:>10.4f}"
                f"{s['cost_minus_005']:>11.4f}{s['cost_plus_005']:>11.4f}"
            )
        out.append("")
    out.append(
        "A flat minimum (cost@-0.05 and cost@+0.05 close to cost@thr) means the "
        "choice is robust; a sharp one means it is fragile."
    )
    out.append("")

    # --- 4. Threshold-vs-cost-ratio curve, best model per cutoff ---
    out.append(section("4. Threshold-vs-cost-ratio curve, best model at each cutoff"))
    for D in CUTOFFS:
        best = best_rung(results[D]["models"])
        out.append(f"--- D={D}, best model = {best} ---")
        out.append(f"{'ratio':<8}{'threshold':>10}{'expected_cost':>15}")
        for point in results[D]["models"][best]["ratio_curve"]:
            marker = "  <- headline (10:1)" if point["ratio"] == HEADLINE_RATIO else ""
            out.append(
                f"{point['ratio']:<8}{point['threshold']:>10.2f}{point['expected_cost']:>15.4f}{marker}"
            )
        out.append("")

    # --- 5. Seed stability ---
    out.append(section("5. Seed stability: ladder ranking by validate expected cost"))
    out.append("Ranking = rungs sorted ascending by cost at the seed-42 frozen threshold.")
    out.append("")
    for D in CUTOFFS:
        out.append(f"--- D={D} ---")
        rankings = results[D]["seed_rankings"]
        seeds = sorted(rankings.keys())
        for seed in seeds:
            out.append(f"  seed {seed:<6}: {' < '.join(rankings[seed])}")
        stable = len({tuple(rankings[s]) for s in seeds}) == 1
        out.append(f"  ranking stable across all seeds: {'YES' if stable else 'NO'}")
        out.append("")

    # --- 6. B3 coefficients ---
    out.append(section("6. B3: selected C and top 15 coefficients by |value|"))
    for D in CUTOFFS:
        b3 = results[D]["models"]["B3"]
        out.append(f"--- D={D}, C={b3['C']}, intercept={b3['intercept']:.4f} ---")
        out.append(f"{'feature':<40}{'coef':>10}")
        for name, coef in b3["top_coefficients"]:
            out.append(f"{name:<40}{coef:>10.4f}")
        out.append("")

    # --- 7. M1 vs B3 at day 28 ---
    out.append(section("7. M1 vs B3, expected cost, validate, D=28"))
    b3_cost = results[HEADLINE_CUTOFF]["models"]["B3"]["expected_cost"]
    m1_cost = results[HEADLINE_CUTOFF]["models"]["M1"]["expected_cost"]
    diff = b3_cost - m1_cost
    if m1_cost < b3_cost:
        verdict = f"M1 beat B3 by {diff:.4f} expected cost (lower is better)."
    elif m1_cost > b3_cost:
        verdict = f"M1 did NOT beat B3: M1 is worse by {-diff:.4f} expected cost."
    else:
        verdict = "M1 and B3 tied exactly on expected cost."
    out.append(f"B3 expected cost: {b3_cost:.4f}")
    out.append(f"M1 expected cost: {m1_cost:.4f}")
    out.append(verdict)
    out.append("Raw validate difference only. No bootstrap interval (that is Stage 5, on test).")
    out.append("")

    # --- 8. Accuracy vs timeliness ---
    out.append(section("8. Accuracy vs timeliness: best model at D=14, D=28, D=56"))
    out.append(f"{'cutoff':<8}{'best_model':<12}{'AUC-PR':>9}{'AUC-ROC':>9}{'Brier':>9}"
               f"{'cost@thr':>10}{'threshold':>11}{'recall@10%':>12}")
    for D in CUTOFFS:
        best = best_rung(results[D]["models"])
        s = results[D]["models"][best]
        m = s["metrics"]
        out.append(
            f"D={D:<6}{best:<12}{m['auc_pr']:>9.4f}{m['auc_roc']:>9.4f}{m['brier']:>9.4f}"
            f"{s['expected_cost']:>10.4f}{s['threshold']:>11.2f}{m['recall_at_10pct']:>12.4f}"
        )
    out.append("")
    out.append(
        "This is the trade-off curve the three cutoffs exist to produce: how much "
        "predictive accuracy and cost is given up by acting earlier."
    )
    out.append("")

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
