"""
Stage 3 report: feature table diagnostics, leakage sentinel, and a sample
export for manual tracing (VERIFICATION STOP 2). Report only.

Usage:
    .venv/bin/python src/stage3_report.py
"""

from pathlib import Path

import duckdb

from stage2_cohort import CUTOFFS

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
REPORT_PATH = ROOT / "reports" / "stage3_features.txt"
SAMPLE_CSV_PATH = ROOT / "reports" / "stage3_sample_d28.csv"

HEADLINE_CUTOFF = 28

CATEGORICAL_FEATURES = [
    "gender", "region", "highest_education", "imd_band", "age_band",
    "disability", "code_module",
]

NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits",
    "total_clicks", "active_days", "clicks_last_7d",
    "mean_clicks_per_active_day", "days_registered_before_start",
    "days_since_last_activity", "click_slope_daily",
    "second_half_click_ratio", "longest_activity_gap",
    "distinct_activity_types", "distinct_materials", "assessment_material_share",
    "n_assessments_submitted", "mean_score_submitted", "min_score_submitted",
    "mean_submission_lateness", "n_due_not_submitted",
    "clicks_percentile", "score_percentile",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
assert len(ALL_FEATURES) == 28, f"expected 28 features, got {len(ALL_FEATURES)}"


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=False)
    out: list[str] = []

    out.append("Stage 3 Feature Report")
    out.append(f"Database: {DB_PATH}")

    # --- 1. Row count vs cohort ---
    out.append(section("1. Row count: features table vs cohort table"))
    out.append(f"{'cutoff':<10}{'features':>12}{'cohort':>12}{'match':>8}")
    for D in CUTOFFS:
        feat_n = con.execute(f"SELECT count(*) FROM features_d{D}").fetchone()[0]
        cohort_n = con.execute(f"SELECT count(*) FROM cohort_d{D}").fetchone()[0]
        match = "PASS" if feat_n == cohort_n else "FAIL"
        out.append(f"D={D:<8}{feat_n:>12}{cohort_n:>12}{match:>8}")
    out.append("")

    # --- 2. Per-feature stats at the headline cutoff ---
    out.append(section(f"2. Per-feature statistics at D={HEADLINE_CUTOFF}"))
    table = f"features_d{HEADLINE_CUTOFF}"
    total_n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    out.append("Numeric features:")
    out.append(
        f"{'feature':<30}{'n_nonnull':>10}{'n_null':>8}{'null_share':>11}"
        f"{'min':>12}{'max':>12}{'mean':>12}{'median':>12}{'stddev':>12}"
    )
    for feat in NUMERIC_FEATURES:
        row = con.execute(
            f"""
            SELECT
                count({feat}) AS n_nonnull,
                min({feat}), max({feat}), avg({feat}), median({feat}), stddev({feat})
            FROM {table}
            """
        ).fetchone()
        n_nonnull, fmin, fmax, fmean, fmedian, fstd = row
        n_null = total_n - n_nonnull
        null_share = n_null / total_n if total_n else 0.0

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else (f"{v}" if v is not None else "NULL")

        out.append(
            f"{feat:<30}{n_nonnull:>10}{n_null:>8}{null_share:>11.4f}"
            f"{fmt(fmin):>12}{fmt(fmax):>12}{fmt(fmean):>12}{fmt(fmedian):>12}{fmt(fstd):>12}"
        )
    out.append("")

    out.append("Categorical features (distinct value counts):")
    for feat in CATEGORICAL_FEATURES:
        out.append(f"{feat}:")
        n_null = con.execute(f"SELECT count(*) FROM {table} WHERE {feat} IS NULL").fetchone()[0]
        rows = con.execute(
            f"SELECT {feat}, count(*) AS n FROM {table} WHERE {feat} IS NOT NULL "
            f"GROUP BY {feat} ORDER BY n DESC"
        ).fetchall()
        for value, n in rows:
            out.append(f"  {value!r}: {n}")
        if n_null:
            out.append(f"  NULL: {n_null}")
        out.append("")

    # --- 3. Null share per feature across all cutoffs ---
    out.append(section("3. Null share per feature, all cutoffs"))
    header = f"{'feature':<30}" + "".join(f"D={D:<10}" for D in CUTOFFS)
    out.append(header)
    for feat in ALL_FEATURES:
        row = f"{feat:<30}"
        for D in CUTOFFS:
            t = f"features_d{D}"
            n_total = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            n_nonnull = con.execute(f"SELECT count({feat}) FROM {t}").fetchone()[0]
            share = (n_total - n_nonnull) / n_total if n_total else 0.0
            row += f"{share:<12.4f}"
        out.append(row)
    out.append("")

    # --- 4. Leakage sentinel ---
    out.append(section("4. Leakage sentinel"))
    out.append(
        "Independently re-derived from the raw source tables (not read back "
        "from the feature tables), restricted to cohort membership."
    )
    out.append("")
    out.append(f"{'cutoff':<10}{'max(vle.date)':>16}{'vle':>8}{'max(sa.date_submitted)':>24}{'sa':>8}")
    for D in CUTOFFS:
        vle_max = con.execute(
            f"""
            SELECT MAX(sv.date)
            FROM cohort_d{D} c
            JOIN student_vle sv
              ON sv.code_module = c.code_module AND sv.code_presentation = c.code_presentation
             AND sv.id_student = c.id_student
            WHERE sv.date < {D}
            """
        ).fetchone()[0]
        vle_pass = "PASS" if vle_max is None or vle_max < D else "FAIL"

        sa_max = con.execute(
            f"""
            SELECT MAX(sa.date_submitted)
            FROM cohort_d{D} c
            JOIN v_student_assessment sa ON sa.id_student = c.id_student
            JOIN v_assessments a
              ON a.id_assessment = sa.id_assessment
             AND a.code_module = c.code_module AND a.code_presentation = c.code_presentation
            WHERE a.assessment_type != 'Exam'
              AND COALESCE(sa.is_banked, 0) != 1
              AND sa.date_submitted < {D}
            """
        ).fetchone()[0]
        sa_pass = "PASS" if sa_max is None or sa_max < D else "FAIL"

        out.append(f"D={D:<8}{str(vle_max):>16}{vle_pass:>8}{str(sa_max):>24}{sa_pass:>8}")
    out.append("")

    # --- 5. Correlation with not_completed, train split, D=28 ---
    out.append(section(f"5. Correlation with not_completed, TRAIN split, D={HEADLINE_CUTOFF}"))
    out.append("Pairwise deletion of NULLs in the feature (DuckDB corr() default).")
    out.append("")
    corr_rows = []
    for feat in NUMERIC_FEATURES:
        corr = con.execute(
            f"SELECT corr({feat}, not_completed) FROM {table} WHERE split = 'train'"
        ).fetchone()[0]
        corr_rows.append((feat, corr))
    corr_rows.sort(key=lambda t: (abs(t[1]) if t[1] is not None else -1), reverse=True)
    out.append(f"{'feature':<30}{'corr':>10}")
    for feat, corr in corr_rows:
        out.append(f"{feat:<30}{(f'{corr:.4f}' if corr is not None else 'NULL'):>10}")
    out.append("")

    # --- 6. Sample export ---
    out.append(section("6. Sample export"))
    df = con.execute(
        f"SELECT * FROM {table} ORDER BY code_module, code_presentation, id_student"
    ).df()
    sample_df = df.sample(n=20, random_state=42)
    sample_df.to_csv(SAMPLE_CSV_PATH, index=False)
    out.append(f"20 rows sampled from {table} (seed 42), written to {SAMPLE_CSV_PATH}")
    out.append("")

    con.close()

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
