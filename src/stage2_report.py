"""
Stage 2 report: target definition and study population diagnostics, per
PROTOCOL.md Section 3, Section 4, Section 10 D3, and Amendments A1/A2.

Counts and diagnostics only. No feature engineering, no modelling.

Usage:
    .venv/bin/python src/stage2_report.py
"""

from pathlib import Path

import duckdb

from stage2_cohort import CUTOFFS, cohort_ctes_sql
from stage2_views import create_views

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
REPORT_PATH = ROOT / "reports" / "stage2_cohort.txt"

PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
SPLIT_OF = {
    "2013B": "train",
    "2013J": "train",
    "2014B": "validate",
    "2014J": "test",
}


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=False)
    out: list[str] = []

    out.append("Stage 2 Cohort Report")
    out.append(f"Database: {DB_PATH}")

    # --- Type normalisation (must run before anything queries the views) ---
    norm_counts = create_views(con)
    out.append(section("0a. Type normalisation"))
    out.append("Rows touched by each cast/normalisation (against Stage 1 raw tables):")
    for label, n in norm_counts.items():
        out.append(f"  {label}: {n}")
    out.append("")
    out.append(
        "Views created: v_student_registration, v_student_assessment, "
        "v_assessments, v_student_info. Stage 1 tables left untouched."
    )
    out.append("")

    # --- Key uniqueness ---
    out.append(section("0b. student_info key uniqueness"))
    total_rows = con.execute("SELECT count(*) FROM student_info").fetchone()[0]
    distinct_keys = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT code_module, code_presentation, "
        "id_student FROM student_info)"
    ).fetchone()[0]
    out.append(f"total rows: {total_rows}")
    out.append(f"distinct (code_module, code_presentation, id_student): {distinct_keys}")
    if total_rows == distinct_keys:
        out.append("Key is unique.")
    else:
        out.append(
            f"KEY IS NOT UNIQUE: {total_rows - distinct_keys} duplicate row(s) "
            "on this key."
        )
    out.append("")

    # --- Diagnostic: final_result vs date_unregistration NULL, BEFORE exclusions ---
    out.append(section("0c. Diagnostic: final_result x date_unregistration NULL (pre-exclusion)"))
    out.append(
        "Cross-tabulation over the full joined population "
        "(v_student_registration x v_student_info), before E1/E2/E3."
    )
    out.append("")
    crosstab = con.execute(
        """
        SELECT
            s.final_result,
            CASE WHEN r.date_unregistration IS NULL THEN 'NULL' ELSE 'NOT NULL' END AS unreg_state,
            count(*) AS n
        FROM v_student_registration r
        JOIN v_student_info s
          ON r.code_module = s.code_module
         AND r.code_presentation = s.code_presentation
         AND r.id_student = s.id_student
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchall()
    out.append(f"{'final_result':<15}{'date_unregistration':<22}{'count':>8}")
    for fr, state, n in crosstab:
        out.append(f"{fr:<15}{state:<22}{n:>8}")
    out.append("")

    withdrawn_null_unreg = con.execute(
        """
        SELECT count(*)
        FROM v_student_registration r
        JOIN v_student_info s
          ON r.code_module = s.code_module
         AND r.code_presentation = s.code_presentation
         AND r.id_student = s.id_student
        WHERE s.final_result = 'Withdrawn' AND r.date_unregistration IS NULL
        """
    ).fetchone()[0]
    out.append(
        f"final_result = 'Withdrawn' but NULL date_unregistration: {withdrawn_null_unreg}"
    )

    nonwithdrawn_nonnull_unreg = con.execute(
        """
        SELECT count(*)
        FROM v_student_registration r
        JOIN v_student_info s
          ON r.code_module = s.code_module
         AND r.code_presentation = s.code_presentation
         AND r.id_student = s.id_student
        WHERE s.final_result != 'Withdrawn' AND r.date_unregistration IS NOT NULL
        """
    ).fetchone()[0]
    out.append(
        "non-NULL date_unregistration but final_result != 'Withdrawn': "
        f"{nonwithdrawn_nonnull_unreg}"
    )
    out.append("")
    out.append("Reported only. Not resolved, not corrected.")
    out.append("")

    # --- 1. Waterfall per cutoff, by presentation ---
    out.append(section("1. Exclusion waterfall (E3, E2, E1 in order), per cutoff"))
    for D in CUTOFFS:
        out.append(f"--- D = {D} ---")
        stages_sql = f"""
        WITH {cohort_ctes_sql(D)}
        SELECT 'start' AS stage, code_presentation, count(*) AS n FROM base GROUP BY code_presentation
        UNION ALL
        SELECT 'after_e3', code_presentation, count(*) FROM after_e3 GROUP BY code_presentation
        UNION ALL
        SELECT 'after_e2', code_presentation, count(*) FROM after_e2 GROUP BY code_presentation
        UNION ALL
        SELECT 'after_e1', code_presentation, count(*) FROM after_e1 GROUP BY code_presentation
        """
        rows = con.execute(stages_sql).fetchall()
        by_stage_pres: dict[tuple, int] = {(stage, pres): n for stage, pres, n in rows}

        header = (
            f"{'presentation':<14}{'start':>8}{'-E3':>8}{'-E2':>8}{'-E1':>8}{'remaining':>11}"
        )
        out.append(header)
        totals = {"start": 0, "after_e3": 0, "after_e2": 0, "after_e1": 0}
        for pres in PRESENTATIONS:
            start = by_stage_pres.get(("start", pres), 0)
            after_e3 = by_stage_pres.get(("after_e3", pres), 0)
            after_e2 = by_stage_pres.get(("after_e2", pres), 0)
            after_e1 = by_stage_pres.get(("after_e1", pres), 0)
            removed_e3 = start - after_e3
            removed_e2 = after_e3 - after_e2
            removed_e1 = after_e2 - after_e1
            totals["start"] += start
            totals["after_e3"] += after_e3
            totals["after_e2"] += after_e2
            totals["after_e1"] += after_e1
            out.append(
                f"{pres:<14}{start:>8}{removed_e3:>8}{removed_e2:>8}{removed_e1:>8}{after_e1:>11}"
            )
        removed_e3_t = totals["start"] - totals["after_e3"]
        removed_e2_t = totals["after_e3"] - totals["after_e2"]
        removed_e1_t = totals["after_e2"] - totals["after_e1"]
        out.append(
            f"{'TOTAL':<14}{totals['start']:>8}{removed_e3_t:>8}{removed_e2_t:>8}"
            f"{removed_e1_t:>8}{totals['after_e1']:>11}"
        )
        out.append("")

    # --- 2. Base rate per cutoff, per presentation, per split, and per module-presentation ---
    out.append(section("2. Base rate (mean not_completed)"))
    for D in CUTOFFS:
        table = f"cohort_d{D}"
        out.append(f"--- D = {D}, by presentation / split ---")
        out.append(f"{'presentation':<14}{'split':<10}{'n':>8}{'base_rate':>12}")
        rows = con.execute(
            f"""
            SELECT code_presentation, split, count(*) AS n, avg(not_completed) AS base_rate
            FROM {table}
            GROUP BY code_presentation, split
            ORDER BY code_presentation
            """
        ).fetchall()
        for pres, split, n, rate in rows:
            out.append(f"{pres:<14}{split:<10}{n:>8}{rate:>12.4f}")
        out.append("")

        out.append(f"--- D = {D}, by split (aggregated) ---")
        out.append(f"{'split':<10}{'n':>8}{'base_rate':>12}")
        rows = con.execute(
            f"""
            SELECT split, count(*) AS n, avg(not_completed) AS base_rate
            FROM {table}
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()
        for split, n, rate in rows:
            out.append(f"{split:<10}{n:>8}{rate:>12.4f}")
        out.append("")

        out.append(f"--- D = {D}, by module-presentation ---")
        out.append(f"{'module':<8}{'presentation':<14}{'n':>8}{'base_rate':>12}")
        rows = con.execute(
            f"""
            SELECT code_module, code_presentation, count(*) AS n, avg(not_completed) AS base_rate
            FROM {table}
            GROUP BY code_module, code_presentation
            ORDER BY code_module, code_presentation
            """
        ).fetchall()
        for mod, pres, n, rate in rows:
            out.append(f"{mod:<8}{pres:<14}{n:>8}{rate:>12.4f}")
        out.append("")

    # --- 3. Section 10 D3: share removed by E1, per presentation, per cutoff ---
    out.append(section("3. D3: share of rows removed by E1, per presentation, per cutoff"))
    out.append(f"{'presentation':<14}" + "".join(f"D={D:<10}" for D in CUTOFFS))
    e1_shares: dict[tuple, float] = {}
    for D in CUTOFFS:
        stages_sql = f"""
        WITH {cohort_ctes_sql(D)}
        SELECT code_presentation, count(*) AS n FROM after_e2 GROUP BY code_presentation
        """
        after_e2_rows = dict(con.execute(stages_sql).fetchall())
        stages_sql2 = f"""
        WITH {cohort_ctes_sql(D)}
        SELECT code_presentation, count(*) AS n FROM after_e1 GROUP BY code_presentation
        """
        after_e1_rows = dict(con.execute(stages_sql2).fetchall())
        for pres in PRESENTATIONS:
            denom = after_e2_rows.get(pres, 0)
            after_e1 = after_e1_rows.get(pres, 0)
            share = (denom - after_e1) / denom if denom else 0.0
            e1_shares[(pres, D)] = share
    for pres in PRESENTATIONS:
        row = f"{pres:<14}"
        for D in CUTOFFS:
            row += f"{e1_shares[(pres, D)]:<12.4f}"
        out.append(row)
    out.append("")
    out.append(
        "Share = (rows entering E1 minus rows surviving E1) / rows entering E1, "
        "i.e. share of the E2-surviving population removed by E1."
    )
    out.append("")

    # --- 4. Amendment A2: student overlap between 2014J test and 2013 train, at D=28 ---
    out.append(section("4. A2: student overlap, 2014J test vs 2013B/2013J train, at D=28"))
    overlap = con.execute(
        """
        SELECT count(DISTINCT t.id_student)
        FROM cohort_d28 t
        WHERE t.code_presentation = '2014J'
          AND t.id_student IN (
              SELECT id_student FROM cohort_d28 WHERE code_presentation IN ('2013B', '2013J')
          )
        """
    ).fetchone()[0]
    test_distinct = con.execute(
        "SELECT count(DISTINCT id_student) FROM cohort_d28 WHERE code_presentation = '2014J'"
    ).fetchone()[0]
    share = overlap / test_distinct if test_distinct else 0.0
    out.append(f"distinct id_student in 2014J test split (cohort_d28): {test_distinct}")
    out.append(f"of those, also appearing in 2013B or 2013J train split: {overlap}")
    out.append(f"share of test split: {share:.4f}")
    out.append("")

    # --- 5. Zero-activity count at D=28, count only ---
    out.append(section("5. Zero-activity count at D=28 (count only, no feature built)"))
    zero_activity = con.execute(
        """
        SELECT
            SUM(CASE WHEN v.id_student IS NULL THEN 1 ELSE 0 END) AS zero_activity_rows,
            count(*) AS total_rows,
            avg(CASE WHEN v.id_student IS NULL THEN c.not_completed END) AS base_rate_zero_activity,
            avg(CASE WHEN v.id_student IS NOT NULL THEN c.not_completed END) AS base_rate_rest
        FROM cohort_d28 c
        LEFT JOIN (
            SELECT DISTINCT code_module, code_presentation, id_student
            FROM student_vle
            WHERE date < 28
        ) v
          ON c.code_module = v.code_module
         AND c.code_presentation = v.code_presentation
         AND c.id_student = v.id_student
        """
    ).fetchone()
    zero_rows, total_rows_d28, base_rate_zero, base_rate_rest = zero_activity
    share_zero = zero_rows / total_rows_d28 if total_rows_d28 else 0.0
    out.append(f"cohort_d28 rows: {total_rows_d28}")
    out.append(f"rows with no student_vle activity (date < 28): {zero_rows}")
    out.append(f"share: {share_zero:.4f}")
    out.append(f"base rate (not_completed) among zero-activity rows: {base_rate_zero:.4f}")
    out.append(f"base rate (not_completed) among the rest: {base_rate_rest:.4f}")
    out.append("")

    # --- 6. Amendment A1: banked assessment rows ---
    out.append(section("6. A1: banked (is_banked = 1) student_assessment rows"))
    banked_rows = con.execute(
        "SELECT count(*) FROM student_assessment WHERE is_banked = 1"
    ).fetchone()[0]
    banked_students = con.execute(
        "SELECT count(DISTINCT id_student) FROM student_assessment WHERE is_banked = 1"
    ).fetchone()[0]
    out.append(f"is_banked = 1 rows: {banked_rows}")
    out.append(f"distinct students affected: {banked_students}")
    out.append("")

    # --- 7. Arm 2 readiness: TMA presence per module-presentation ---
    out.append(section("7. Arm 2 readiness: earliest TMA per module-presentation"))
    tma_rows = con.execute(
        """
        SELECT code_module, code_presentation, min(date) AS earliest_tma_date, count(*) AS n_tma
        FROM v_assessments
        WHERE assessment_type = 'TMA'
        GROUP BY code_module, code_presentation
        ORDER BY code_module, code_presentation
        """
    ).fetchall()
    tma_present = {(m, p) for m, p, _, _ in tma_rows}
    out.append(f"{'module':<8}{'presentation':<14}{'earliest_TMA_date':>18}{'n_TMA':>8}")
    for mod, pres, earliest, n in tma_rows:
        out.append(f"{mod:<8}{pres:<14}{str(earliest):>18}{n:>8}")
    out.append("")

    all_module_presentations = con.execute(
        "SELECT DISTINCT code_module, code_presentation FROM courses ORDER BY 1, 2"
    ).fetchall()
    missing_tma = [
        (m, p) for m, p in all_module_presentations if (m, p) not in tma_present
    ]
    if missing_tma:
        out.append("module-presentation(s) with NO TMA assessment:")
        for m, p in missing_tma:
            out.append(f"  {m} {p}")
    else:
        out.append("Every module-presentation in courses has at least one TMA.")
    out.append("")

    con.close()

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
