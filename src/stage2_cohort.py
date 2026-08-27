"""
Stage 2 cohort: target definition and study population, per PROTOCOL.md
Section 3. No feature engineering, no modelling.

Builds cohort_d14, cohort_d28, cohort_d56 in DuckDB, applying the three
Section 3 exclusions in the declared order (E3, E2, E1) to the population
joined from v_student_registration and v_student_info. All three splits
(train/validate/test) are written; nothing is filtered to a split here.

Usage:
    .venv/bin/python src/stage2_cohort.py
"""

from pathlib import Path

import duckdb

from stage2_views import create_views

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"

CUTOFFS = [14, 28, 56]

VALID_FINAL_RESULTS = {"Fail", "Withdrawn", "Pass", "Distinction"}
NOT_COMPLETED = {"Fail", "Withdrawn"}
COMPLETED = {"Pass", "Distinction"}

TRAIN_PRESENTATIONS = {"2013B", "2013J"}
VALIDATE_PRESENTATIONS = {"2014B"}
TEST_PRESENTATIONS = {"2014J"}


def cohort_ctes_sql(D: int) -> str:
    """The shared exclusion pipeline (E3, E2, E1 in that order) as a set of
    CTEs, without a trailing SELECT. Reused verbatim by both the cohort
    table build and the Stage 2 report's waterfall, so the two can never
    drift apart."""
    return f"""
    base AS (
        SELECT
            r.code_module,
            r.code_presentation,
            r.id_student,
            r.date_registration,
            r.date_unregistration,
            s.final_result
        FROM v_student_registration r
        JOIN v_student_info s
          ON r.code_module = s.code_module
         AND r.code_presentation = s.code_presentation
         AND r.id_student = s.id_student
    ),
    after_e3 AS (
        -- E3: exclude null date_registration
        SELECT * FROM base WHERE date_registration IS NOT NULL
    ),
    after_e2 AS (
        -- E2: exclude date_registration > D
        SELECT * FROM after_e3 WHERE date_registration <= {D}
    ),
    after_e1 AS (
        -- E1: exclude date_unregistration IS NOT NULL AND date_unregistration < D
        SELECT * FROM after_e2
        WHERE NOT (date_unregistration IS NOT NULL AND date_unregistration < {D})
    )
    """


def build_cohort_select_sql(D: int) -> str:
    return f"""
    WITH {cohort_ctes_sql(D)}
    SELECT
        code_module,
        code_presentation,
        id_student,
        date_registration,
        date_unregistration,
        final_result,
        CASE
            WHEN final_result IN ('Fail', 'Withdrawn') THEN 1
            WHEN final_result IN ('Pass', 'Distinction') THEN 0
        END AS not_completed,
        CASE
            WHEN code_presentation IN ('2013B', '2013J') THEN 'train'
            WHEN code_presentation = '2014B' THEN 'validate'
            WHEN code_presentation = '2014J' THEN 'test'
        END AS split
    FROM after_e1
    """


def check_final_result_coverage(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT DISTINCT final_result FROM student_info"
    ).fetchall()
    seen = {r[0] for r in rows}
    unexpected = seen - VALID_FINAL_RESULTS
    if unexpected:
        raise RuntimeError(
            f"student_info.final_result has value(s) outside the four "
            f"expected categories: {unexpected}. Target mapping (Section 3) "
            "does not cover this; refusing to guess."
        )


def check_presentation_coverage(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        "SELECT DISTINCT code_presentation FROM courses"
    ).fetchall()
    seen = {r[0] for r in rows}
    expected = TRAIN_PRESENTATIONS | VALIDATE_PRESENTATIONS | TEST_PRESENTATIONS
    unexpected = seen - expected
    if unexpected:
        raise RuntimeError(
            f"courses.code_presentation has value(s) outside the four "
            f"presentations the Section 4 split covers: {unexpected}."
        )


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        create_views(con)
        check_final_result_coverage(con)
        check_presentation_coverage(con)

        for D in CUTOFFS:
            table = f"cohort_d{D}"
            con.execute(f"CREATE OR REPLACE TABLE {table} AS {build_cohort_select_sql(D)}")

            null_target = con.execute(
                f"SELECT count(*) FROM {table} WHERE not_completed IS NULL"
            ).fetchone()[0]
            if null_target:
                raise RuntimeError(
                    f"{table}: {null_target} row(s) have a NULL not_completed "
                    "after mapping — final_result did not fall into any of "
                    "the four expected categories for these rows."
                )
            null_split = con.execute(
                f"SELECT count(*) FROM {table} WHERE split IS NULL"
            ).fetchone()[0]
            if null_split:
                raise RuntimeError(
                    f"{table}: {null_split} row(s) have a NULL split — "
                    "code_presentation did not map to train/validate/test."
                )

            count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count} rows")
    finally:
        con.close()


if __name__ == "__main__":
    main()
