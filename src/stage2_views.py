"""
Stage 2 shared setup: typed views over the Stage 1 tables.

The Stage 1 CSVs use a literal '?' as a missing-value placeholder, which
forced several numeric columns to load as VARCHAR (see
reports/stage1_validation.txt, sections 4 and 5). This module creates
read-only views that cast those columns properly, converting '?' to NULL,
and normalises the one malformed imd_band category. The underlying Stage 1
tables are left untouched.

Both stage2_cohort.py and stage2_report.py call create_views() before
querying, so the views always exist and are always up to date (CREATE OR
REPLACE VIEW is idempotent).
"""

import duckdb

VIEW_DDL = {
    "v_student_registration": """
        CREATE OR REPLACE VIEW v_student_registration AS
        SELECT
            code_module,
            code_presentation,
            id_student,
            CAST(NULLIF(date_registration, '?') AS INTEGER) AS date_registration,
            CAST(NULLIF(date_unregistration, '?') AS INTEGER) AS date_unregistration
        FROM student_registration
    """,
    "v_student_assessment": """
        CREATE OR REPLACE VIEW v_student_assessment AS
        SELECT
            id_assessment,
            id_student,
            date_submitted,
            is_banked,
            CAST(NULLIF(score, '?') AS DOUBLE) AS score
        FROM student_assessment
    """,
    "v_assessments": """
        CREATE OR REPLACE VIEW v_assessments AS
        SELECT
            code_module,
            code_presentation,
            id_assessment,
            assessment_type,
            CAST(NULLIF(date, '?') AS INTEGER) AS date,
            weight
        FROM assessments
    """,
    "v_student_info": """
        CREATE OR REPLACE VIEW v_student_info AS
        SELECT
            code_module,
            code_presentation,
            id_student,
            gender,
            region,
            highest_education,
            CASE
                WHEN imd_band = '?' THEN NULL
                WHEN imd_band = '10-20' THEN '10-20%'
                ELSE imd_band
            END AS imd_band,
            age_band,
            num_of_prev_attempts,
            studied_credits,
            disability,
            final_result
        FROM student_info
    """,
}


def normalisation_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Row counts of each value being normalised, computed against the raw
    Stage 1 tables before the views are (re)created."""
    return {
        "student_registration.date_registration: '?' -> NULL": con.execute(
            "SELECT count(*) FROM student_registration WHERE date_registration = '?'"
        ).fetchone()[0],
        "student_registration.date_unregistration: '?' -> NULL": con.execute(
            "SELECT count(*) FROM student_registration WHERE date_unregistration = '?'"
        ).fetchone()[0],
        "student_assessment.score: '?' -> NULL": con.execute(
            "SELECT count(*) FROM student_assessment WHERE score = '?'"
        ).fetchone()[0],
        "assessments.date: '?' -> NULL": con.execute(
            "SELECT count(*) FROM assessments WHERE date = '?'"
        ).fetchone()[0],
        "student_info.imd_band: '?' -> NULL": con.execute(
            "SELECT count(*) FROM student_info WHERE imd_band = '?'"
        ).fetchone()[0],
        "student_info.imd_band: '10-20' -> '10-20%'": con.execute(
            "SELECT count(*) FROM student_info WHERE imd_band = '10-20'"
        ).fetchone()[0],
    }


def create_views(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts = normalisation_counts(con)
    for ddl in VIEW_DDL.values():
        con.execute(ddl)
    return counts
