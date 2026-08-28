"""
Stage 3 feature construction, per PROTOCOL.md Section 5 (leakage boundary)
and Section 6 (feature specification), plus Amendments A1/A3.

Builds features_d14, features_d28, features_d56 in DuckDB. All computation
happens in SQL. No modelling, no imputation, no scaling, no encoding, no
train/test filtering.

Usage:
    .venv/bin/python src/stage3_features.py
"""

from pathlib import Path

import duckdb

from stage2_cohort import CUTOFFS
from stage2_views import create_views

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"


def build_features_sql(D: int) -> str:
    D_minus_1 = D - 1
    D_minus_7 = D - 7
    D_half = D // 2

    return f"""
    WITH
    cohort AS (
        SELECT * FROM cohort_d{D}
    ),

    -- L1: student_vle rows strictly before D.
    vle_pre AS (
        SELECT code_module, code_presentation, id_student, id_site, date, sum_click
        FROM student_vle
        WHERE date < {D}
    ),

    -- Group B aggregates.
    b_agg AS (
        SELECT
            code_module, code_presentation, id_student,
            SUM(sum_click) AS total_clicks,
            COUNT(DISTINCT date) AS active_days,
            SUM(CASE WHEN date >= {D_minus_7} AND date < {D} THEN sum_click ELSE 0 END) AS clicks_last_7d,
            MAX(date) AS max_date
        FROM vle_pre
        GROUP BY 1, 2, 3
    ),

    -- Group C: daily click slope (Amendment A3), zero-filled over 0..D-1.
    days_series AS (
        SELECT day FROM range(0, {D}) AS t(day)
    ),
    daily_clicks AS (
        SELECT code_module, code_presentation, id_student, date, SUM(sum_click) AS clicks
        FROM vle_pre
        GROUP BY 1, 2, 3, 4
    ),
    slope_agg AS (
        SELECT
            c.code_module, c.code_presentation, c.id_student,
            regr_slope(COALESCE(dc.clicks, 0), ds.day) AS click_slope_daily
        FROM cohort c
        CROSS JOIN days_series ds
        LEFT JOIN daily_clicks dc
          ON dc.code_module = c.code_module
         AND dc.code_presentation = c.code_presentation
         AND dc.id_student = c.id_student
         AND dc.date = ds.day
        GROUP BY 1, 2, 3
    ),

    -- Group C: second-half / first-half click ratio.
    first_half_agg AS (
        SELECT code_module, code_presentation, id_student, SUM(sum_click) AS first_half_clicks
        FROM vle_pre
        WHERE date >= 0 AND date < {D_half}
        GROUP BY 1, 2, 3
    ),
    second_half_agg AS (
        SELECT code_module, code_presentation, id_student, SUM(sum_click) AS second_half_clicks
        FROM vle_pre
        WHERE date >= {D_half} AND date < {D}
        GROUP BY 1, 2, 3
    ),

    -- Group C: longest gap between consecutive distinct active dates.
    active_dates AS (
        SELECT DISTINCT code_module, code_presentation, id_student, date
        FROM vle_pre
    ),
    gaps AS (
        SELECT
            code_module, code_presentation, id_student,
            date - LAG(date) OVER (
                PARTITION BY code_module, code_presentation, id_student
                ORDER BY date
            ) AS gap
        FROM active_dates
    ),
    longest_gap_agg AS (
        SELECT code_module, code_presentation, id_student, MAX(gap) AS longest_activity_gap
        FROM gaps
        WHERE gap IS NOT NULL
        GROUP BY 1, 2, 3
    ),

    -- Group D: engagement breadth, joining pre-D activity to vle metadata.
    d_join AS (
        SELECT vp.code_module, vp.code_presentation, vp.id_student,
               vp.id_site, vp.sum_click, v.activity_type
        FROM vle_pre vp
        LEFT JOIN vle v ON vp.id_site = v.id_site
    ),
    d_agg AS (
        SELECT
            code_module, code_presentation, id_student,
            COUNT(DISTINCT activity_type) AS distinct_activity_types,
            COUNT(DISTINCT id_site) AS distinct_materials,
            SUM(CASE WHEN activity_type IN ('quiz', 'externalquiz') THEN sum_click ELSE 0 END) AS assessment_clicks
        FROM d_join
        GROUP BY 1, 2, 3
    ),

    -- Group E: assessment behaviour.
    -- L3: exclude Exam. A1: exclude banked submissions. L2: filter on
    -- date_submitted, never on the due date, except for n_due_not_submitted
    -- which legitimately counts against the due date.
    sa_pre AS (
        SELECT
            sa.id_assessment, sa.id_student, sa.date_submitted, sa.score, sa.is_banked,
            a.code_module, a.code_presentation, a.assessment_type, a.date AS due_date
        FROM v_student_assessment sa
        LEFT JOIN v_assessments a ON sa.id_assessment = a.id_assessment
    ),
    sa_filtered AS (
        SELECT * FROM sa_pre
        WHERE assessment_type != 'Exam'
          AND COALESCE(is_banked, 0) != 1
    ),
    submitted AS (
        SELECT code_module, code_presentation, id_student, id_assessment, date_submitted, score, due_date
        FROM sa_filtered
        WHERE date_submitted < {D}
    ),
    e_agg AS (
        SELECT
            code_module, code_presentation, id_student,
            COUNT(*) AS n_assessments_submitted,
            AVG(score) AS mean_score_submitted,
            MIN(score) AS min_score_submitted,
            AVG(date_submitted - due_date) AS mean_submission_lateness
        FROM submitted
        GROUP BY 1, 2, 3
    ),
    due_assessments AS (
        SELECT code_module, code_presentation, id_assessment
        FROM v_assessments
        WHERE assessment_type != 'Exam' AND date < {D}
    ),
    due_not_submitted AS (
        SELECT c.code_module, c.code_presentation, c.id_student, COUNT(*) AS n_due_not_submitted
        FROM cohort c
        LEFT JOIN due_assessments da
          ON da.code_module = c.code_module AND da.code_presentation = c.code_presentation
        LEFT JOIN submitted s
          ON s.code_module = c.code_module AND s.code_presentation = c.code_presentation
         AND s.id_student = c.id_student AND s.id_assessment = da.id_assessment
        WHERE da.id_assessment IS NOT NULL AND s.id_assessment IS NULL
        GROUP BY 1, 2, 3
    ),

    -- Groups A-E assembled, one row per cohort row. Every join from cohort
    -- to activity/assessment aggregates is a LEFT JOIN so zero-activity
    -- students are retained per Section 3.
    assembled AS (
        SELECT
            c.code_module, c.code_presentation, c.id_student,
            c.not_completed, c.split,

            -- Group A (L7: known-at-registration studentInfo fields)
            i.gender, i.region, i.highest_education, i.imd_band, i.age_band,
            i.disability, i.num_of_prev_attempts, i.studied_credits,

            -- Group B
            COALESCE(b.total_clicks, 0) AS total_clicks,
            COALESCE(b.active_days, 0) AS active_days,
            COALESCE(b.clicks_last_7d, 0) AS clicks_last_7d,
            CASE WHEN COALESCE(b.active_days, 0) = 0 THEN NULL
                 ELSE COALESCE(b.total_clicks, 0)::DOUBLE / b.active_days
            END AS mean_clicks_per_active_day,
            (0 - c.date_registration) AS days_registered_before_start,

            -- Group C
            CASE WHEN b.max_date IS NULL THEN NULL ELSE {D_minus_1} - b.max_date END AS days_since_last_activity,
            sl.click_slope_daily,
            CASE WHEN COALESCE(fh.first_half_clicks, 0) = 0 THEN NULL
                 ELSE COALESCE(sh.second_half_clicks, 0)::DOUBLE / fh.first_half_clicks
            END AS second_half_click_ratio,
            lg.longest_activity_gap,

            -- Group D
            COALESCE(d.distinct_activity_types, 0) AS distinct_activity_types,
            COALESCE(d.distinct_materials, 0) AS distinct_materials,
            CASE WHEN COALESCE(b.total_clicks, 0) = 0 THEN 0
                 ELSE COALESCE(d.assessment_clicks, 0)::DOUBLE / b.total_clicks
            END AS assessment_material_share,

            -- Group E
            COALESCE(e.n_assessments_submitted, 0) AS n_assessments_submitted,
            e.mean_score_submitted,
            e.min_score_submitted,
            e.mean_submission_lateness,
            COALESCE(dns.n_due_not_submitted, 0) AS n_due_not_submitted

        FROM cohort c
        LEFT JOIN v_student_info i
          ON i.code_module = c.code_module AND i.code_presentation = c.code_presentation
         AND i.id_student = c.id_student
        LEFT JOIN b_agg b
          ON b.code_module = c.code_module AND b.code_presentation = c.code_presentation
         AND b.id_student = c.id_student
        LEFT JOIN slope_agg sl
          ON sl.code_module = c.code_module AND sl.code_presentation = c.code_presentation
         AND sl.id_student = c.id_student
        LEFT JOIN first_half_agg fh
          ON fh.code_module = c.code_module AND fh.code_presentation = c.code_presentation
         AND fh.id_student = c.id_student
        LEFT JOIN second_half_agg sh
          ON sh.code_module = c.code_module AND sh.code_presentation = c.code_presentation
         AND sh.id_student = c.id_student
        LEFT JOIN longest_gap_agg lg
          ON lg.code_module = c.code_module AND lg.code_presentation = c.code_presentation
         AND lg.id_student = c.id_student
        LEFT JOIN d_agg d
          ON d.code_module = c.code_module AND d.code_presentation = c.code_presentation
         AND d.id_student = c.id_student
        LEFT JOIN e_agg e
          ON e.code_module = c.code_module AND e.code_presentation = c.code_presentation
         AND e.id_student = c.id_student
        LEFT JOIN due_not_submitted dns
          ON dns.code_module = c.code_module AND dns.code_presentation = c.code_presentation
         AND dns.id_student = c.id_student
    ),

    -- Group F: cohort-relative, computed only over this cutoff's cohort (L6).
    -- score_percentile is ranked only among students with a defined mean
    -- score; students without one get NULL, not a diluted percentile.
    -- Per Amendment A6, the third Group F feature is code_module (already a
    -- key column), not module_presentation: module_presentation is
    -- degenerate under the Section 4 split (no value recurs across train,
    -- validate and test), so it is not produced as a separate column here.
    score_percentile_agg AS (
        SELECT
            code_module, code_presentation, id_student,
            PERCENT_RANK() OVER (
                PARTITION BY code_module, code_presentation
                ORDER BY mean_score_submitted
            ) AS score_percentile
        FROM assembled
        WHERE mean_score_submitted IS NOT NULL
    )

    SELECT
        a.code_module, a.code_presentation, a.id_student,
        a.not_completed, a.split,

        a.gender, a.region, a.highest_education, a.imd_band, a.age_band,
        a.disability, a.num_of_prev_attempts, a.studied_credits,

        a.total_clicks, a.active_days, a.clicks_last_7d,
        a.mean_clicks_per_active_day, a.days_registered_before_start,

        a.days_since_last_activity, a.click_slope_daily,
        a.second_half_click_ratio, a.longest_activity_gap,

        a.distinct_activity_types, a.distinct_materials, a.assessment_material_share,

        a.n_assessments_submitted, a.mean_score_submitted, a.min_score_submitted,
        a.mean_submission_lateness, a.n_due_not_submitted,

        PERCENT_RANK() OVER (
            PARTITION BY a.code_module, a.code_presentation
            ORDER BY a.total_clicks
        ) AS clicks_percentile,
        sp.score_percentile

    FROM assembled a
    LEFT JOIN score_percentile_agg sp
      ON sp.code_module = a.code_module AND sp.code_presentation = a.code_presentation
     AND sp.id_student = a.id_student
    """


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        create_views(con)

        for D in CUTOFFS:
            table = f"features_d{D}"
            cohort_table = f"cohort_d{D}"
            sql = build_features_sql(D)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS {sql}")

            feat_count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            cohort_count = con.execute(f"SELECT count(*) FROM {cohort_table}").fetchone()[0]
            if feat_count != cohort_count:
                raise RuntimeError(
                    f"{table} has {feat_count} rows but {cohort_table} has "
                    f"{cohort_count}. A join dropped or duplicated rows."
                )
            print(f"{table}: {feat_count} rows (matches {cohort_table})")
    finally:
        con.close()


if __name__ == "__main__":
    main()
