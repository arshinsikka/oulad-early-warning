"""
Stage 1 validation: query the DuckDB database and report table structure,
distributions and referential integrity. Report only — no fixing, no
filtering, no dropping.

Usage:
    .venv/bin/python src/stage1_validate.py
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "oulad.duckdb"
REPORT_PATH = ROOT / "reports" / "stage1_validation.txt"

TABLES = [
    "student_info",
    "student_registration",
    "student_assessment",
    "assessments",
    "courses",
    "vle",
    "student_vle",
]


def section(title: str) -> str:
    bar = "=" * len(title)
    return f"\n{title}\n{bar}\n"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    out: list[str] = []

    out.append("Stage 1 Validation Report")
    out.append(f"Database: {DB_PATH}")

    # 1. Row count, column count, column list per table.
    out.append(section("1. Table shapes"))
    for table in TABLES:
        row_count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_names = [c[0] for c in cols]
        out.append(f"{table}")
        out.append(f"  rows: {row_count}")
        out.append(f"  columns: {len(col_names)}")
        out.append(f"  column names: {col_names}")
        out.append("")

    # 2. courses module/presentation grid.
    out.append(section("2. courses: module x presentation grid"))
    modules = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT code_module FROM courses ORDER BY code_module"
        ).fetchall()
    ]
    presentations = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT code_presentation FROM courses ORDER BY code_presentation"
        ).fetchall()
    ]
    pairs = set(
        con.execute(
            "SELECT code_module, code_presentation FROM courses"
        ).fetchall()
    )
    out.append(f"distinct code_module values ({len(modules)}): {modules}")
    out.append(
        f"distinct code_presentation values ({len(presentations)}): {presentations}"
    )
    out.append(f"total module-presentation combinations present: {len(pairs)}")
    out.append("")

    col_width = max(len(p) for p in presentations) + 2
    header = "module".ljust(10) + "".join(p.ljust(col_width) for p in presentations)
    out.append(header)
    for m in modules:
        row = m.ljust(10)
        for p in presentations:
            mark = "X" if (m, p) in pairs else "."
            row += mark.ljust(col_width)
        out.append(row)
    out.append("")

    # 3. student_info distributions.
    out.append(section("3. student_info distributions"))
    for col in [
        "final_result",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]:
        out.append(f"{col}:")
        rows = con.execute(
            f"SELECT {col}, count(*) AS n FROM student_info "
            f"GROUP BY {col} ORDER BY n DESC"
        ).fetchall()
        for value, n in rows:
            out.append(f"  {value!r}: {n}")
        out.append("")

    distinct_students = con.execute(
        "SELECT count(DISTINCT id_student) FROM student_info"
    ).fetchone()[0]
    total_rows = con.execute("SELECT count(*) FROM student_info").fetchone()[0]
    out.append(f"distinct id_student: {distinct_students}")
    out.append(f"total rows: {total_rows}")
    out.append(
        f"rows per student (rows / distinct students): "
        f"{total_rows / distinct_students:.4f}"
    )
    out.append("")

    # 4. student_registration.
    out.append(section("4. student_registration"))
    row_count = con.execute("SELECT count(*) FROM student_registration").fetchone()[0]
    # Both date columns are stored as VARCHAR in the source: a literal '?'
    # is used as a placeholder in addition to true SQL NULL, which is why
    # neither column inferred as numeric. Both are reported, and TRY_CAST
    # treats both as missing for the min/max calculation.
    null_reg = con.execute(
        "SELECT count(*) FROM student_registration WHERE date_registration IS NULL"
    ).fetchone()[0]
    non_numeric_reg = con.execute(
        "SELECT count(*) FROM student_registration "
        "WHERE date_registration IS NOT NULL "
        "AND TRY_CAST(date_registration AS DOUBLE) IS NULL"
    ).fetchone()[0]
    null_unreg = con.execute(
        "SELECT count(*) FROM student_registration WHERE date_unregistration IS NULL"
    ).fetchone()[0]
    non_numeric_unreg = con.execute(
        "SELECT count(*) FROM student_registration "
        "WHERE date_unregistration IS NOT NULL "
        "AND TRY_CAST(date_unregistration AS DOUBLE) IS NULL"
    ).fetchone()[0]
    reg_min, reg_max = con.execute(
        "SELECT min(TRY_CAST(date_registration AS DOUBLE)), "
        "max(TRY_CAST(date_registration AS DOUBLE)) FROM student_registration"
    ).fetchone()
    unreg_min, unreg_max = con.execute(
        "SELECT min(TRY_CAST(date_unregistration AS DOUBLE)), "
        "max(TRY_CAST(date_unregistration AS DOUBLE)) FROM student_registration"
    ).fetchone()
    out.append(f"rows: {row_count}")
    out.append("date_registration and date_unregistration source type: VARCHAR")
    out.append(f"null date_registration (true SQL NULL): {null_reg}")
    out.append(
        f"non-numeric date_registration placeholder (e.g. '?'): {non_numeric_reg}"
    )
    out.append(f"null date_unregistration (true SQL NULL): {null_unreg}")
    out.append(
        f"non-numeric date_unregistration placeholder (e.g. '?'): {non_numeric_unreg}"
    )
    out.append(
        "date ranges below computed via TRY_CAST(... AS DOUBLE), which "
        "treats both SQL NULL and non-numeric placeholders as missing:"
    )
    out.append(f"date_registration range: [{reg_min}, {reg_max}]")
    out.append(f"date_unregistration range: [{unreg_min}, {unreg_max}]")
    out.append("")

    # 5. student_assessment + assessments.
    out.append(section("5. student_assessment and assessments"))
    sa_count = con.execute("SELECT count(*) FROM student_assessment").fetchone()[0]
    sa_null_score = con.execute(
        "SELECT count(*) FROM student_assessment WHERE score IS NULL"
    ).fetchone()[0]
    # score is stored as VARCHAR in the source: a literal '?' is used as a
    # missing-value placeholder for some rows, which is why the column did
    # not infer as numeric. Reported here rather than silently cast away.
    sa_non_numeric_score = con.execute(
        "SELECT count(*) FROM student_assessment "
        "WHERE score IS NOT NULL AND TRY_CAST(score AS DOUBLE) IS NULL"
    ).fetchone()[0]
    sa_min, sa_max, sa_mean, sa_median = con.execute(
        "SELECT min(TRY_CAST(score AS DOUBLE)), max(TRY_CAST(score AS DOUBLE)), "
        "avg(TRY_CAST(score AS DOUBLE)), median(TRY_CAST(score AS DOUBLE)) "
        "FROM student_assessment"
    ).fetchone()
    out.append(f"student_assessment rows: {sa_count}")
    out.append(f"student_assessment score column source type: VARCHAR")
    out.append(f"student_assessment null score (true SQL NULL): {sa_null_score}")
    out.append(
        "student_assessment non-numeric score placeholder (e.g. '?'), "
        f"not SQL NULL but not parseable as a number: {sa_non_numeric_score}"
    )
    out.append(
        "score min/max/mean/median below computed via TRY_CAST(score AS "
        "DOUBLE), which treats both of the above as missing:"
    )
    out.append(
        f"score min={sa_min}, max={sa_max}, mean={sa_mean:.4f}, median={sa_median}"
    )
    out.append("")

    out.append("assessments.assessment_type distinct values and counts:")
    for value, n in con.execute(
        "SELECT assessment_type, count(*) AS n FROM assessments "
        "GROUP BY assessment_type ORDER BY n DESC"
    ).fetchall():
        out.append(f"  {value!r}: {n}")
    out.append("")

    # assessments.date is also stored as VARCHAR due to a '?' placeholder
    # in addition to true SQL NULL.
    a_min, a_max = con.execute(
        "SELECT min(TRY_CAST(date AS DOUBLE)), max(TRY_CAST(date AS DOUBLE)) "
        "FROM assessments"
    ).fetchone()
    a_null = con.execute(
        "SELECT count(*) FROM assessments WHERE date IS NULL"
    ).fetchone()[0]
    a_non_numeric = con.execute(
        "SELECT count(*) FROM assessments "
        "WHERE date IS NOT NULL AND TRY_CAST(date AS DOUBLE) IS NULL"
    ).fetchone()[0]
    out.append(
        "assessments.date range (via TRY_CAST(date AS DOUBLE)): "
        f"[{a_min}, {a_max}]"
    )
    out.append(f"assessments.date null count (true SQL NULL): {a_null}")
    out.append(
        f"assessments.date non-numeric placeholder count (e.g. '?'): {a_non_numeric}"
    )
    out.append("")

    # 6. student_vle.
    out.append(section("6. student_vle"))
    sv_count = con.execute("SELECT count(*) FROM student_vle").fetchone()[0]
    sv_min, sv_max = con.execute(
        "SELECT min(date), max(date) FROM student_vle"
    ).fetchone()
    sv_neg = con.execute(
        "SELECT count(*) FROM student_vle WHERE date < 0"
    ).fetchone()[0]
    sv_distinct_students = con.execute(
        "SELECT count(DISTINCT id_student) FROM student_vle"
    ).fetchone()[0]
    sv_distinct_sites = con.execute(
        "SELECT count(DISTINCT id_site) FROM student_vle"
    ).fetchone()[0]
    out.append(f"rows: {sv_count}")
    out.append(f"date range: [{sv_min}, {sv_max}]")
    out.append(f"rows with date < 0: {sv_neg}")
    out.append(f"distinct id_student: {sv_distinct_students}")
    out.append(f"distinct id_site: {sv_distinct_sites}")
    out.append("")

    # 7. vle.
    out.append(section("7. vle"))
    v_count = con.execute("SELECT count(*) FROM vle").fetchone()[0]
    out.append(f"rows: {v_count}")
    out.append("distinct activity_type values and counts:")
    for value, n in con.execute(
        "SELECT activity_type, count(*) AS n FROM vle "
        "GROUP BY activity_type ORDER BY n DESC"
    ).fetchall():
        out.append(f"  {value!r}: {n}")
    out.append("")

    # 8. Referential integrity.
    out.append(section("8. Referential integrity"))

    sv_orphan_students = con.execute(
        """
        SELECT count(*)
        FROM student_vle sv
        LEFT JOIN student_info si
          ON sv.id_student = si.id_student
         AND sv.code_module = si.code_module
         AND sv.code_presentation = si.code_presentation
        WHERE si.id_student IS NULL
        """
    ).fetchone()[0]
    out.append(
        "student_vle.id_student not found in student_info for the same "
        f"module-presentation: {sv_orphan_students}"
    )

    sa_orphan_students = con.execute(
        """
        SELECT count(*)
        FROM student_assessment sa
        LEFT JOIN student_info si ON sa.id_student = si.id_student
        WHERE si.id_student IS NULL
        """
    ).fetchone()[0]
    out.append(
        f"student_assessment.id_student not found in student_info: "
        f"{sa_orphan_students}"
    )

    sa_orphan_assessments = con.execute(
        """
        SELECT count(*)
        FROM student_assessment sa
        LEFT JOIN assessments a ON sa.id_assessment = a.id_assessment
        WHERE a.id_assessment IS NULL
        """
    ).fetchone()[0]
    out.append(
        f"student_assessment.id_assessment not found in assessments: "
        f"{sa_orphan_assessments}"
    )

    sv_orphan_sites = con.execute(
        """
        SELECT count(*)
        FROM student_vle sv
        LEFT JOIN vle v ON sv.id_site = v.id_site
        WHERE v.id_site IS NULL
        """
    ).fetchone()[0]
    out.append(f"student_vle.id_site not found in vle: {sv_orphan_sites}")
    out.append("")

    con.close()

    report_text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text + "\n")

    print(report_text)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
