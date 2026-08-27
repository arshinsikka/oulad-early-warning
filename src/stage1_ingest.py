"""
Stage 1 ingest: extract the OULAD zip, checksum the CSVs, load them into
DuckDB. No target definition, no feature engineering, no modelling.

Usage:
    .venv/bin/python src/stage1_ingest.py /path/to/oulad.zip
"""

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHECKSUMS_PATH = ROOT / "data" / "CHECKSUMS.txt"
DB_PATH = ROOT / "data" / "oulad.duckdb"

# Lowercased CSV stem -> target DuckDB table name. Matching is
# case-insensitive on purpose; we do not assume the capitalisation the
# archive ships with.
EXPECTED_CONCEPTS = {
    "studentinfo": "student_info",
    "studentregistration": "student_registration",
    "studentassessment": "student_assessment",
    "assessments": "assessments",
    "courses": "courses",
    "vle": "vle",
    "studentvle": "student_vle",
}


def extract_zip(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name:
                continue
            target = raw_dir / name
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def discover_csvs(raw_dir: Path) -> dict[str, Path]:
    csv_files = sorted(
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"
    )

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}:")
    for p in csv_files:
        print(f"  {p.name}")

    matched: dict[str, Path] = {}
    unmatched: list[Path] = []
    for p in csv_files:
        key = p.stem.lower()
        table = EXPECTED_CONCEPTS.get(key)
        if table is None:
            unmatched.append(p)
            continue
        if table in matched:
            raise RuntimeError(
                f"Two CSV files both map to the '{table}' concept: "
                f"'{matched[table].name}' and '{p.name}'. Refusing to guess "
                "which one is correct."
            )
        matched[table] = p

    if unmatched:
        raise RuntimeError(
            "Found CSV file(s) that do not match any of the seven expected "
            f"OULAD concepts: {[p.name for p in unmatched]}. Expected one "
            f"file for each of: {sorted(EXPECTED_CONCEPTS.keys())}."
        )

    missing = set(EXPECTED_CONCEPTS.values()) - set(matched.keys())
    if len(csv_files) != 7 or missing:
        raise RuntimeError(
            f"Expected exactly 7 CSV files, one per OULAD table, found "
            f"{len(csv_files)}. Missing concept(s): {sorted(missing)}."
        )

    return matched


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_checksums(matched: dict[str, Path]) -> dict[str, tuple[int, str]]:
    checksums = {}
    for path in matched.values():
        checksums[path.name] = (path.stat().st_size, sha256_of(path))
    return checksums


def read_existing_checksums(path: Path) -> dict[str, tuple[int, str]]:
    existing = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        filename, size, sha = [part.strip() for part in line.split(",")]
        existing[filename] = (int(size), sha)
    return existing


def verify_or_write_checksums(new_checksums: dict[str, tuple[int, str]]) -> None:
    if CHECKSUMS_PATH.exists():
        existing = read_existing_checksums(CHECKSUMS_PATH)
        for filename, (size, sha) in new_checksums.items():
            if filename not in existing:
                raise RuntimeError(
                    f"data/CHECKSUMS.txt exists but has no entry for "
                    f"'{filename}'. Not overwriting the existing file."
                )
            old_size, old_sha = existing[filename]
            if old_size != size or old_sha != sha:
                raise RuntimeError(
                    f"Checksum mismatch for '{filename}': CHECKSUMS.txt "
                    f"records size={old_size}, sha256={old_sha}; the file "
                    f"on disk now has size={size}, sha256={sha}. The source "
                    "CSV has changed since first ingest. Not overwriting "
                    "data/CHECKSUMS.txt."
                )
        print(f"Verified {len(new_checksums)} file(s) against existing data/CHECKSUMS.txt — no changes.")
    else:
        lines = [
            f"{filename}, {size}, {sha}"
            for filename, (size, sha) in sorted(new_checksums.items())
        ]
        CHECKSUMS_PATH.write_text("\n".join(lines) + "\n")
        print(f"Wrote data/CHECKSUMS.txt with {len(new_checksums)} entries.")


def load_into_duckdb(matched: dict[str, Path]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        for table, path in sorted(matched.items()):
            escaped = str(path).replace("'", "''")
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS "
                f"SELECT * FROM read_csv_auto('{escaped}')"
            )
            count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  loaded {table}: {count} rows")
    finally:
        con.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: stage1_ingest.py /path/to/oulad.zip", file=sys.stderr)
        sys.exit(1)

    zip_path = Path(sys.argv[1]).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"No such file: {zip_path}")

    print(f"Extracting {zip_path} into {RAW_DIR} ...")
    extract_zip(zip_path, RAW_DIR)

    matched = discover_csvs(RAW_DIR)

    print("Computing SHA-256 checksums ...")
    new_checksums = compute_checksums(matched)
    verify_or_write_checksums(new_checksums)

    print(f"Loading tables into {DB_PATH} ...")
    load_into_duckdb(matched)

    print("Stage 1 ingest complete.")


if __name__ == "__main__":
    main()
