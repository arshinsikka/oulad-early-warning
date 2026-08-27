# OULAD Early Warning

An offline study on the Open University Learning Analytics Dataset (OULAD)
investigating whether students at risk of non-completion can be identified
early enough, and reliably enough, for an intervention to be worth running.

The analytical protocol is fixed in advance in `docs/PROTOCOL.md`, committed
as the root commit of this repository before any code or data existed. Every
later decision is made against that document; deviations are recorded as new
commits with the original wording intact.

## Stage 1: Ingest and Validation

Extracts the OULAD CSVs, checksums them, loads them into DuckDB, and reports
their structure. No target definition, no feature engineering, no modelling.

### Setup

```
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run

```
.venv/bin/python src/stage1_ingest.py /path/to/oulad.zip
.venv/bin/python src/stage1_validate.py
```

`stage1_ingest.py` extracts the zip, checksums the CSVs into
`data/CHECKSUMS.txt`, and loads the tables into `data/oulad.duckdb`.
`stage1_validate.py` queries that database and writes
`reports/stage1_validation.txt`.
