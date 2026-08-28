"""
Stage 4 hard guard: THE TEST SPLIT MUST NOT BE READ.

Every data-loading function in Stage 4 goes through load_split(), which
raises if 'test' appears in the requested splits. This is the single place
that enforces the rule so no code path can quietly bypass it.
"""

import duckdb
import pandas as pd

ALLOWED_SPLITS = {"train", "validate", "test"}


def load_split(con: duckdb.DuckDBPyConnection, table: str, splits: list[str]) -> pd.DataFrame:
    if "test" in splits:
        raise RuntimeError(
            f"Stage 4 guard: refusing to load split='test' from {table}. "
            "The test split must not be read at Stage 4, for any reason."
        )
    unknown = set(splits) - ALLOWED_SPLITS
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown} requested for {table}.")
    if not splits:
        raise ValueError("load_split requires at least one split.")

    placeholders = ", ".join(f"'{s}'" for s in splits)
    return con.execute(f"SELECT * FROM {table} WHERE split IN ({placeholders})").df()
