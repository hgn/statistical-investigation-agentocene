"""Table IO with a graceful Parquet-or-CSV fallback.

Parquet needs pyarrow, which may be absent. We always write CSV (the brief wants
R/Pandas-friendly output anyway) and additionally write Parquet when an engine is
available. Loading prefers Parquet, falls back to CSV.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd


@cache
def has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, out_dir: Path, name: str, csv: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if has_parquet():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    if csv or not has_parquet():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def load_table(in_dir: Path, name: str) -> pd.DataFrame:
    pq = in_dir / f"{name}.parquet"
    if pq.exists() and has_parquet():
        return pd.read_parquet(pq)
    csv = in_dir / f"{name}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"no table {name}.(parquet|csv) in {in_dir}")
    return pd.read_csv(csv, dtype={"ym": str, "first_month": str, "last_month": str})
