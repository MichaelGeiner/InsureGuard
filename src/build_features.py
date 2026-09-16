"""
InsureGuard | Step 2: build the transaction feature table in PostgreSQL.

Runs sql/02_features.sql (creates insureguard.txn_features), then prints
sql/03_feature_validation.sql so you can see each fraud scenario separate
from legitimate traffic.

Usage:
  python src/build_features.py
"""
from __future__ import annotations

import time

import pandas as pd

from db import ROOT, connect, run_sql_file


def main() -> None:
    with connect() as conn:
        start = time.perf_counter()
        run_sql_file(conn, "02_features.sql")
        conn.commit()
        rows = conn.execute("SELECT COUNT(*) FROM insureguard.txn_features").fetchone()[0]
        print(f"built insureguard.txn_features: {rows:,} rows in {time.perf_counter() - start:.1f}s\n")

        cur = conn.execute((ROOT / "sql" / "03_feature_validation.sql").read_text(encoding="utf-8"))
        report = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])

    numeric = report.columns.drop("scenario")
    report[numeric] = report[numeric].astype(float).round(2)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
