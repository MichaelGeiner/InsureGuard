"""
InsureGuard | Step 1b: load the raw CSVs into PostgreSQL.

Creates the database if needed, applies sql/01_schema.sql, then bulk-loads
data/raw/*.csv with COPY (fast, and safe to re-run: tables are truncated first).

Usage:
  python src/load_to_postgres.py
"""
from __future__ import annotations

import os

from db import ROOT, connect, run_sql_file  # must come first: makes libpq findable

from psycopg import sql  # noqa: E402

RAW = ROOT / "data" / "raw"

# load order respects foreign keys
TABLES = [
    ("dim_customer", "customers.csv"),
    ("dim_device", "devices.csv"),
    ("fact_transaction", "transactions.csv"),
]


def ensure_database(name: str) -> None:
    with connect("postgres", autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            print(f"created database {name}")


def main() -> None:
    ensure_database(os.environ["PGDATABASE"])

    with connect() as conn:
        run_sql_file(conn, "01_schema.sql")
        conn.execute("DROP TABLE IF EXISTS insureguard.txn_features")  # derived; rebuilt in Step 2
        conn.execute("TRUNCATE insureguard.fact_transaction, insureguard.dim_device, insureguard.dim_customer")

        for table, filename in TABLES:
            with (RAW / filename).open(encoding="utf-8") as f:
                columns = f.readline().strip().split(",")
                copy_sql = sql.SQL("COPY insureguard.{} ({}) FROM STDIN WITH (FORMAT csv)").format(
                    sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns)))
                with conn.cursor().copy(copy_sql) as copy:
                    while chunk := f.read(1 << 20):
                        copy.write(chunk)
            count = conn.execute(sql.SQL("SELECT COUNT(*) FROM insureguard.{}").format(sql.Identifier(table))).fetchone()[0]
            print(f"loaded {table:<18} {count:>8,} rows")

        conn.execute("ANALYZE insureguard.fact_transaction")


if __name__ == "__main__":
    main()
