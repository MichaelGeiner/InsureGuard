"""
InsureGuard | Step 1b: load the raw CSVs into PostgreSQL.

Creates the database if needed, applies sql/01_schema.sql, then bulk-loads
data/raw/*.csv with COPY (fast, and safe to re-run: tables are truncated first).

Connection settings come from .env (see .env.example).

Usage:
  python src/load_to_postgres.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
load_dotenv(ROOT / ".env")

# psycopg's pure-Python driver needs libpq.dll; use the one shipped with PostgreSQL
if os.environ.get("PG_BIN"):
    os.environ["PATH"] = os.environ["PG_BIN"] + os.pathsep + os.environ["PATH"]

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

# load order respects foreign keys
TABLES = [
    ("dim_customer", "customers.csv"),
    ("dim_device", "devices.csv"),
    ("fact_transaction", "transactions.csv"),
]


def conninfo(dbname: str) -> str:
    return psycopg.conninfo.make_conninfo(
        host=os.environ["PGHOST"], port=os.environ.get("PGPORT", "5432"),
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"], dbname=dbname,
    )


def ensure_database(name: str) -> None:
    with psycopg.connect(conninfo("postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            print(f"created database {name}")


def main() -> None:
    db =os.environ["PGDATABASE"]
    ensure_database(db)

    with psycopg.connect(conninfo(db)) as conn:
        conn.execute((ROOT / "sql" / "01_schema.sql").read_text())
        conn.execute("TRUNCATE insureguard.fact_transaction, insureguard.dim_device, insureguard.dim_customer")

        for table, filename in TABLES:
            path = RAW / filename
            with path.open(encoding="utf-8") as f:
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
