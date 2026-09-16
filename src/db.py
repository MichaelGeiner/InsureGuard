"""Shared PostgreSQL connection helpers. Settings come from .env (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# psycopg's pure-Python driver needs libpq.dll; use the one shipped with PostgreSQL
if os.environ.get("PG_BIN"):
    os.environ["PATH"] = os.environ["PG_BIN"] + os.pathsep + os.environ["PATH"]

import psycopg  # noqa: E402


def conninfo(dbname: str | None = None) -> str:
    return psycopg.conninfo.make_conninfo(
        host=os.environ["PGHOST"], port=os.environ.get("PGPORT", "5432"),
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        dbname=dbname or os.environ["PGDATABASE"],
    )


def connect(dbname: str | None = None, **kwargs) -> psycopg.Connection:
    return psycopg.connect(conninfo(dbname), **kwargs)


def run_sql_file(conn: psycopg.Connection, name: str) -> None:
    conn.execute((ROOT / "sql" / name).read_text(encoding="utf-8-sig"))  # tolerate BOMs from Windows editors
