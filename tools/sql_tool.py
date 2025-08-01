# tools/sql_tool.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Sequence

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Result
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL") or "sqlite:///data/cricket.db"
# ↑ fallback lets your unit test run even without a .env

class SQLTool:
    """
    Thin wrapper around SQLAlchemy for read‑only inspection/query tasks.
    """

    _engine: Engine | None = None   # process‑wide singleton

    # ---------- engine helper ---------- #
    @classmethod
    def _get_engine(cls) -> Engine:
        if cls._engine is None:
            cls._engine = create_engine(DB_URL, pool_pre_ping=True)
        return cls._engine

    # ---------- public helpers ---------- #
    def list_tables(self) -> List[str]:
        """
        Return all table names in the connected database (sorted).
        """
        eng  = self._get_engine()
        insp = inspect(eng)
        return sorted(insp.get_table_names())

    def get_schema(self, table: str) -> str:
        """
        Return a human‑readable column list for `table`.
        """
        eng  = self._get_engine()
        insp = inspect(eng)
        cols = insp.get_columns(table)
        return ", ".join(f"{c['name']} {c['type']}" for c in cols)

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Run a **read‑only** SQL query and return rows as list[dict].
        """
        lowered = sql.lstrip().lower()
        if not lowered.startswith("select"):
            raise ValueError("Only SELECT statements are allowed.")

        eng = self._get_engine()
        with eng.connect() as conn:
            result: Result = conn.execute(text(sql))
            cols: Sequence[str] = result.keys()
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]
