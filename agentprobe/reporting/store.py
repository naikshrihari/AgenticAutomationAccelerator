"""Results store.

Persists graded results and run summaries to a local database. DuckDB is used
when available (columnar, great for analytical queries over many runs); otherwise
we fall back to the stdlib ``sqlite3`` so the accelerator has zero mandatory
database dependency. An Oracle table backend can be added behind the same
interface without touching callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models import GradedResult, RunSummary


class ResultsStore:
    """A thin key-value-ish store over graded results, backed by DuckDB or SQLite."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.backend, self._conn = self._connect()
        self._ensure_schema()

    def _connect(self):
        try:
            import duckdb  # type: ignore

            return "duckdb", duckdb.connect(self.path)
        except ImportError:
            import sqlite3

            conn = sqlite3.connect(self.path)
            return "sqlite", conn

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                target TEXT,
                total INTEGER, passed INTEGER, partial INTEGER,
                failed INTEGER, errors INTEGER,
                pass_rate DOUBLE,
                started_at TEXT, finished_at TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT,
                case_id TEXT,
                question_type TEXT,
                verdict TEXT,
                score DOUBLE,
                latency_ms DOUBLE,
                error_kind TEXT,
                payload TEXT
            )
            """
        )
        self._commit()

    # -- writes ----------------------------------------------------------- #
    def save_run(self, summary: RunSummary, results: list[GradedResult]) -> None:
        self._conn.execute(
            f"INSERT INTO runs VALUES ({self._ph(10)})",
            (
                summary.run_id,
                summary.target,
                summary.total,
                summary.passed,
                summary.partial,
                summary.failed,
                summary.errors,
                summary.pass_rate,
                summary.started_at.isoformat(),
                summary.finished_at.isoformat() if summary.finished_at else None,
            ),
        )
        for r in results:
            self._conn.execute(
                f"INSERT INTO results VALUES ({self._ph(8)})",
                (
                    summary.run_id,
                    r.case_id,
                    r.question_type.value,
                    r.verdict.value,
                    r.score,
                    r.latency_ms,
                    r.error_kind,
                    r.model_dump_json(),
                ),
            )
        self._commit()

    # -- reads ------------------------------------------------------------ #
    def load_results(self, run_id: str) -> list[GradedResult]:
        cur = self._conn.execute(
            f"SELECT payload FROM results WHERE run_id = {self._ph(1)}", (run_id,)
        )
        return [GradedResult.model_validate_json(row[0]) for row in cur.fetchall()]

    def list_runs(self, target: Optional[str] = None, limit: int = 200) -> list[dict]:
        """Return past runs (newest first) as dicts, for the history dashboard."""
        cols = ["run_id", "target", "total", "passed", "partial", "failed",
                "errors", "pass_rate", "started_at", "finished_at"]
        if target:
            cur = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM runs WHERE target = {self._ph(1)} "
                f"ORDER BY started_at DESC LIMIT {int(limit)}", (target,))
        else:
            cur = self._conn.execute(
                f"SELECT {', '.join(cols)} FROM runs ORDER BY started_at DESC LIMIT {int(limit)}")
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def targets(self) -> list[str]:
        """Distinct target names that have runs stored."""
        cur = self._conn.execute("SELECT DISTINCT target FROM runs ORDER BY target")
        return [row[0] for row in cur.fetchall()]

    def latest_run_id(self, target: str, before: Optional[str] = None) -> Optional[str]:
        if before:
            cur = self._conn.execute(
                f"SELECT run_id FROM runs WHERE target = {self._ph(1)} AND run_id < {self._ph(1)} "
                "ORDER BY started_at DESC LIMIT 1",
                (target, before),
            )
        else:
            cur = self._conn.execute(
                f"SELECT run_id FROM runs WHERE target = {self._ph(1)} ORDER BY started_at DESC LIMIT 1",
                (target,),
            )
        row = cur.fetchone()
        return row[0] if row else None

    # -- helpers ---------------------------------------------------------- #
    def _ph(self, n: int) -> str:
        """Return a comma-joined placeholder string for the active backend."""
        token = "?" if self.backend == "sqlite" else "?"
        return ", ".join([token] * n)

    def _commit(self) -> None:
        if self.backend == "sqlite":
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
