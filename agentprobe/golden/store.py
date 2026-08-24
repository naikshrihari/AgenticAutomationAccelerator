"""Versioned JSONL store for the golden set.

Approved cases are written as one JSON object per line, together with their
citations and the document version they were generated from. Each save creates a
new immutable version file plus a ``latest`` pointer, so a run can always be tied
back to the exact set it used and regressions can be compared version to version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from ..models import TestCase


class GoldenSetStore:
    """Reads and writes versioned JSONL golden sets under a directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------- #
    def save_version(self, cases: Iterable[TestCase], version: Optional[str] = None) -> Path:
        """Persist ``cases`` as a new immutable version; return its path."""
        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.root / f"golden-{version}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for case in cases:
                fh.write(case.model_dump_json())
                fh.write("\n")
        # Update the 'latest' pointer.
        (self.root / "latest.txt").write_text(path.name, encoding="utf-8")
        return path

    # -- reading ---------------------------------------------------------- #
    def load(self, path: str | Path) -> list[TestCase]:
        cases: list[TestCase] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(TestCase.model_validate_json(line))
        return cases

    def load_latest(self) -> list[TestCase]:
        pointer = self.root / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"No golden set saved under {self.root}")
        return self.load(self.root / pointer.read_text(encoding="utf-8").strip())

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("golden-*.jsonl"))
