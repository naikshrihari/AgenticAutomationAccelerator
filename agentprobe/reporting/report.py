"""Report builder: HTML report, CSV export, and native-evaluator export.

Produces a self-contained HTML report (via Jinja2, with a stdlib fallback so the
package works without it), a flat CSV for spreadsheets, and a JSON export shaped
for pushing the graded set into a platform's native evaluator (Oracle Evaluation
Sets, ServiceNow Agentic Evaluations).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from ..models import GradedResult, RunSummary, Verdict
from .differ import RegressionReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportBuilder:
    """Renders run results into shareable artifacts."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- CSV -------------------------------------------------------------- #
    def to_csv(self, results: list[GradedResult], filename: str = "results.csv") -> Path:
        path = self.out_dir / filename
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["case_id", "question_type", "verdict", "score", "fact_coverage",
                 "correctness", "groundedness", "completeness", "latency_ms",
                 "error_kind", "rationale"]
            )
            for r in results:
                writer.writerow(
                    [r.case_id, r.question_type.value, r.verdict.value, r.score,
                     round(r.fact_coverage.ratio, 3), r.judge.correctness,
                     r.judge.groundedness, r.judge.completeness, round(r.latency_ms, 1),
                     r.error_kind, r.rationale.replace("\n", " ")]
                )
        return path

    # -- HTML ------------------------------------------------------------- #
    def to_html(
        self,
        summary: RunSummary,
        results: list[GradedResult],
        regression: Optional[RegressionReport] = None,
        filename: str = "report.html",
    ) -> Path:
        by_type = _breakdown_by_type(results)
        context = {
            "summary": summary,
            "results": results,
            "by_type": by_type,
            "regression": regression,
            "Verdict": Verdict,
        }
        html = self._render(context)
        path = self.out_dir / filename
        path.write_text(html, encoding="utf-8")
        return path

    # -- native evaluator export ----------------------------------------- #
    def to_native_evaluator(self, results: list[GradedResult], filename: str = "eval_set.json") -> Path:
        """Export a payload other platforms' evaluators can ingest."""
        payload = {
            "version": 1,
            "cases": [
                {
                    "id": r.case_id,
                    "input": r.question,
                    "expected_output": r.expected_answer,
                    "actual_output": r.agent_answer,
                    "verdict": r.verdict.value,
                    "score": r.score,
                }
                for r in results
            ],
        }
        path = self.out_dir / filename
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # -- rendering -------------------------------------------------------- #
    def _render(self, context: dict) -> str:
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            env = Environment(
                loader=FileSystemLoader(str(_TEMPLATE_DIR)),
                autoescape=select_autoescape(["html"]),
            )
            return env.get_template("report.html.j2").render(**context)
        except ImportError:
            return _render_fallback(context)


def _breakdown_by_type(results: list[GradedResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = out.setdefault(r.question_type.value, Counter())
        bucket[r.verdict.value] += 1
    return {k: dict(v) for k, v in out.items()}


def _render_fallback(context: dict) -> str:
    """Minimal HTML when Jinja2 is not installed."""
    s: RunSummary = context["summary"]
    rows = "".join(
        f"<tr><td>{r.case_id}</td><td>{r.question_type.value}</td>"
        f"<td>{r.verdict.value}</td><td>{r.score:.2f}</td></tr>"
        for r in context["results"]
    )
    return (
        f"<html><head><meta charset='utf-8'><title>AgentProbe — {s.target}</title></head>"
        f"<body><h1>AgentProbe report: {s.target}</h1>"
        f"<p>Pass rate: {s.pass_rate:.1%} — {s.passed} pass / {s.partial} partial / "
        f"{s.failed} fail / {s.errors} error (of {s.total})</p>"
        f"<table border='1' cellpadding='4'><tr><th>Case</th><th>Type</th>"
        f"<th>Verdict</th><th>Score</th></tr>{rows}</table></body></html>"
    )
