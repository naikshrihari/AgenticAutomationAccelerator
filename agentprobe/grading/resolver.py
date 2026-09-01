"""Verdict resolver.

Combines the three grading signals — the deterministic fact-coverage ratio, the
LLM judge's scores, and the groundedness check — into a single verdict (pass,
partial, or fail) with an aggregate score and the judge's rationale. Thresholds
come from the target config so different agents can be held to different bars.
Infrastructure errors from execution short-circuit to an ERROR verdict and are
excluded from the pass rate upstream.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Settings, Thresholds
from ..models import AgentResponse, GradedResult, Verdict
from ..llm.ollama_client import OllamaClient
from .fact_coverage import check_fact_coverage
from .groundedness import check_groundedness
from .judge import LLMJudge

logger = logging.getLogger(__name__)

# Weights for the aggregate score. Fact coverage and judged correctness carry
# the most weight; completeness and groundedness refine the margin.
_WEIGHTS = {
    "fact_coverage": 0.35,
    "correctness": 0.35,
    "completeness": 0.15,
    "groundedness": 0.15,
}


class VerdictResolver:
    """Runs the full grading stack for one (case, response) pair."""

    def __init__(
        self,
        thresholds: Optional[Thresholds] = None,
        judge: Optional[LLMJudge] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.thresholds = thresholds or Thresholds()
        self.judge = judge or LLMJudge(settings=self.settings)

    def grade(self, case, response: AgentResponse) -> GradedResult:
        # Infrastructure failures never count as a wrong answer.
        if not response.ok:
            return GradedResult(
                case_id=case.case_id,
                question=case.question,
                expected_answer=case.expected_answer,
                agent_answer="",
                verdict=Verdict.ERROR,
                score=0.0,
                rationale=f"Infrastructure error: {response.error_kind} ({response.error_detail})",
                question_type=case.question_type,
                latency_ms=response.latency_ms,
                error_kind=response.error_kind,
            )

        coverage = check_fact_coverage(case.required_facts, response.answer)
        grounded = check_groundedness(case, response)

        # The LLM judge can time out or error on a slow local model. Isolate that
        # so one bad grading call doesn't crash the whole run: mark just this case
        # as an ERROR (excluded from the pass rate) and keep going.
        try:
            judge = self.judge.score(case, response.answer)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge failed for case %s: %s", case.case_id, exc)
            return GradedResult(
                case_id=case.case_id,
                question=case.question,
                expected_answer=case.expected_answer,
                agent_answer=response.answer,
                verdict=Verdict.ERROR,
                score=0.0,
                fact_coverage=coverage,
                groundedness_ok=grounded,
                rationale=f"Grading error (judge unavailable): {exc}",
                question_type=case.question_type,
                latency_ms=response.latency_ms,
                error_kind="judge_error",
            )

        score = (
            _WEIGHTS["fact_coverage"] * coverage.ratio
            + _WEIGHTS["correctness"] * judge.correctness
            + _WEIGHTS["completeness"] * judge.completeness
            + _WEIGHTS["groundedness"] * (judge.groundedness if grounded else judge.groundedness * 0.5)
        )
        verdict = self._resolve(score, grounded)

        return GradedResult(
            case_id=case.case_id,
            question=case.question,
            expected_answer=case.expected_answer,
            agent_answer=response.answer,
            verdict=verdict,
            score=round(score, 4),
            fact_coverage=coverage,
            judge=judge,
            groundedness_ok=grounded,
            rationale=judge.rationale,
            question_type=case.question_type,
            latency_ms=response.latency_ms,
        )

    def _resolve(self, score: float, grounded: bool) -> Verdict:
        if self.thresholds.require_groundedness and not grounded:
            # An ungrounded answer can never fully pass when grounding is required.
            return Verdict.PARTIAL if score >= self.thresholds.partial_score else Verdict.FAIL
        if score >= self.thresholds.pass_score:
            return Verdict.PASS
        if score >= self.thresholds.partial_score:
            return Verdict.PARTIAL
        return Verdict.FAIL
