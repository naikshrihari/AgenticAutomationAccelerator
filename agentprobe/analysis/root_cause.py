"""Cluster failed cases and diagnose each cluster with the local LLM.

The flow:

1. Take the graded results, keep the genuine failures (fail / partial; infra
   errors are excluded — they are not the agent's fault).
2. Build a short "failure signature" per case (question, expected vs actual,
   missing facts) and embed it with the local embedding model.
3. Greedily cluster the signatures by cosine similarity so similar failures group
   together (e.g. all "retrieved the wrong section" cases).
4. Ask the judge model to diagnose each cluster: a category, the likely cause,
   and a concrete suggested fix.

Everything runs on the local models, so this adds insight at no extra cost and
keeps data on-premise. If embeddings are unavailable, all failures fall into a
single cluster so the diagnosis still runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..llm.embeddings import EmbeddingModel, cosine
from ..llm.ollama_client import OllamaClient
from ..models import GradedResult, TestCase, Verdict

logger = logging.getLogger(__name__)


class FailureCluster(BaseModel):
    """One group of related failures, with an LLM diagnosis."""

    category: str = "uncategorized"
    summary: str = ""
    likely_cause: str = ""
    suggested_fix: str = ""
    case_ids: list[str] = Field(default_factory=list)
    sample_questions: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.case_ids)


class RootCauseReport(BaseModel):
    """The full root-cause analysis for a run."""

    total_failures: int = 0
    clusters: list[FailureCluster] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_findings(self) -> bool:
        return bool(self.clusters)


class _Diagnosis(BaseModel):
    """The shape we ask the judge model to return for a cluster."""

    category: str
    summary: str
    likely_cause: str
    suggested_fix: str


_SYSTEM = (
    "You are a senior QA lead analysing why an AI agent failed a group of similar "
    "test cases. You are given the questions, the expected answers, the agent's "
    "actual answers, and which required facts were missing. Diagnose the shared "
    "root cause and propose one concrete, actionable fix (e.g. retrieval/ranking "
    "change, prompt instruction, knowledge-base gap, formatting, wrong refusal). "
    "Be specific and concise; do not restate every case."
)

# Common, human-readable buckets we nudge the model toward for consistency.
_CATEGORY_HINT = (
    "Pick a short category such as: wrong_source_retrieved, hallucination, "
    "missing_knowledge, incomplete_answer, wrong_refusal, formatting, "
    "outdated_information, or other."
)


class RootCauseAnalyzer:
    """Produces a :class:`RootCauseReport` from graded results."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        embedder: Optional[EmbeddingModel] = None,
        settings: Optional[Settings] = None,
        similarity_threshold: float = 0.78,
        max_clusters: int = 8,
        include_partial: bool = True,
    ):
        self.settings = settings or Settings.from_env()
        self.client = client or OllamaClient(self.settings, model=self.settings.judge_model)
        self.embedder = embedder or EmbeddingModel(self.settings)
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self.include_partial = include_partial

    def analyze(
        self,
        results: list[GradedResult],
        cases_by_id: Optional[dict[str, TestCase]] = None,
    ) -> RootCauseReport:
        cases_by_id = cases_by_id or {}
        failures = [r for r in results if self._is_failure(r)]
        if not failures:
            return RootCauseReport(total_failures=0)

        groups = self._cluster(failures)
        clusters: list[FailureCluster] = []
        for group in groups[: self.max_clusters]:
            clusters.append(self._diagnose(group, cases_by_id))
        # Largest clusters first — they matter most.
        clusters.sort(key=lambda c: c.size, reverse=True)
        return RootCauseReport(total_failures=len(failures), clusters=clusters)

    # -- internals -------------------------------------------------------- #
    def _is_failure(self, r: GradedResult) -> bool:
        if r.verdict is Verdict.FAIL:
            return True
        return self.include_partial and r.verdict is Verdict.PARTIAL

    def _signature(self, r: GradedResult) -> str:
        missing = "; ".join(r.fact_coverage.missing) if r.fact_coverage.missing else "n/a"
        return (
            f"QUESTION: {r.question}\n"
            f"EXPECTED: {r.expected_answer}\n"
            f"AGENT: {r.agent_answer or '(no answer)'}\n"
            f"MISSING FACTS: {missing}"
        )

    def _cluster(self, failures: list[GradedResult]) -> list[list[GradedResult]]:
        """Greedy cosine clustering; falls back to one cluster without embeddings."""
        try:
            vectors = self.embedder.embed([self._signature(r) for r in failures])
        except Exception as exc:  # noqa: BLE001 - embeddings optional
            logger.warning("root-cause: embeddings unavailable (%s); one cluster", exc)
            return [failures]

        clusters: list[list[GradedResult]] = []
        centroids: list[list[float]] = []
        for r, vec in zip(failures, vectors):
            placed = False
            for i, centroid in enumerate(centroids):
                if cosine(vec, centroid) >= self.similarity_threshold:
                    clusters[i].append(r)
                    placed = True
                    break
            if not placed:
                clusters.append([r])
                centroids.append(vec)
        return clusters

    def _diagnose(self, group: list[GradedResult], cases_by_id: dict[str, TestCase]) -> FailureCluster:
        case_ids = [r.case_id for r in group]
        sample_questions = [r.question for r in group[:5]]
        # Cap how many cases we feed the model so the prompt stays bounded.
        blob = "\n\n".join(self._signature(r) for r in group[:8])
        user = (
            f"{len(group)} similar failing cases:\n\n{blob}\n\n"
            f"{_CATEGORY_HINT}\n"
            "Return category, summary, likely_cause, and suggested_fix."
        )
        try:
            diag = self.client.structured(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                _Diagnosis,
                model=self.settings.judge_model,
                temperature=self.settings.judge_temperature,
            )
        except Exception as exc:  # noqa: BLE001 - never fail the run over analysis
            logger.warning("root-cause: diagnosis failed (%s)", exc)
            return FailureCluster(
                category="undiagnosed",
                summary="Could not diagnose automatically.",
                case_ids=case_ids,
                sample_questions=sample_questions,
            )
        return FailureCluster(
            category=diag.category,
            summary=diag.summary,
            likely_cause=diag.likely_cause,
            suggested_fix=diag.suggested_fix,
            case_ids=case_ids,
            sample_questions=sample_questions,
        )
