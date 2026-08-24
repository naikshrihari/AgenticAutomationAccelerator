"""Typed data models that flow through the AgentProbe pipeline.

Every stage exchanges these Pydantic models so that the boundaries between
ingestion, generation, execution, grading, and reporting stay explicit and
validated. Test cases and verdicts are serialised to JSONL for the versioned
golden set and the results store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QuestionType(str, Enum):
    """The kinds of questions the generator is asked to produce.

    These map directly to the coverage the accelerator aims for: not just happy
    path facts, but the edges where agents typically fail.
    """

    FACTUAL = "factual"
    EDGE_CASE = "edge_case"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"
    PARAPHRASE = "paraphrase"
    BILINGUAL = "bilingual"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Verdict(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    ERROR = "error"  # infrastructure failure, kept out of the pass rate


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class Chunk(BaseModel):
    """A section-aware slice of a source document.

    The chunker preserves headings so that every generated question can be
    traced back to the exact section it came from.
    """

    chunk_id: str
    document: str = Field(..., description="Source document filename or id")
    document_version: str = Field(
        default="unknown",
        description="Hash or version tag of the source document at ingest time",
    )
    section: str = Field(default="", description="Heading path, e.g. 'Leave > Sick leave'")
    text: str
    audience: str = Field(default="general")
    language: str = Field(default="en")
    token_estimate: int = 0

    def citation(self) -> str:
        return f"{self.document} § {self.section}".strip(" §")


# --------------------------------------------------------------------------- #
# Generation / Golden set
# --------------------------------------------------------------------------- #
class TestCase(BaseModel):
    """A single graded test case in the golden set.

    Produced by the generation stage, validated by the self-consistency pass,
    and stored as a line of versioned JSONL together with its citation.
    """

    case_id: str
    question: str
    expected_answer: str
    required_facts: list[str] = Field(
        default_factory=list,
        description="Atomic facts a correct answer must contain; drives the "
        "deterministic fact-coverage checklist during grading",
    )
    citation: str = ""
    source_chunk_id: str = ""
    document: str = ""
    document_version: str = "unknown"
    question_type: QuestionType = QuestionType.FACTUAL
    difficulty: Difficulty = Difficulty.MEDIUM
    language: str = "en"

    # Provenance / review
    generated_by: str = ""  # model name
    self_consistent: Optional[bool] = None
    approved: bool = False
    is_seed: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
class AgentResponse(BaseModel):
    """The raw outcome of asking the target agent one question."""

    case_id: str
    answer: str = ""
    cited_sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    ok: bool = True
    error_kind: str = ""  # 'timeout', 'auth', 'http', 'connector' ... empty when ok
    error_detail: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
class FactCoverage(BaseModel):
    """Deterministic check: which required facts appear in the response."""

    covered: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @property
    def ratio(self) -> float:
        total = len(self.covered) + len(self.missing)
        return 1.0 if total == 0 else len(self.covered) / total


class JudgeScore(BaseModel):
    """LLM-as-judge output, scored against the expected answer and source."""

    correctness: float = 0.0  # 0..1
    groundedness: float = 0.0  # 0..1
    completeness: float = 0.0  # 0..1
    rationale: str = ""


class GradedResult(BaseModel):
    """The verdict resolver's combined judgement for one case."""

    case_id: str
    question: str
    expected_answer: str
    agent_answer: str
    verdict: Verdict
    score: float = 0.0  # 0..1 aggregate
    fact_coverage: FactCoverage = Field(default_factory=FactCoverage)
    judge: JudgeScore = Field(default_factory=JudgeScore)
    groundedness_ok: bool = False
    rationale: str = ""
    question_type: QuestionType = QuestionType.FACTUAL
    latency_ms: float = 0.0
    error_kind: str = ""
    graded_at: datetime = Field(default_factory=_utcnow)


class RunSummary(BaseModel):
    """Aggregate metrics for one execution of the suite against a target."""

    run_id: str
    target: str
    total: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    errors: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None

    @property
    def scored(self) -> int:
        """Cases that count toward the pass rate (infra errors excluded)."""
        return self.total - self.errors

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.scored == 0 else self.passed / self.scored
