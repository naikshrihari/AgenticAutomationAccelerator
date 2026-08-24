"""Self-consistency validation of generated cases.

A second pass on Ollama re-reads the source chunk and judges whether the
expected answer is actually supported by it. Unsupported cases are flagged (and,
by policy, dropped from the golden set) so hallucinated answers never become the
ground truth we grade agents against. Out-of-scope cases are handled inversely:
they are supported precisely when the chunk does NOT answer the question.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from ..config import Settings
from ..llm.ollama_client import OllamaClient
from ..models import Chunk, QuestionType, TestCase


class _Support(BaseModel):
    supported: bool
    reason: str = ""


_SYSTEM = (
    "You verify whether an expected answer is fully supported by a source "
    "section. Answer strictly from the source; do not use outside knowledge."
)


class SelfConsistencyChecker:
    """Confirms each expected answer is grounded in its source chunk."""

    def __init__(self, client: Optional[OllamaClient] = None, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        # Judge model is the stronger one; grounding is a judgement task.
        self.client = client or OllamaClient(self.settings, model=self.settings.judge_model)

    def check(self, case: TestCase, chunk: Chunk) -> bool:
        """Return True if the case survives the consistency check."""
        user = (
            f"SOURCE SECTION:\n\"\"\"\n{chunk.text}\n\"\"\"\n\n"
            f"QUESTION: {case.question}\n"
            f"EXPECTED ANSWER: {case.expected_answer}\n\n"
            "Is the expected answer fully supported by the source section?"
        )
        result = self.client.structured(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            _Support,
            model=self.settings.judge_model,
            temperature=self.settings.judge_temperature,
        )
        supported = result.supported
        # For out-of-scope cases, "supported" means the source genuinely does
        # not answer it, so we invert the signal.
        if case.question_type is QuestionType.OUT_OF_SCOPE:
            supported = not supported
        case.self_consistent = supported
        return supported

    def filter(self, cases: list[TestCase], chunks_by_id: dict[str, Chunk]) -> list[TestCase]:
        """Return only cases whose expected answer is grounded in their chunk."""
        kept: list[TestCase] = []
        for case in cases:
            chunk = chunks_by_id.get(case.source_chunk_id)
            if chunk is None:
                continue
            if self.check(case, chunk):
                kept.append(case)
        return kept
