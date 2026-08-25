"""Question + expected-answer generation on Ollama.

For each chunk the generation model produces one or more test cases: a question,
a grounded expected answer, the atomic facts a correct answer must contain, a
source citation, a question type, and a difficulty. Structured output is
enforced with Pydantic so malformed generations are rejected and retried rather
than silently stored.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from ..config import Settings
from ..llm.ollama_client import OllamaClient
from ..models import Chunk, Difficulty, QuestionType, TestCase

logger = logging.getLogger(__name__)

# The mix of question types requested per chunk. Out-of-scope questions
# deliberately ask something the chunk cannot answer, to test refusal.
DEFAULT_TYPE_MIX: tuple[QuestionType, ...] = (
    QuestionType.FACTUAL,
    QuestionType.EDGE_CASE,
    QuestionType.MULTI_HOP,
    QuestionType.OUT_OF_SCOPE,
    QuestionType.PARAPHRASE,
    QuestionType.BILINGUAL,
)

_SYSTEM = (
    "You are a meticulous QA engineer building an evaluation set for an AI "
    "agent that answers questions from company policy documents. You write "
    "questions that are answerable ONLY from the provided source section, "
    "except for 'out_of_scope' questions, which must ask something the section "
    "does NOT cover so that a correct agent refuses or escalates. Every expected "
    "answer must be grounded strictly in the source text — never invent facts."
)

_INSTRUCTIONS = {
    QuestionType.FACTUAL: "a direct factual question answerable from the section",
    QuestionType.EDGE_CASE: "a boundary/exception question about a limit, threshold, or special case in the section",
    QuestionType.MULTI_HOP: "a question that requires combining two or more facts within the section",
    QuestionType.OUT_OF_SCOPE: "a plausible question the section does NOT answer; the expected answer states the agent should refuse or escalate rather than guess",
    QuestionType.PARAPHRASE: "a factual question worded very differently from the source phrasing",
    QuestionType.BILINGUAL: "the same style of factual question, written in Spanish, with a Spanish expected answer",
}


class _GeneratedCase(BaseModel):
    """The raw shape we ask the model to return (before we attach provenance)."""

    question: str
    expected_answer: str
    required_facts: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM


class CaseGenerator:
    """Generates :class:`TestCase` objects from chunks using the local model."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        settings: Optional[Settings] = None,
        type_mix: Iterable[QuestionType] = DEFAULT_TYPE_MIX,
        max_workers: int = 1,
    ):
        self.settings = settings or Settings.from_env()
        self.client = client or OllamaClient(self.settings, model=self.settings.generation_model)
        self.type_mix = tuple(type_mix)
        # How many chunks to process concurrently. Ollama serves requests in
        # parallel, so a small pool overlaps generation and cuts wall-clock time.
        self.max_workers = max(1, max_workers)

    def generate_for_chunk(self, chunk: Chunk) -> list[TestCase]:
        """Produce one test case per configured question type for a chunk."""
        cases: list[TestCase] = []
        for qtype in self.type_mix:
            try:
                gen = self._generate_one(chunk, qtype)
            except Exception:  # noqa: BLE001 - one bad type shouldn't kill the chunk
                continue
            cases.append(self._to_test_case(chunk, qtype, gen))
        return cases

    def generate(self, chunks: Iterable[Chunk]) -> list[TestCase]:
        """Generate cases for every chunk, logging progress as it goes."""
        chunk_list = list(chunks)
        total = len(chunk_list)
        out: list[TestCase] = []
        if self.max_workers == 1:
            for i, chunk in enumerate(chunk_list, start=1):
                out.extend(self.generate_for_chunk(chunk))
                logger.info("generation: chunk %d/%d done (%d cases so far)", i, total, len(out))
            return out

        # Concurrent path: overlap chunk generation across a thread pool.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.generate_for_chunk, c): c for c in chunk_list}
            done = 0
            for future in as_completed(futures):
                done += 1
                out.extend(future.result())
                logger.info("generation: chunk %d/%d done (%d cases so far)", done, total, len(out))
        return out

    # -- internals -------------------------------------------------------- #
    def _generate_one(self, chunk: Chunk, qtype: QuestionType) -> _GeneratedCase:
        user = (
            f"SOURCE SECTION (document: {chunk.document}, section: {chunk.section or 'n/a'}):\n"
            f"\"\"\"\n{chunk.text}\n\"\"\"\n\n"
            f"Write {_INSTRUCTIONS[qtype]}.\n"
            "Return: question, expected_answer, required_facts (2-5 atomic facts "
            "that must appear in a correct answer), and difficulty (easy|medium|hard)."
        )
        return self.client.structured(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            _GeneratedCase,
            model=self.settings.generation_model,
            temperature=self.settings.generation_temperature,
        )

    def _to_test_case(self, chunk: Chunk, qtype: QuestionType, gen: _GeneratedCase) -> TestCase:
        language = "es" if qtype is QuestionType.BILINGUAL else chunk.language
        return TestCase(
            case_id=f"gen-{uuid.uuid4().hex[:12]}",
            question=gen.question,
            expected_answer=gen.expected_answer,
            required_facts=gen.required_facts,
            citation=chunk.citation(),
            source_chunk_id=chunk.chunk_id,
            document=chunk.document,
            document_version=chunk.document_version,
            question_type=qtype,
            difficulty=gen.difficulty,
            language=language,
            generated_by=self.settings.generation_model,
        )
