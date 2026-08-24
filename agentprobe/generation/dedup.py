"""Embedding-based deduplication and a semantic index over generated cases.

Generation across many overlapping chunks produces near-duplicate questions.
The deduplicator embeds each question with the local embedding model and drops
any case whose question is within a cosine-similarity threshold of one already
kept. The same embeddings back a small in-memory semantic index for querying the
golden set ("show me cases like this one").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import Settings
from ..llm.embeddings import EmbeddingModel, cosine
from ..models import TestCase


class Deduplicator:
    """Removes near-duplicate cases by question-embedding similarity."""

    def __init__(
        self,
        embedder: Optional[EmbeddingModel] = None,
        settings: Optional[Settings] = None,
        threshold: float = 0.92,
    ):
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or EmbeddingModel(self.settings)
        self.threshold = threshold

    def dedupe(self, cases: list[TestCase]) -> list[TestCase]:
        if len(cases) <= 1:
            return list(cases)
        vectors = self.embedder.embed([c.question for c in cases])
        kept: list[TestCase] = []
        kept_vecs: list[list[float]] = []
        for case, vec in zip(cases, vectors):
            if any(cosine(vec, kv) >= self.threshold for kv in kept_vecs):
                continue
            kept.append(case)
            kept_vecs.append(vec)
        return kept


@dataclass
class _Entry:
    case: TestCase
    vector: list[float]


class SemanticIndex:
    """A tiny cosine-similarity index over the generated/golden cases."""

    def __init__(self, embedder: Optional[EmbeddingModel] = None, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or EmbeddingModel(self.settings)
        self._entries: list[_Entry] = []

    def add(self, cases: list[TestCase]) -> None:
        if not cases:
            return
        vectors = self.embedder.embed([c.question for c in cases])
        self._entries.extend(_Entry(case=c, vector=v) for c, v in zip(cases, vectors))

    def search(self, query: str, k: int = 5) -> list[tuple[TestCase, float]]:
        """Return the ``k`` most similar cases to a free-text query."""
        if not self._entries:
            return []
        qv = self.embedder.embed_one(query)
        scored = [(e.case, cosine(qv, e.vector)) for e in self._entries]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]
