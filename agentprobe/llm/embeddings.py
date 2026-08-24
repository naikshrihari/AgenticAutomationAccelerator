"""Embedding model wrapper used for dedup and the semantic index.

Uses Ollama's OpenAI-compatible ``/v1/embeddings`` endpoint with a dedicated
embedding model (``nomic-embed-text`` by default). Cosine similarity powers both
near-duplicate removal of generated cases and a lightweight semantic index.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import httpx

from ..config import Settings


class EmbeddingModel:
    """Batch-capable embedding client for the local Ollama server."""

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None):
        self.settings = settings or Settings.from_env()
        self.model = model or self.settings.embedding_model
        self._client = httpx.Client(
            base_url=self.settings.ollama_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.ollama_api_key}"},
            timeout=self.settings.request_timeout_s,
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input string."""
        if not texts:
            return []
        resp = self._client.post(
            "/embeddings", json={"model": self.model, "input": list(texts)}
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # The API preserves input order via the 'index' field.
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in ordered]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
