"""Local LLM access. All inference is routed to Ollama."""

from .ollama_client import OllamaClient
from .embeddings import EmbeddingModel

__all__ = ["OllamaClient", "EmbeddingModel"]
