"""Agentic Assurance Accelerator — AI Agent Test Automation Accelerator.

A Python pipeline that reads source documents, generates a graded test set,
runs it against a target AI agent through a platform connector, and reports the
results. All language-model work — question generation, expected-answer
generation, semantic grading, and embeddings — runs locally on Ollama, so
document and HR data never leave the network.

See :class:`agentprobe.pipeline.Pipeline` for the end-to-end entry point.
"""

from .config import Settings, TargetConfig
from .models import (
    Chunk,
    GradedResult,
    QuestionType,
    RunSummary,
    TestCase,
    Verdict,
)
from .pipeline import Pipeline

__version__ = "1.0.0"

__all__ = [
    "Pipeline",
    "Settings",
    "TargetConfig",
    "TestCase",
    "Chunk",
    "GradedResult",
    "RunSummary",
    "QuestionType",
    "Verdict",
    "__version__",
]
