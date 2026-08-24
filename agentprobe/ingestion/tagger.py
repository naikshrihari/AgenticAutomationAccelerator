"""Metadata tagger.

Records document, section, audience, and language on each chunk. Audience and
language are inferred with cheap heuristics here; a deployment can swap in a
classifier or an LLM pass without changing the chunk contract.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..models import Chunk

# Small stop-word-free signal sets; enough to separate en from es reliably
# on policy text without pulling in a language-detection dependency.
_SPANISH_MARKERS = re.compile(
    r"\b(el|la|los|las|de|para|con|una|que|empleado|política|días|trabajo)\b", re.I
)
_AUDIENCE_HINTS = {
    "manager": re.compile(r"\b(manager|supervisor|approval|approver)\b", re.I),
    "employee": re.compile(r"\b(employee|staff|team member|you may|you are entitled)\b", re.I),
    "hr": re.compile(r"\b(human resources|hr team|payroll|benefits administration)\b", re.I),
}


def _detect_language(text: str) -> str:
    return "es" if len(_SPANISH_MARKERS.findall(text)) >= 3 else "en"


def _detect_audience(text: str, section: str) -> str:
    haystack = f"{section}\n{text}"
    for audience, pattern in _AUDIENCE_HINTS.items():
        if pattern.search(haystack):
            return audience
    return "general"


def tag_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Populate audience and language on each chunk in place and return them."""
    tagged: list[Chunk] = []
    for chunk in chunks:
        chunk.language = _detect_language(chunk.text)
        chunk.audience = _detect_audience(chunk.text, chunk.section)
        tagged.append(chunk)
    return tagged
