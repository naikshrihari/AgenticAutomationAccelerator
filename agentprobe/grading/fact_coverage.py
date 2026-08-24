"""Deterministic fact-coverage checklist.

Before any LLM judgement, we check — with no model in the loop — whether each
required fact from the expected answer actually appears in the agent's response.
This is cheap, reproducible, and catches the common failure of a fluent answer
that omits a key figure or condition. Matching is token-overlap based so it
tolerates paraphrase and punctuation while still requiring the substance.
"""

from __future__ import annotations

import re

from ..models import FactCoverage

_WORD = re.compile(r"[a-z0-9]+")
# A required fact counts as covered when this fraction of its content words
# appear in the response.
_COVERAGE_THRESHOLD = 0.7


def _content_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _fact_present(fact: str, response_words: set[str]) -> bool:
    fact_words = _content_words(fact)
    if not fact_words:
        return True
    hits = sum(1 for w in fact_words if w in response_words)
    return hits / len(fact_words) >= _COVERAGE_THRESHOLD


def check_fact_coverage(required_facts: list[str], response: str) -> FactCoverage:
    """Return which required facts are covered vs missing in ``response``."""
    response_words = set(_content_words(response))
    covered: list[str] = []
    missing: list[str] = []
    for fact in required_facts:
        (covered if _fact_present(fact, response_words) else missing).append(fact)
    return FactCoverage(covered=covered, missing=missing)
