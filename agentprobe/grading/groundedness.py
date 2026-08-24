"""Groundedness check.

Confirms the agent used or cited the correct source section. When the target
returns citations, we check whether the expected source is among them. As a
fallback (many agents answer without structured citations), we look for lexical
overlap between the expected citation's section terms and the agent's answer.
This is separate from the judge's groundedness score: it specifically answers
"did it point at the right source?".
"""

from __future__ import annotations

import re

from ..models import AgentResponse, TestCase

_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def check_groundedness(case: TestCase, response: AgentResponse) -> bool:
    """Return True if the response is grounded in the expected source section."""
    expected = case.citation.strip()
    if not expected:
        return True  # nothing to check against

    # 1) Structured citations returned by the agent.
    if response.cited_sources:
        exp_terms = _terms(expected)
        for cited in response.cited_sources:
            cited_terms = _terms(cited)
            if exp_terms and len(exp_terms & cited_terms) / len(exp_terms) >= 0.5:
                return True
        return False

    # 2) Fallback: does the answer mention the section's distinctive terms?
    section = case.citation.split("§")[-1]
    exp_terms = _terms(section)
    if not exp_terms:
        return True
    answer_terms = _terms(response.answer)
    overlap = len(exp_terms & answer_terms) / len(exp_terms)
    return overlap >= 0.4
