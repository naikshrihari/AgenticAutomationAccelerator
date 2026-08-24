"""Optional human review gate.

Before generated cases enter the golden set, a reviewer may approve, edit, or
reject each one. The gate is deliberately transport-agnostic: it takes a
callback that renders a case and returns a decision, so the same logic backs a
CLI prompt, a notebook widget, or a web queue. When no reviewer is supplied the
gate auto-approves (suitable for CI where the self-consistency pass is the only
guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from ..models import TestCase


class ReviewAction(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


@dataclass
class ReviewDecision:
    """A reviewer's verdict on one candidate case."""

    action: ReviewAction
    edited: Optional[TestCase] = None  # required when action is EDIT


# A reviewer function takes a candidate and returns a decision.
Reviewer = Callable[[TestCase], ReviewDecision]


def auto_approve(case: TestCase) -> ReviewDecision:
    """Default reviewer: accept everything that reached the gate."""
    return ReviewDecision(action=ReviewAction.APPROVE)


class ReviewGate:
    """Runs candidate cases past a reviewer and returns the approved set."""

    def __init__(self, reviewer: Optional[Reviewer] = None):
        self.reviewer = reviewer or auto_approve

    def run(self, candidates: list[TestCase]) -> list[TestCase]:
        approved: list[TestCase] = []
        for candidate in candidates:
            decision = self.reviewer(candidate)
            if decision.action is ReviewAction.REJECT:
                continue
            case = decision.edited if decision.action is ReviewAction.EDIT and decision.edited else candidate
            case.approved = True
            approved.append(case)
        return approved
