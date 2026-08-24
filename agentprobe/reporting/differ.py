"""Regression differ.

Compares a run against a baseline run case-by-case and flags cases that flipped
— most importantly regressions (pass/partial → fail) and, for context, fixes
(fail → pass). This is what turns the suite into a guardrail: a prompt or node
change that breaks previously-passing behaviour shows up immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import GradedResult, Verdict

# Ordinal ranking so we can say a verdict got "worse" or "better".
_RANK = {Verdict.FAIL: 0, Verdict.ERROR: 0, Verdict.PARTIAL: 1, Verdict.PASS: 2}


@dataclass
class CaseFlip:
    case_id: str
    question_type: str
    before: Verdict
    after: Verdict
    score_before: float
    score_after: float


@dataclass
class RegressionReport:
    regressions: list[CaseFlip] = field(default_factory=list)  # got worse
    fixes: list[CaseFlip] = field(default_factory=list)  # got better
    unchanged: int = 0
    new_cases: list[str] = field(default_factory=list)
    dropped_cases: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


class RegressionDiffer:
    def diff(self, baseline: list[GradedResult], current: list[GradedResult]) -> RegressionReport:
        base = {r.case_id: r for r in baseline}
        curr = {r.case_id: r for r in current}
        report = RegressionReport()

        for case_id, cur in curr.items():
            if case_id not in base:
                report.new_cases.append(case_id)
                continue
            b = base[case_id]
            if _RANK[cur.verdict] < _RANK[b.verdict]:
                report.regressions.append(_flip(b, cur))
            elif _RANK[cur.verdict] > _RANK[b.verdict]:
                report.fixes.append(_flip(b, cur))
            else:
                report.unchanged += 1

        report.dropped_cases = [cid for cid in base if cid not in curr]
        return report


def _flip(before: GradedResult, after: GradedResult) -> CaseFlip:
    return CaseFlip(
        case_id=after.case_id,
        question_type=after.question_type.value,
        before=before.verdict,
        after=after.verdict,
        score_before=before.score,
        score_after=after.score,
    )
