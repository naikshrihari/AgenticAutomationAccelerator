"""Tests for failure root-cause analysis (clustering + diagnosis)."""

from __future__ import annotations

from agentprobe.analysis.root_cause import RootCauseAnalyzer, _Diagnosis
from agentprobe.models import FactCoverage, GradedResult, JudgeScore, QuestionType, Verdict


def _graded(cid, verdict, question, missing=None):
    return GradedResult(
        case_id=cid, question=question, expected_answer="e", agent_answer="a",
        verdict=verdict, score=0.2, judge=JudgeScore(),
        fact_coverage=FactCoverage(covered=[], missing=missing or []),
        question_type=QuestionType.FACTUAL,
    )


class _FakeEmbedder:
    """Two well-separated clusters based on a keyword in the signature."""

    def embed(self, texts):
        return [[1.0, 0.0] if "vacation" in t.lower() else [0.0, 1.0] for t in texts]


class _FakeClient:
    def structured(self, messages, schema, **kw):
        return _Diagnosis(
            category="wrong_source_retrieved",
            summary="Agent retrieved an adjacent policy section.",
            likely_cause="Retrieval returns neighboring chunks.",
            suggested_fix="Add section titles to the embedding text.",
        )


def test_analyzer_clusters_and_diagnoses():
    results = [
        _graded("c1", Verdict.FAIL, "What is the vacation policy?", ["20 days"]),
        _graded("c2", Verdict.FAIL, "How much vacation do I get?", ["accrual"]),
        _graded("c3", Verdict.FAIL, "What is the smoking policy?"),
        _graded("c4", Verdict.PASS, "unused passing case"),  # excluded
    ]
    analyzer = RootCauseAnalyzer(client=_FakeClient(), embedder=_FakeEmbedder())
    report = analyzer.analyze(results)

    assert report.total_failures == 3          # the PASS is excluded
    assert report.has_findings
    # Two vacation questions cluster together, smoking is its own cluster.
    sizes = sorted(c.size for c in report.clusters)
    assert sizes == [1, 2]
    biggest = report.clusters[0]               # sorted largest-first
    assert biggest.size == 2
    assert biggest.suggested_fix               # a fix was produced


def test_analyzer_no_failures_returns_empty():
    results = [_graded("c1", Verdict.PASS, "q")]
    report = RootCauseAnalyzer(client=_FakeClient(), embedder=_FakeEmbedder()).analyze(results)
    assert report.total_failures == 0
    assert not report.has_findings
