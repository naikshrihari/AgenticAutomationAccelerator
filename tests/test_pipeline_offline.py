"""Offline tests for the deterministic parts of the pipeline.

These exercise ingestion, the deterministic grading checks, the verdict
resolver, the regression differ, and the results store — everything that does
NOT require a live Ollama server. LLM-backed stages (generation, judge) are
covered by integration tests run against a real Ollama instance.
"""

from __future__ import annotations

from pathlib import Path

from agentprobe.grading.fact_coverage import check_fact_coverage
from agentprobe.grading.groundedness import check_groundedness
from agentprobe.grading.resolver import VerdictResolver
from agentprobe.ingestion import chunk_document, clean_text, tag_chunks
from agentprobe.ingestion.loaders import load_document
from agentprobe.models import (
    AgentResponse,
    GradedResult,
    JudgeScore,
    QuestionType,
    TestCase,
    Verdict,
)
from agentprobe.reporting.differ import RegressionDiffer

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "documents"


def test_ingestion_is_section_aware():
    doc = load_document(EXAMPLES / "leave_policy.md")
    doc.text = clean_text(doc.text)
    chunks = tag_chunks(chunk_document(doc))
    assert chunks, "expected at least one chunk"
    sections = {c.section for c in chunks}
    # Headings should be preserved so answers trace back to a section.
    assert any("Annual Leave" in s for s in sections)
    assert any("Sick Leave" in s for s in sections)
    # Citations are non-empty and reference the source document.
    assert all(c.document == "leave_policy.md" for c in chunks)


def test_cleaner_strips_page_numbers_and_hyphenation():
    dirty = "Annual leave pol-\nicy applies.\n\n12\n\nSick leave applies."
    cleaned = clean_text(dirty)
    assert "policy" in cleaned  # hyphenation re-joined
    assert "\n12\n" not in cleaned  # bare page number dropped


def test_fact_coverage_counts_required_facts():
    cov = check_fact_coverage(
        ["20 days annual leave", "carry over 5 days"],
        "Employees get 20 days of annual leave and may carry over 5 days.",
    )
    assert cov.missing == []
    assert cov.ratio == 1.0

    cov2 = check_fact_coverage(["medical certificate after 3 days"], "You get sick leave.")
    assert cov2.ratio == 0.0


def test_groundedness_via_citations():
    case = TestCase(
        case_id="c1", question="q", expected_answer="a",
        citation="leave_policy.md § Sick Leave",
    )
    grounded = AgentResponse(case_id="c1", answer="See sick leave.", cited_sources=["leave_policy.md § Sick Leave"])
    assert check_groundedness(case, grounded) is True
    ungrounded = AgentResponse(case_id="c1", answer="Unrelated.", cited_sources=["other.md § Payroll"])
    assert check_groundedness(case, ungrounded) is False


def test_resolver_error_short_circuits(monkeypatch):
    # A resolver whose judge is never called, because the response is an error.
    resolver = VerdictResolver.__new__(VerdictResolver)
    resolver.thresholds = VerdictResolver().thresholds  # default thresholds
    case = TestCase(case_id="c1", question="q", expected_answer="a")
    err = AgentResponse(case_id="c1", ok=False, error_kind="timeout", error_detail="boom")
    result = resolver.grade(case, err)
    assert result.verdict is Verdict.ERROR
    assert result.error_kind == "timeout"


def test_regression_differ_flags_flips():
    def graded(cid, verdict, score):
        return GradedResult(
            case_id=cid, question="q", expected_answer="a", agent_answer="b",
            verdict=verdict, score=score, judge=JudgeScore(),
            question_type=QuestionType.FACTUAL,
        )

    baseline = [graded("a", Verdict.PASS, 0.9), graded("b", Verdict.FAIL, 0.2)]
    current = [graded("a", Verdict.FAIL, 0.3), graded("b", Verdict.PASS, 0.85)]
    report = RegressionDiffer().diff(baseline, current)
    assert report.has_regressions
    assert report.regressions[0].case_id == "a"
    assert report.fixes[0].case_id == "b"


def test_requirements_are_optional_and_injected():
    """The generator injects BRD context only when it is provided."""
    from agentprobe.generation.generator import CaseGenerator
    from agentprobe.models import Chunk, QuestionType

    chunk = Chunk(chunk_id="c1", document="d.md", section="Leave", text="20 days annual leave.")

    class _CapturingClient:
        def __init__(self):
            self.messages = None

        def structured(self, messages, schema, **kw):
            self.messages = messages
            from agentprobe.generation.generator import _GeneratedCase

            return _GeneratedCase(question="q", expected_answer="a", required_facts=["x"])

    # Without requirements: no BRD block in the prompt.
    gen = CaseGenerator(client=_CapturingClient(), type_mix=[QuestionType.FACTUAL])
    gen.settings = type("S", (), {"generation_model": "m", "generation_temperature": 0.0})()
    gen.generate_for_chunk(chunk)
    assert "BUSINESS REQUIREMENTS" not in gen.client.messages[-1]["content"]

    # With requirements: the BRD text is present.
    gen2 = CaseGenerator(
        client=_CapturingClient(),
        type_mix=[QuestionType.FACTUAL],
        requirements="Agent must never disclose salaries.",
    )
    gen2.settings = type("S", (), {"generation_model": "m", "generation_temperature": 0.0})()
    gen2.generate_for_chunk(chunk)
    prompt = gen2.client.messages[-1]["content"]
    assert "BUSINESS REQUIREMENTS" in prompt
    assert "never disclose salaries" in prompt


def test_generic_connector_parses_bare_string_response():
    """An agent that returns the answer as a plain JSON string is handled."""
    from agentprobe.config import TargetConfig
    from agentprobe.execution.connectors.base import BaseConnector

    cfg = TargetConfig(name="x", connector="generic_rest", base_url="http://h",
                       endpoint="/api/ask", request_template={"message": "{question}"})
    cfg.response.answer_path = ""  # whole body is the answer
    conn = BaseConnector(cfg)
    resp = conn._parse("c1", "The leave policy allows 20 days.", 12.0)
    assert resp.answer == "The leave policy allows 20 days."
    assert resp.ok is True


def test_results_store_roundtrip(tmp_path):
    from agentprobe.models import RunSummary
    from agentprobe.reporting.store import ResultsStore

    store = ResultsStore(tmp_path / "r.db")
    summary = RunSummary(run_id="run-1", target="t", total=1, passed=1)
    result = GradedResult(
        case_id="c1", question="q", expected_answer="a", agent_answer="b",
        verdict=Verdict.PASS, score=0.9, question_type=QuestionType.FACTUAL,
    )
    store.save_run(summary, [result])
    loaded = store.load_results("run-1")
    assert len(loaded) == 1 and loaded[0].verdict is Verdict.PASS
    assert store.latest_run_id("t") == "run-1"
    store.close()
