"""End-to-end orchestration.

Wires the layers together into the flow described in the architecture:

    documents + target config
        -> ingestion (load, clean, chunk, tag)
        -> generation on Ollama (generate, self-consistency, dedup)
        -> golden set (review gate, versioned JSONL)
        -> execution (async runner via the right connector)
        -> grading on Ollama (fact coverage, judge, groundedness, resolver)
        -> reporting (store, regression diff, HTML/CSV/native export)

The same flow runs against the next platform by changing only the target config
and connector — nothing in this module is platform-specific.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings, TargetConfig
from .golden import GoldenSetStore, ReviewGate
from .execution import ExecutionEngine
from .generation import CaseGenerator, Deduplicator, SelfConsistencyChecker
from .grading import VerdictResolver
from .ingestion import chunk_document, clean_text, load_folder, tag_chunks
from .llm import EmbeddingModel, OllamaClient
from .models import Chunk, GradedResult, RunSummary, TestCase, Verdict
from .reporting import RegressionDiffer, ReportBuilder, ResultsStore

logger = logging.getLogger(__name__)


def _quiet_windows_event_loop() -> None:
    """Use the Selector event loop on Windows to avoid noisy, harmless
    ConnectionResetError (WinError 10054) logs that the Proactor loop emits
    when httpx connections are torn down at the end of a run."""
    import sys

    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:  # noqa: BLE001 - best effort; never fail the run over this
            pass


class Pipeline:
    """High-level façade over the whole accelerator."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.from_env()

    # ------------------------------------------------------------------ #
    # Generation side: documents -> golden set
    # ------------------------------------------------------------------ #
    def build_golden_set(
        self,
        docs_folder: str | Path,
        *,
        seed_cases: Optional[list[TestCase]] = None,
        review_gate: Optional[ReviewGate] = None,
        dedupe: bool = True,
        self_consistency: bool = True,
        limit: Optional[int] = None,
        max_workers: int = 1,
        type_mix: Optional[list] = None,
        requirements: Optional[str] = None,
    ) -> list[TestCase]:
        """Ingest documents and generate a validated, versioned golden set.

        ``limit`` caps how many chunks are used (handy for a quick trial run on a
        large document); ``max_workers`` overlaps generation across chunks;
        ``type_mix`` restricts which question types are generated (fewer types =
        fewer model calls = faster). ``requirements`` is optional business-
        requirements text that steers generation toward what the business cares
        about; when omitted, generation proceeds from the documents alone.
        """
        # 1. Ingestion
        chunks = self._ingest(docs_folder)
        logger.info("ingested %d chunks", len(chunks))
        if limit is not None:
            chunks = chunks[:limit]
            logger.info("limited to first %d chunks", len(chunks))

        # 2. Generation
        client = OllamaClient(self.settings, model=self.settings.generation_model)
        gen_kwargs = {
            "client": client,
            "settings": self.settings,
            "max_workers": max_workers,
            "requirements": requirements,
        }
        if type_mix:
            gen_kwargs["type_mix"] = type_mix
        generator = CaseGenerator(**gen_kwargs)
        if requirements:
            logger.info("using business-requirements context (%d chars)", len(requirements))
        cases = generator.generate(chunks)
        logger.info("generated %d raw cases", len(cases))

        # 2a. Self-consistency filter
        if self_consistency:
            checker = SelfConsistencyChecker(settings=self.settings)
            chunks_by_id = {c.chunk_id: c for c in chunks}
            cases = checker.filter(cases, chunks_by_id)
            logger.info("%d cases passed self-consistency", len(cases))

        # 2b. Dedup
        if dedupe:
            cases = Deduplicator(EmbeddingModel(self.settings), self.settings).dedupe(cases)
            logger.info("%d cases after dedup", len(cases))

        # Seed cases are trusted curated adversarial/negative cases — never
        # generated, always kept.
        if seed_cases:
            for s in seed_cases:
                s.is_seed = True
            cases = list(seed_cases) + cases

        # 3. Golden set: review gate + versioned store
        approved = (review_gate or ReviewGate()).run(cases)
        store = GoldenSetStore(self.settings.golden_dir)
        path = store.save_version(approved)
        logger.info("saved golden set (%d cases) -> %s", len(approved), path)
        return approved

    def build_redteam_set(
        self,
        *,
        domain: str = "a company HR policy assistant",
        categories: Optional[list[str]] = None,
        llm_variants_per_category: int = 0,
        save: bool = True,
    ) -> list[TestCase]:
        """Generate an adversarial / red-team suite and (optionally) save it."""
        from .adversarial import RedTeamGenerator

        cases = RedTeamGenerator(settings=self.settings, domain=domain).generate(
            categories=categories, llm_variants_per_category=llm_variants_per_category)
        if save and cases:
            path = GoldenSetStore(self.settings.golden_dir).save_version(cases)
            logger.info("saved red-team set (%d cases) -> %s", len(cases), path)
        return cases

    # ------------------------------------------------------------------ #
    # Execution + grading + reporting side: golden set -> report
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        target: TargetConfig,
        cases: Optional[list[TestCase]] = None,
        *,
        compare_baseline: bool = True,
        analyze_failures: bool = False,
        on_execute=None,
        on_grade=None,
    ) -> RunSummary:
        """Run the golden set against a target and produce a full report.

        ``on_execute(done, total)`` is called as each agent response comes back;
        ``on_grade(done, total, result)`` is called as each case is graded. Both
        are optional and let a UI show live per-record progress.

        When ``analyze_failures`` is set, a root-cause analysis clusters the
        failures and diagnoses each cluster (with a suggested fix) on the local
        model, and the result is included in the HTML report.
        """
        if cases is None:
            cases = GoldenSetStore(self.settings.golden_dir).load_latest()

        run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        summary = RunSummary(run_id=run_id, target=target.name, total=len(cases))
        total = len(cases)

        # 4. Execution (async)
        _quiet_windows_event_loop()
        exec_done = {"n": 0}

        def _on_result() -> None:
            exec_done["n"] += 1
            if on_execute is not None:
                on_execute(exec_done["n"], total)

        responses = asyncio.run(ExecutionEngine(target).run(cases, on_result=_on_result))
        responses_by_id = {r.case_id: r for r in responses}

        # 5. Grading
        resolver = VerdictResolver(thresholds=target.thresholds, settings=self.settings)
        cases_by_id = {c.case_id: c for c in cases}
        results: list[GradedResult] = []
        for case_id, response in responses_by_id.items():
            result = resolver.grade(cases_by_id[case_id], response)
            results.append(result)
            if on_grade is not None:
                on_grade(len(results), total, result)
        _tally(summary, results)
        summary.finished_at = datetime.now(timezone.utc)

        # 6. Reporting: store, diff, render
        store = ResultsStore(self.settings.results_dir / "results.db")
        regression = None
        if compare_baseline:
            baseline_id = store.latest_run_id(target.name)
            if baseline_id:
                regression = RegressionDiffer().diff(store.load_results(baseline_id), results)
        store.save_run(summary, results)
        store.close()

        # Optional: diagnose why cases failed (clusters + suggested fixes).
        root_cause = None
        if analyze_failures:
            from .analysis import RootCauseAnalyzer

            try:
                root_cause = RootCauseAnalyzer(settings=self.settings).analyze(results, cases_by_id)
            except Exception as exc:  # noqa: BLE001 - analysis is best-effort
                logger.warning("root-cause analysis failed: %s", exc)

        builder = ReportBuilder(self.settings.reports_dir / run_id)
        builder.to_html(summary, results, regression, root_cause=root_cause)
        builder.to_csv(results)
        builder.to_native_evaluator(results)
        logger.info(
            "run %s: %.1f%% pass (%d/%d scored)",
            run_id, summary.pass_rate * 100, summary.passed, summary.scored,
        )
        return summary

    # ------------------------------------------------------------------ #
    def _ingest(self, docs_folder: str | Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in load_folder(docs_folder):
            doc.text = clean_text(doc.text)
            chunks.extend(chunk_document(doc))
        return tag_chunks(chunks)


def _tally(summary: RunSummary, results: list[GradedResult]) -> None:
    for r in results:
        if r.verdict is Verdict.PASS:
            summary.passed += 1
        elif r.verdict is Verdict.PARTIAL:
            summary.partial += 1
        elif r.verdict is Verdict.FAIL:
            summary.failed += 1
        elif r.verdict is Verdict.ERROR:
            summary.errors += 1
