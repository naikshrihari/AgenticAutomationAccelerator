"""Command-line interface for AgentProbe.

Three subcommands mirror the pipeline's two halves plus a convenience 'all':

    agentprobe generate --docs ./documents [--seeds seeds.jsonl]
    agentprobe evaluate --target config/oracle_fusion.yaml
    agentprobe all      --docs ./documents --target config/oracle_fusion.yaml

Global model/endpoint settings come from environment variables (see Settings);
per-target behaviour comes from the YAML file. This is the same hook a CI job
calls to run the suite on every prompt or node change.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import Settings, TargetConfig
from .golden import GoldenSetStore
from .models import TestCase
from .pipeline import Pipeline


def _load_seeds(path: Optional[str]) -> list[TestCase]:
    if not path:
        return []
    seeds: list[TestCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            seeds.append(TestCase.model_validate_json(line))
    return seeds


def cmd_generate(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    cases = pipeline.build_golden_set(
        args.docs,
        seed_cases=_load_seeds(args.seeds),
        dedupe=not args.no_dedupe,
        self_consistency=not args.no_self_consistency,
    )
    print(f"Generated golden set with {len(cases)} approved cases -> {pipeline.settings.golden_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    target = TargetConfig.from_yaml(args.target)
    cases = None
    if args.golden:
        cases = GoldenSetStore(pipeline.settings.golden_dir).load(args.golden)
    summary = pipeline.evaluate(target, cases=cases, compare_baseline=not args.no_baseline)
    _print_summary(summary, pipeline.settings)
    return 1 if args.fail_under and summary.pass_rate < args.fail_under else 0


def cmd_all(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    cases = pipeline.build_golden_set(args.docs, seed_cases=_load_seeds(args.seeds))
    target = TargetConfig.from_yaml(args.target)
    summary = pipeline.evaluate(target, cases=cases)
    _print_summary(summary, pipeline.settings)
    return 1 if args.fail_under and summary.pass_rate < args.fail_under else 0


def _print_summary(summary, settings: Settings) -> None:
    print(json.dumps(
        {
            "run_id": summary.run_id,
            "target": summary.target,
            "pass_rate": round(summary.pass_rate, 4),
            "passed": summary.passed,
            "partial": summary.partial,
            "failed": summary.failed,
            "errors": summary.errors,
            "total": summary.total,
            "report_dir": str(settings.reports_dir / summary.run_id),
        },
        indent=2,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentprobe", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Build the golden set from documents")
    g.add_argument("--docs", required=True, help="Documents folder")
    g.add_argument("--seeds", help="Optional JSONL of curated seed cases")
    g.add_argument("--no-dedupe", action="store_true")
    g.add_argument("--no-self-consistency", action="store_true")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="Run the golden set against a target")
    e.add_argument("--target", required=True, help="Target YAML config")
    e.add_argument("--golden", help="Specific golden JSONL (defaults to latest)")
    e.add_argument("--no-baseline", action="store_true", help="Skip regression diff")
    e.add_argument("--fail-under", type=float, help="Exit non-zero if pass rate below this")
    e.set_defaults(func=cmd_evaluate)

    a = sub.add_parser("all", help="Generate then evaluate in one go")
    a.add_argument("--docs", required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--seeds")
    a.add_argument("--fail-under", type=float)
    a.set_defaults(func=cmd_all)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
