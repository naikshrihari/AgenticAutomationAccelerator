"""Reporting: store results, diff against a baseline, build the report."""

from .store import ResultsStore
from .differ import RegressionDiffer, RegressionReport
from .report import ReportBuilder

__all__ = ["ResultsStore", "RegressionDiffer", "RegressionReport", "ReportBuilder"]
