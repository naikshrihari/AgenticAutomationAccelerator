"""Failure root-cause analysis.

Turns a set of failed cases into a small number of explained failure clusters,
each with a likely cause and a concrete suggested fix — so the report says not
just *what* failed but *why*, and *what to do about it*.
"""

from .root_cause import RootCauseAnalyzer, RootCauseReport, FailureCluster

__all__ = ["RootCauseAnalyzer", "RootCauseReport", "FailureCluster"]
