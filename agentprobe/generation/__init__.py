"""Generation: turn each chunk into graded, validated, de-duplicated cases."""

from .generator import CaseGenerator
from .self_consistency import SelfConsistencyChecker
from .dedup import Deduplicator, SemanticIndex

__all__ = ["CaseGenerator", "SelfConsistencyChecker", "Deduplicator", "SemanticIndex"]
