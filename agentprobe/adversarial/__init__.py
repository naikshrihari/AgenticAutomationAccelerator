"""Adversarial / red-team test generation.

Produces safety and compliance probes — PII leakage, prompt injection, jailbreak,
unauthorized actions, and bias — where a *correct* agent refuses, declines, or
escalates. Especially important for regulated HR / gaming data.
"""

from .red_team import RedTeamGenerator, ATTACK_CATEGORIES

__all__ = ["RedTeamGenerator", "ATTACK_CATEGORIES"]
