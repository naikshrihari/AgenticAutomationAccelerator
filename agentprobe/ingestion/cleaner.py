"""Strip common extraction artifacts from loaded document text.

Loaders — especially PDF ones — leave behind hyphenation across line breaks,
repeated page headers/footers, page-number lines, and runs of whitespace. The
cleaner removes these so that chunks and, downstream, generated questions are
not polluted by layout noise.
"""

from __future__ import annotations

import re
from collections import Counter

_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", re.I)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Return a de-noised copy of ``text``."""
    # Re-join words split by a hyphen at a line break.
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = _drop_repeated_running_heads(lines)
    kept = [ln for ln in lines if not _PAGE_NUMBER.match(ln)]

    cleaned = "\n".join(kept)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    cleaned = _MULTINEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()


def _drop_repeated_running_heads(lines: list[str], threshold: int = 3) -> list[str]:
    """Drop short lines that repeat many times — typical headers/footers."""
    counts = Counter(ln.strip() for ln in lines if 0 < len(ln.strip()) <= 80)
    boilerplate = {ln for ln, n in counts.items() if n >= threshold}
    if not boilerplate:
        return lines
    return [ln for ln in lines if ln.strip() not in boilerplate]
