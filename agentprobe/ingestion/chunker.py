"""Section-aware chunker.

Splitting on headings (rather than a blind sliding window) is what lets every
generated question cite the exact section it came from. Headings are detected
from Markdown ``#`` markers (emitted by the DOCX/Markdown loaders) and from
short, title-like lines. Within a section, long text is further split on
paragraph boundaries to keep chunks within a target size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Chunk
from .loaders import LoadedDocument

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class _Section:
    heading_path: str
    text: str


def _approx_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token. Good enough for sizing.
    return max(1, len(text) // 4)


def _is_title_like(line: str) -> bool:
    """Heuristic for headings in documents that carry no explicit markup."""
    s = line.strip()
    if not (0 < len(s) <= 80):
        return False
    if s.endswith((".", ":", ",", ";")):
        return False
    words = s.split()
    if len(words) > 12:
        return False
    # Numbered headings ("3.1 Ingestion") or Title Case lines.
    if re.match(r"^\d+(\.\d+)*\s+\S", s):
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return len(words) >= 1 and caps / len(words) >= 0.6


def _split_sections(text: str) -> list[_Section]:
    """Break document text into sections, tracking a nested heading path."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)
    buf: list[str] = []

    def flush() -> None:
        if buf and any(ln.strip() for ln in buf):
            path = " > ".join(h for _, h in stack)
            sections.append(_Section(heading_path=path, text="\n".join(buf).strip()))
        buf.clear()

    for line in text.splitlines():
        md = _MD_HEADING.match(line)
        if md:
            flush()
            level, heading = len(md.group(1)), md.group(2).strip()
        elif _is_title_like(line):
            flush()
            level, heading = 2, line.strip()
        else:
            buf.append(line)
            continue
        # Maintain the heading stack so nested sections carry their parents.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
    flush()

    if not sections:  # no headings found at all
        sections.append(_Section(heading_path="", text=text.strip()))
    return sections


def _split_by_size(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split a section's text into size-bounded pieces on paragraph breaks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in paras:
        ptok = _approx_tokens(para)
        if size + ptok > max_tokens and current:
            pieces.append("\n\n".join(current))
            # Carry a little overlap for cross-paragraph context.
            if overlap and current:
                current = current[-1:]
                size = _approx_tokens(current[0])
            else:
                current, size = [], 0
        current.append(para)
        size += ptok
    if current:
        pieces.append("\n\n".join(current))
    return pieces or [text]


def chunk_document(
    doc: LoadedDocument,
    *,
    max_tokens: int = 400,
    overlap: int = 1,
) -> list[Chunk]:
    """Turn one loaded document into a list of section-aware chunks."""
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(_split_sections(doc.text)):
        for p_idx, piece in enumerate(_split_by_size(section.text, max_tokens, overlap)):
            chunk_id = f"{doc.name}::{s_idx:03d}::{p_idx:02d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document=doc.name,
                    document_version=doc.version,
                    section=section.heading_path,
                    text=piece,
                    token_estimate=_approx_tokens(piece),
                )
            )
    return chunks
