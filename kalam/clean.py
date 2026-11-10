"""Cleaning pipeline for raw Wikipedia-style text.

Raw encyclopedia text carries markup noise that a small language model
should not waste capacity on. The pipeline runs, in order:

1. Drop leftover templates (``{{...}}``).
2. Drop reference / bibliography sections entirely.
3. Drop inline citation markers such as ``[1]`` or ``[٢، ٣]``.
4. Drop very short paragraphs (section stubs, captions, lists fragments).
5. Collapse repeated blank lines.
"""

from __future__ import annotations

import re

_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_CITATION = re.compile(r"\[[0-9٠-٩۰-۹,،\s]+\]")

_REFERENCE_HEADINGS = {
    "المراجع",
    "المصادر",
    "انظر ايضا",
    "references",
    "bibliography",
    "see also",
    "notes",
}


def _heading_title(line: str) -> str | None:
    """Return the title if ``line`` looks like a section heading, else None."""
    stripped = line.strip().strip("#=").strip()
    if not stripped or len(stripped) > 60 or len(stripped.split()) > 5:
        return None
    if line.strip().startswith(("#", "=")) or stripped == line.strip():
        return stripped
    return None


def remove_reference_sections(text: str) -> str:
    """Remove sections whose heading names a references-like section."""
    from .normalize import normalize_arabic

    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        title = _heading_title(line)
        if title is not None:
            normalized_title = normalize_arabic(title).strip().lower()
            skipping = normalized_title in _REFERENCE_HEADINGS
            if skipping:
                continue
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def remove_citations(text: str) -> str:
    """Remove inline numeric citation markers."""
    return _CITATION.sub("", text)


def remove_templates(text: str) -> str:
    """Remove leftover ``{{template}}`` markup."""
    return _TEMPLATE.sub("", text)


def drop_short_paragraphs(text: str, *, min_chars: int = 40) -> str:
    """Keep only paragraphs with at least ``min_chars`` characters."""
    paragraphs = re.split(r"\n\s*\n", text)
    kept = [p.strip() for p in paragraphs if len(p.strip()) >= min_chars]
    return "\n\n".join(kept)


def clean_wikipedia_text(text: str, *, min_paragraph_chars: int = 40) -> str:
    """Run the full cleaning pipeline on one raw article text."""
    text = remove_templates(text)
    text = remove_reference_sections(text)
    text = remove_citations(text)
    text = drop_short_paragraphs(text, min_chars=min_paragraph_chars)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
