"""Input cleaning utilities before chunking/adaptation."""

from __future__ import annotations

import re


def clean_source_novel(text: str) -> str:
    """Remove common front-matter/noise blocks from public-domain dumps.

    Keeps the narrative body while dropping license headers, TOC, and page marks.
    """

    cleaned = text.replace("\ufeff", "").replace("\r\n", "\n")

    # Prefer content after Gutenberg START marker when present.
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    if start_marker in cleaned:
        cleaned = cleaned.split(start_marker, 1)[1]

    # Trim everything before first chapter-like marker if available.
    chapter_match = re.search(r"(?im)\bchapter\s*1\b[\.:：\- ]", cleaned)
    if chapter_match:
        cleaned = cleaned[chapter_match.start() :]

    # Drop obvious license/update/credits lines and page-number artifacts.
    drop_line_patterns = [
        r"(?i)^\s*title:\s*",
        r"(?i)^\s*author:\s*",
        r"(?i)^\s*release\s+date:\s*",
        r"(?i)^\s*most\s+recently\s+updated:\s*",
        r"(?i)^\s*language:\s*",
        r"(?i)^\s*credits:\s*",
        r"(?i)^\s*project\s+gutenberg\s*",
        r"(?i)^\s*www\.gutenberg\.org",
        r"(?i)^\s*contents\s*$",
        r"^\s*\d{3,5}[a-z]?\s*$",
    ]

    kept_lines: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if any(re.search(pat, stripped) for pat in drop_line_patterns):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)

    # Collapse excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
