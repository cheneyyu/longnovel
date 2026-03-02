"""Text chunking utilities for long novel processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import DEFAULT_CHUNK_TOKEN_LIMIT, SLIDING_WINDOW_WORDS


@dataclass(slots=True)
class TextChunk:
    """One chunk of source text plus minimal context for regeneration."""

    index: int
    text: str
    context: str = ""


def _estimate_tokens(text: str) -> int:
    """Rough token estimate using word count fallback.

    For an English-heavy source this approximation is generally acceptable for
    chunk sizing in a local/offline pre-processing step.
    """

    words = text.split()
    if not words:
        return 0
    return max(1, int(len(words) * 1.3))


def chunk_text(
    text: str,
    token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT,
    sliding_window_words: int = SLIDING_WINDOW_WORDS,
) -> list[TextChunk]:
    """Split text into chunks by sentence while preserving lightweight context."""

    stripped = text.strip()
    if not stripped:
        return []

    # lightweight sentence boundary strategy without external deps
    normalized = stripped.replace("\r\n", "\n").replace("\n", " ")
    sentences: list[str] = []
    current = []
    for char in normalized:
        current.append(char)
        if char in ".!?。！？":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)

    chunks: list[TextChunk] = []
    cursor_words: list[str] = []
    chunk_words: list[str] = []

    def flush_chunk() -> None:
        nonlocal chunk_words, cursor_words
        if not chunk_words:
            return
        chunk_text_value = " ".join(chunk_words).strip()
        context = " ".join(cursor_words[-sliding_window_words:]).strip()
        chunks.append(TextChunk(index=len(chunks) + 1, text=chunk_text_value, context=context))
        cursor_words.extend(chunk_words)
        chunk_words = []

    for sentence in sentences:
        sentence_words = sentence.split()
        tentative = " ".join(chunk_words + sentence_words)
        if chunk_words and _estimate_tokens(tentative) > token_limit:
            flush_chunk()
        chunk_words.extend(sentence_words)

    flush_chunk()
    return chunks


def iter_chunks(
    paragraphs: Iterable[str],
    token_limit: int = DEFAULT_CHUNK_TOKEN_LIMIT,
    sliding_window_words: int = SLIDING_WINDOW_WORDS,
) -> list[TextChunk]:
    """Chunk multiple paragraph blocks in order."""

    merged = "\n\n".join(p.strip() for p in paragraphs if p and p.strip())
    return chunk_text(merged, token_limit=token_limit, sliding_window_words=sliding_window_words)
