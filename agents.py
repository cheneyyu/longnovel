"""Agent implementations for adaptation, critique, and revision passes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chunker import TextChunk
from database import DatabaseManager


@dataclass(slots=True)
class GenerationResult:
    chunk_index: int
    draft: str
    critique: str
    revised: str


class AdaptationAgent:
    """Rule-based baseline adapter.

    This keeps the project runnable even without an API key and can later be
    replaced by an LLM-backed implementation.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        world_map = db.fetch_world_map()
        characters = db.fetch_characters()
        self.term_map = {row["original_term"]: row["xianxia_term"] for row in world_map}
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}

    def generate(self, chunk: TextChunk) -> str:
        text = chunk.text
        for original, replaced in sorted(self.name_map.items(), key=lambda i: len(i[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(original)}\b", replaced, text)
        for original, replaced in sorted(self.term_map.items(), key=lambda i: len(i[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(original)}\b", replaced, text)

        prefix = f"[Chapter Fragment {chunk.index}]"
        return f"{prefix}\n{text}"


class CriticAgent:
    """Simple deterministic checks that mimic quality gating."""

    def evaluate(self, chunk: TextChunk, draft: str) -> str:
        issues: list[str] = []
        if len(draft.split()) < 20:
            issues.append("Draft may be too short; consider richer narration.")
        if "[Chapter Fragment" not in draft:
            issues.append("Missing section marker.")
        if not issues:
            return "PASS: no critical issues found."
        return "NEEDS_REVISION: " + " ".join(issues)


class RevisionAgent:
    """Applies deterministic revisions based on critique tags."""

    def revise(self, draft: str, critique: str) -> str:
        if critique.startswith("PASS"):
            return draft

        revised = draft
        if "richer narration" in critique:
            revised += "\nHeaven and earth qi churned in silence, foreshadowing upheaval."
        if "Missing section marker" in critique and "[Chapter Fragment" not in revised:
            revised = "[Chapter Fragment]\n" + revised
        return revised
