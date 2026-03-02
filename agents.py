"""Agent implementations for multi-agent long-novel generation with continuity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunker import TextChunk
from database import DatabaseManager


@dataclass(slots=True)
class StoryMemory:
    """Cross-chunk memory used by agents to keep narrative coherence."""

    style_guide: str = ""
    known_characters: dict[str, dict[str, str]] = field(default_factory=dict)
    chunk_summaries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    chunk_index: int
    draft: str
    critique: str
    revised: str


class StyleAgent:
    """Converts user style preference into a normalized writing guide."""

    def create_style_guide(self, user_style: str = "") -> str:
        base = (
            "使用中文第三人称叙事，长篇连载节奏，章节结尾保留悬念；"
            "描写兼顾动作、心理与环境，段落长度有变化。"
        )
        custom = user_style.strip()
        if not custom:
            return base
        if len(custom) < 20:
            return f"{base} 用户补充风格：{custom}（其余由系统补全）。"
        return f"{base} 用户详细风格：{custom}"


class CharacterAgent:
    """Tracks characters from DB and current chunk to maintain role consistency."""

    def __init__(self, db: DatabaseManager):
        characters = db.fetch_characters()
        self.character_rows = characters
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}

    def update_memory(self, chunk: TextChunk, memory: StoryMemory) -> list[str]:
        notes: list[str] = []
        for row in self.character_rows:
            original = row["original_name"]
            adapted = row["xianxia_name"]
            if re.search(rf"\b{re.escape(original)}\b", chunk.text):
                memory.known_characters[adapted] = row
                notes.append(f"{adapted}({row['sect']}, {row['cultivation_level']}, {row['status']})")
        return notes


class AdaptationAgent:
    """Rule-based generator that applies term mapping and story memory."""

    def __init__(self, db: DatabaseManager):
        world_map = db.fetch_world_map()
        characters = db.fetch_characters()
        self.term_map = {row["original_term"]: row["xianxia_term"] for row in world_map}
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}

    def generate(self, chunk: TextChunk, memory: StoryMemory, role_notes: list[str]) -> str:
        text = chunk.text
        for original, replaced in sorted(self.name_map.items(), key=lambda i: len(i[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(original)}\b", replaced, text)
        for original, replaced in sorted(self.term_map.items(), key=lambda i: len(i[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(original)}\b", replaced, text)

        continuity = "；".join(role_notes) if role_notes else "延续上一章叙事焦点"
        prefix = (
            f"[Chapter Fragment {chunk.index}]\n"
            f"[Style Guide] {memory.style_guide}\n"
            f"[Role Continuity] {continuity}\n"
        )
        if chunk.context:
            prefix += f"[Context Window] {chunk.context}\n"
        return f"{prefix}{text}"


class ContinuityAgent:
    """Produces compact memory summary for each generated fragment."""

    def summarize(self, revised_text: str) -> str:
        clean = revised_text.replace("\n", " ").strip()
        if len(clean) <= 100:
            return clean
        return clean[:100] + "..."


class CriticAgent:
    """Deterministic checks to enforce style and continuity markers."""

    def evaluate(self, draft: str) -> str:
        issues: list[str] = []
        required_markers = ("[Chapter Fragment", "[Style Guide]", "[Role Continuity]")
        for marker in required_markers:
            if marker not in draft:
                issues.append(f"Missing marker: {marker}")
        if len(draft.split()) < 30:
            issues.append("Draft may be too short; consider richer narration.")
        if not issues:
            return "PASS: no critical issues found."
        return "NEEDS_REVISION: " + " ".join(issues)


class RevisionAgent:
    """Applies deterministic revisions based on critique tags."""

    def revise(self, draft: str, critique: str, memory: StoryMemory) -> str:
        if critique.startswith("PASS"):
            return draft

        revised = draft
        if "Missing marker: [Style Guide]" in critique:
            revised = f"[Style Guide] {memory.style_guide}\n" + revised
        if "Missing marker: [Role Continuity]" in critique:
            revised = "[Role Continuity] 延续人物目标与关系\n" + revised
        if "Missing marker: [Chapter Fragment" in critique and "[Chapter Fragment" not in revised:
            revised = "[Chapter Fragment]\n" + revised
        if "richer narration" in critique:
            revised += "\n灵气翻涌，旧誓与新局在夜色中交错，下一步选择将改写众人命途。"
        return revised
