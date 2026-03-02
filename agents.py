"""Agent implementations for multi-agent long-novel generation with continuity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chunker import TextChunk
from database import DatabaseManager
from llm import LLMRouter


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

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def create_style_guide(self, user_style: str = "") -> str:
        base = (
            "使用中文第三人称叙事，长篇连载节奏，章节结尾保留悬念；"
            "描写兼顾动作、心理与环境，段落长度有变化。"
        )
        custom = user_style.strip()
        if not custom:
            fallback = base
        elif len(custom) < 20:
            fallback = f"{base} 用户补充风格：{custom}（其余由系统补全）。"
        else:
            fallback = f"{base} 用户详细风格：{custom}"

        return self.llm.fast_chat(
            system_prompt="你是网文改编编辑，请把输入风格整理为可执行写作规则，控制在120字以内。",
            user_prompt=f"用户风格需求：{custom or '未提供'}\n请输出：叙事视角、节奏、段落风格、章节收束方式。",
            fallback=fallback,
        )


class CharacterAgent:
    """Tracks characters from DB and current chunk to maintain role consistency."""

    def __init__(self, db: DatabaseManager, llm: LLMRouter):
        characters = db.fetch_characters()
        self.character_rows = characters
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}
        self.llm = llm

    def update_memory(self, chunk: TextChunk, memory: StoryMemory) -> list[str]:
        notes: list[str] = []
        for row in self.character_rows:
            original = row["original_name"]
            adapted = row["xianxia_name"]
            if re.search(rf"\b{re.escape(original)}\b", chunk.text):
                memory.known_characters[adapted] = row
                notes.append(f"{adapted}({row['sect']}, {row['cultivation_level']}, {row['status']})")

        fallback = notes if notes else ["延续上一章角色目标与关系张力"]
        role_card = self.llm.fast_chat(
            system_prompt="你是小说角色统筹，请根据人物库与当前片段生成一条简短角色卡更新。",
            user_prompt=(
                f"人物库：{self.character_rows}\n"
                f"当前片段：{chunk.text[:1000]}\n"
                f"已知角色记忆：{list(memory.known_characters.keys())}\n"
                "输出1-2条，每条一句，强调动机/关系/风险。"
            ),
            fallback="；".join(fallback),
        )
        return [line.strip("-• ") for line in role_card.splitlines() if line.strip()] or fallback


class AdaptationAgent:
    """Rule-based generator that applies term mapping and story memory."""

    def __init__(self, db: DatabaseManager, llm: LLMRouter):
        world_map = db.fetch_world_map()
        characters = db.fetch_characters()
        self.term_map = {row["original_term"]: row["xianxia_term"] for row in world_map}
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}
        self.llm = llm

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
        fallback = f"{prefix}{text}"
        return self.llm.long_chat(
            system_prompt=(
                "你是长篇网文改编作者。请根据角色卡、上下文和风格，将输入片段改写为更具网文张力的章节片段。"
                "保留专有名词映射，不要丢失关键信息。"
            ),
            user_prompt=(
                f"{prefix}"
                f"[Mapped Source]\n{text}\n"
                "要求：输出中文；有明显情节推进；结尾保留悬念。"
            ),
            fallback=fallback,
        )


class ContinuityAgent:
    """Produces compact memory summary for each generated fragment."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def summarize(self, revised_text: str) -> str:
        clean = revised_text.replace("\n", " ").strip()
        fallback = clean if len(clean) <= 100 else clean[:100] + "..."
        return self.llm.fast_chat(
            system_prompt="你是剧情连续性助手，请把片段提炼为下一段可用的短记忆。",
            user_prompt=f"请在80字内总结剧情推进、角色状态变化和悬念：\n{clean[:2000]}",
            fallback=fallback,
        )


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
