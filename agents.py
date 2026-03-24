"""Agent implementations for multi-agent long-novel generation with continuity."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chunker import TextChunk
from database import DatabaseManager
from llm import LLMRouter


@dataclass(slots=True)
class StoryMemory:
    """Cross-chunk memory used by agents to keep narrative coherence."""

    style_guide: str = ""
    project_brief: str = ""
    story_bible: str = ""
    arc_plan: str = ""
    known_characters: dict[str, dict[str, str]] = field(default_factory=dict)
    chunk_summaries: list[str] = field(default_factory=list)
    chapter_ledger: list[dict[str, Any]] = field(default_factory=list)
    foreshadow_ledger: list[dict[str, str]] = field(default_factory=list)


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
        base = "中文第三人称、仙侠网文节奏、冲突递进、结尾留钩子、人物关系持续升级。"
        custom = user_style.strip()
        fallback = f"{base} 用户风格：{custom}" if custom else base
        return self.llm.fast_chat(
            system_prompt="你是网文改编总编。仅输出中文，不要解释。",
            user_prompt=(
                f"将以下需求整理为80字以内写作规则，偏仙侠改编：{custom or '未提供'}。"
                "必须包含：叙事视角、节奏、情绪张力、章节收束方式。"
            ),
            fallback=fallback,
        )


class RequirementAgent:
    """Parses loose user input into a compact project brief."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def build_project_brief(self, user_style: str, source_text: str) -> str:
        fallback = (
            "题材:东方玄幻\n"
            "风格:热血+权谋\n"
            "节奏:中速推进\n"
            "目标:长篇连载，单章有钩子\n"
            "限制:只做结构转译，不贴皮复写"
        )
        return self.llm.fast_chat(
            system_prompt="你是网文项目策划，请输出精炼中文项目规格，不要解释。",
            user_prompt=(
                f"用户风格输入：{user_style or '未提供'}\n"
                f"原文前1200字：{source_text[:1200]}\n"
                "请输出“创作规格书”，格式固定5行：题材/风格/节奏/目标/限制。"
            ),
            fallback=fallback,
        )


class WorldbuildingAgent:
    """Builds concise world rules and power boundaries."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def build_story_bible(self, project_brief: str) -> str:
        fallback = (
            "世界层级:凡域-灵域-圣域\n"
            "修炼体系:炼体-聚气-筑基-金丹-元婴\n"
            "边界规则:跨境最多一阶, 复活需付重大代价\n"
            "核心矛盾:资源争夺与宗门审判并行\n"
            "终局方向:揭示上界收割真相"
        )
        return self.llm.fast_chat(
            system_prompt="你是玄幻世界观架构师，只输出中文要点。",
            user_prompt=(
                f"创作规格书：\n{project_brief}\n"
                "请输出5行“世界观圣经摘要”：世界层级/修炼体系/边界规则/核心矛盾/终局方向。"
            ),
            fallback=fallback,
        )


class StructureAgent:
    """Creates arc-level plan for stable long-form pacing."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def build_arc_plan(self, project_brief: str, story_bible: str) -> str:
        fallback = (
            "卷1 微末崛起: 建立仇恨与目标\n"
            "卷2 宗门立足: 同辈竞争与资源线\n"
            "卷3 边域争锋: 地图扩大与外敌登场\n"
            "卷4 旧史真相: 回收前期伏笔并抬升主线"
        )
        return self.llm.fast_chat(
            system_prompt="你是网文结构规划编辑，只输出中文。",
            user_prompt=(
                f"规格书：\n{project_brief}\n世界观：\n{story_bible}\n"
                "请给出4卷压缩卷纲，每卷一行，格式：卷名: 主冲突 + 卷末钩子方向。"
            ),
            fallback=fallback,
        )


class CharacterAgent:
    """Tracks characters from DB and current chunk to maintain role consistency."""

    def __init__(self, db: DatabaseManager, llm: LLMRouter):
        characters = db.fetch_characters()
        self.character_rows = characters
        self.llm = llm

    def update_memory(self, chunk: TextChunk, memory: StoryMemory) -> list[str]:
        notes: list[str] = []
        for row in self.character_rows:
            original = row["original_name"]
            adapted = row["xianxia_name"]
            if re.search(rf"\b{re.escape(original)}\b", chunk.text):
                memory.known_characters[adapted] = row
                notes.append(f"{adapted}({row['sect']},{row['cultivation_level']},{row['status']})")

        fallback = notes if notes else ["沿用上一段人物目标与冲突关系"]
        role_card = self.llm.fast_chat(
            system_prompt="你是角色统筹编辑。仅输出中文要点，每行一句。",
            user_prompt=(
                f"人物库：{self.character_rows}\n"
                f"当前片段（截断）：{chunk.text[:800]}\n"
                f"已有角色记忆：{list(memory.known_characters.keys())}\n"
                "生成1-2条角色卡更新，强调动机、关系变化、潜在风险。"
            ),
            fallback="\n".join(fallback),
        )
        return [line.strip("-• ") for line in role_card.splitlines() if line.strip()] or fallback


class AdaptationAgent:
    """LLM generator with term mapping and continuity context."""

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

        continuity = "；".join(role_notes) if role_notes else "沿用上一段叙事焦点"
        context_text = chunk.context[:1200] if chunk.context else ""
        fallback_excerpt = text[:900]
        fallback = (
            "【离线草稿】未检测到可用LLM，以下为待改写片段摘要：\n"
            f"{fallback_excerpt}\n"
            "（请配置 OPENAI_API_KEY / OPENAI_BASE_URL / 模型名后自动生成中文仙侠改写）"
        )

        return self.llm.long_chat(
            system_prompt=(
                "你是中文仙侠网文改编作者。"
                "只输出正文，不要输出任何标签、说明、英文、目录、版权声明、页码。"
                "遇到版权前言/目录/元信息时直接忽略，仅保留剧情叙事。"
            ),
            user_prompt=(
                f"写作规则：{memory.style_guide}\n"
                f"项目规格：{memory.project_brief}\n"
                f"世界设定：{memory.story_bible}\n"
                f"卷纲参考：{memory.arc_plan}\n"
                f"角色连续性：{continuity}\n"
                f"上文窗口：{context_text}\n"
                f"待改写片段：{text[:4000]}\n"
                "要求：\n"
                "1) 输出中文仙侠风正文；\n"
                "2) 有明确情节推进和人物冲突；\n"
                "3) 结尾留悬念；\n"
                "4) 不要出现[Chapter Fragment]等标记。"
            ),
            fallback=fallback,
        )

    def generate_recursive(self, memory: StoryMemory, role_notes: list[str], previous_text: str) -> str:
        continuity = "；".join(role_notes) if role_notes else "沿用上一段叙事焦点"
        summary_context = "\n".join(memory.chunk_summaries[-3:]) if memory.chunk_summaries else "暂无前情，直接从冲突起势。"
        seed = previous_text[-1200:].strip()
        fallback = (
            "【离线续写草稿】未检测到可用LLM，以下为续写占位段落：\n"
            "危机并未结束，新的线索把众人推向更深的局中局，真正的敌意才刚刚显形。"
        )

        return self.llm.long_chat(
            system_prompt=(
                "你是中文仙侠网文连载作者。"
                "只输出正文，不要解释，不要编号，不要出现系统标记。"
            ),
            user_prompt=(
                f"写作规则：{memory.style_guide}\n"
                f"项目规格：{memory.project_brief}\n"
                f"世界设定：{memory.story_bible}\n"
                f"卷纲参考：{memory.arc_plan}\n"
                f"角色连续性：{continuity}\n"
                f"最近前情摘要：{summary_context}\n"
                f"上一段正文（截断）：{seed}\n"
                "请在不重复已有内容的前提下，继续推进剧情，新增有效事件与冲突升级，"
                "并以新的悬念收束本段。"
            ),
            fallback=fallback,
        )


class ContinuityAgent:
    """Produces compact memory summary for each generated fragment."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def summarize(self, revised_text: str) -> str:
        clean = revised_text.replace("\n", " ").strip()
        fallback = clean[:100] + "..." if len(clean) > 100 else clean
        return self.llm.fast_chat(
            system_prompt="你是剧情连续性助手，仅输出中文一句话。",
            user_prompt=f"请用60字内总结本段的事件推进、角色变化和未解悬念：{clean[:1500]}",
            fallback=fallback,
        )


class MemoryCompressionAgent:
    """Writes structured chapter-state ledger entries."""

    def __init__(self, llm: LLMRouter):
        self.llm = llm

    def build_ledger_entry(self, chapter_index: int, text: str, role_notes: list[str]) -> dict[str, Any]:
        fallback = (
            f"章节{chapter_index}：事件推进；角色关系变化；埋下后续悬念。"
        )
        summary = self.llm.fast_chat(
            system_prompt="你是剧情账本记录员，只输出中文单段，不要解释。",
            user_prompt=(
                f"章节编号：{chapter_index}\n"
                f"角色线索：{'；'.join(role_notes) if role_notes else '无'}\n"
                f"正文：{text[:1500]}\n"
                "请输出120字以内摘要，包含：事件推进+状态变化+下章接口。"
            ),
            fallback=fallback,
        )
        return {
            "chapter": chapter_index,
            "summary": summary.strip(),
            "roles": role_notes[:],
        }


class CriticAgent:
    """Deterministic checks to enforce output quality."""

    def evaluate(self, draft: str) -> str:
        issues: list[str] = []
        if len(draft.strip()) < 120:
            issues.append("Draft too short; expand scene details.")
        if re.search(r"\[Chapter Fragment|\[Style Guide|\[Role Continuity", draft):
            issues.append("Contains internal markers; remove them from final prose.")
        if re.search(r"(?i)project gutenberg|table of contents|release date|ebook", draft):
            issues.append("Contains metadata/license text; remove non-story content.")
        if not issues:
            return "PASS: no critical issues found."
        return "NEEDS_REVISION: " + " ".join(issues)


class RevisionAgent:
    """Applies deterministic revisions based on critique tags."""

    def revise(self, draft: str, critique: str, memory: StoryMemory) -> str:
        if critique.startswith("PASS"):
            return draft

        revised = draft
        revised = re.sub(r"\[(Chapter Fragment|Style Guide|Role Continuity)[^\n]*\n?", "", revised)
        revised = re.sub(r"(?im)^.*(Project Gutenberg|Release date|Table of Contents|Credits).*$", "", revised)
        if "Draft too short" in critique:
            revised += "\n夜色沉沉，众人各怀心机，表面的平静下已暗潮汹涌。"
        return revised.strip()
