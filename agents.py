"""Agent implementations for long-novel generation with consistency checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from chunker import TextChunk
from database import DatabaseManager
from llm import LLMClient


@dataclass(slots=True)
class StoryMemory:
    """Cross-chunk memory used by agents to keep narrative coherence."""

    style_guide: str = ""
    setting: str = ""
    outline: str = ""
    known_characters: dict[str, dict[str, str]] = field(default_factory=dict)
    chapter_summaries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    chapter_index: int
    draft: str
    critique: str
    revised: str


class SettingAgent:
    """Mixes source material and user direction into a new unified setting."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build(self, source_text: str, user_style: str) -> str:
        system_prompt = "你是网文策划编辑，擅长融合原著要素与作者需求，输出可执行设定。"
        user_prompt = (
            "请基于输入原著片段与用户描述，生成一个新的世界观设定。\n"
            "输出格式：\n"
            "1) 核心主题\n2) 世界规则\n3) 势力与角色关系\n4) 叙事限制（避免崩设）\n\n"
            f"[原著]\n{source_text[:4000]}\n\n[用户描述]\n{user_style.strip() or '保持热血成长与群像冲突'}"
        )
        return self.llm.complete(system_prompt, user_prompt, max_tokens=1000)


class PlanningAgent:
    """Builds an outline plus fine-grained beat plan around every 10k characters."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def build(self, source_text: str, setting: str) -> str:
        estimated_chars = max(len(source_text), 10000)
        unit_count = max(1, math.ceil(estimated_chars / 10000))
        system_prompt = "你是长篇网文大纲设计师，重点保证人物动机、伏笔回收与节奏一致性。"
        user_prompt = (
            "请输出两层计划：\n"
            "A. 总体大纲（阶段目标、主线冲突、关键反转）。\n"
            "B. 细纲：每1万字一个单元，每个单元约500字，包含出场角色、推进事件、伏笔与一致性约束。\n"
            f"预计单元数：{unit_count}。\n\n"
            f"[原著输入]\n{source_text[:4000]}\n\n[设定]\n{setting}"
        )
        return self.llm.complete(system_prompt, user_prompt, max_tokens=1800)


class ChapterWriterAgent:
    """Writes chapter drafts using setting, outline and rolling memory."""

    def __init__(self, llm: LLMClient, db: DatabaseManager):
        self.llm = llm
        world_map = db.fetch_world_map()
        characters = db.fetch_characters()
        self.term_map = {row["original_term"]: row["xianxia_term"] for row in world_map}
        self.name_map = {row["original_name"]: row["xianxia_name"] for row in characters}

    def _pre_replace_terms(self, text: str) -> str:
        converted = text
        for original, replaced in sorted(self.name_map.items(), key=lambda i: len(i[0]), reverse=True):
            converted = re.sub(rf"\b{re.escape(original)}\b", replaced, converted)
        for original, replaced in sorted(self.term_map.items(), key=lambda i: len(i[0]), reverse=True):
            converted = re.sub(rf"\b{re.escape(original)}\b", replaced, converted)
        return converted

    def write(self, chunk: TextChunk, memory: StoryMemory, role_notes: list[str]) -> str:
        converted_chunk = self._pre_replace_terms(chunk.text)
        continuity = "；".join(role_notes) if role_notes else "延续既有人物关系并避免动机突变"
        system_prompt = "你是网络小说主笔，重视剧情质量、角色一致性、伏笔衔接和可读性。"
        user_prompt = (
            f"[章节序号] {chunk.index}\n"
            f"[统一设定]\n{memory.setting}\n\n"
            f"[总体计划]\n{memory.outline}\n\n"
            f"[人物连续性]\n{continuity}\n\n"
            f"[近期记忆]\n{' | '.join(memory.chapter_summaries[-3:])}\n\n"
            f"[待改编原文]\n{converted_chunk}\n\n"
            "要求：\n"
            "1) 以中文输出成稿，避免提及提示词。\n"
            "2) 章末保留推进钩子。\n"
            "3) 保持术语与人物称谓前后一致。"
        )
        return self.llm.complete(system_prompt, user_prompt, max_tokens=1400)


class CharacterAgent:
    """Tracks characters from DB and current chunk to maintain role consistency."""

    def __init__(self, db: DatabaseManager):
        self.character_rows = db.fetch_characters()

    def update_memory(self, chunk: TextChunk, memory: StoryMemory) -> list[str]:
        notes: list[str] = []
        for row in self.character_rows:
            original = row["original_name"]
            adapted = row["xianxia_name"]
            if re.search(rf"\b{re.escape(original)}\b", chunk.text):
                memory.known_characters[adapted] = row
                notes.append(f"{adapted}({row['sect']}, {row['cultivation_level']}, {row['status']})")
        return notes


class ContinuityAgent:
    """Produces compact memory summary for each generated chapter."""

    def summarize(self, revised_text: str) -> str:
        clean = revised_text.replace("\n", " ").strip()
        return clean if len(clean) <= 120 else clean[:120] + "..."


class CriticAgent:
    """Deterministic checks for quality gate before final output."""

    def evaluate(self, draft: str) -> str:
        issues: list[str] = []
        if len(draft) < 220:
            issues.append("Draft too short for chapter quality baseline.")
        if "[LLM_FALLBACK]" in draft:
            issues.append("LLM fallback used; provide OPENAI_API_KEY for production-quality output.")
        if not re.search(r"[。！？]$", draft.strip()):
            issues.append("Chapter ending punctuation missing.")
        if not issues:
            return "PASS: no critical issues found."
        return "NEEDS_REVISION: " + " ".join(issues)


class RevisionAgent:
    """Applies deterministic revisions based on critique tags."""

    def revise(self, draft: str, critique: str) -> str:
        revised = draft
        if "Draft too short" in critique:
            revised += "\n夜色渐沉，新的冲突在众人尚未察觉时悄然逼近。"
        if "ending punctuation missing" in critique:
            revised = revised.rstrip() + "。"
        return revised
