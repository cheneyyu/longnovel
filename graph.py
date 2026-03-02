"""Pipeline graph orchestration for novel-in -> setting -> plan -> novel-out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import (
    ChapterWriterAgent,
    CharacterAgent,
    ContinuityAgent,
    CriticAgent,
    GenerationResult,
    PlanningAgent,
    RevisionAgent,
    SettingAgent,
    StoryMemory,
)
from chunker import chunk_text
from config import MAX_CRITIC_RETRIES
from database import DatabaseManager
from llm import LLMClient


@dataclass(slots=True)
class PipelineOutput:
    setting: str
    outline: str
    chunk_results: list[GenerationResult]

    @property
    def merged_text(self) -> str:
        return "\n\n".join(item.revised for item in self.chunk_results)


class LongNovelPipeline:
    """Coordinates setting generation, planning, and chapter writing."""

    def __init__(self, db: DatabaseManager):
        llm = LLMClient()
        self.setting_agent = SettingAgent(llm)
        self.planning_agent = PlanningAgent(llm)
        self.character_agent = CharacterAgent(db)
        self.writer_agent = ChapterWriterAgent(llm, db)
        self.continuity_agent = ContinuityAgent()
        self.critic = CriticAgent()
        self.revisor = RevisionAgent()

    def run(
        self,
        source_text: str,
        user_style: str = "",
        critic_retries: int = MAX_CRITIC_RETRIES,
    ) -> PipelineOutput:
        chunks = chunk_text(source_text)
        setting = self.setting_agent.build(source_text, user_style)
        outline = self.planning_agent.build(source_text, setting)

        results: list[GenerationResult] = []
        memory = StoryMemory(setting=setting, outline=outline, style_guide=user_style)

        for chunk in chunks:
            role_notes = self.character_agent.update_memory(chunk, memory)
            draft = self.writer_agent.write(chunk, memory, role_notes)
            critique = self.critic.evaluate(draft)
            revised = draft

            retries = 0
            while critique.startswith("NEEDS_REVISION") and retries < critic_retries:
                revised = self.revisor.revise(revised, critique)
                critique = self.critic.evaluate(revised)
                retries += 1

            memory.chapter_summaries.append(self.continuity_agent.summarize(revised))
            results.append(
                GenerationResult(
                    chapter_index=chunk.index,
                    draft=draft,
                    critique=critique,
                    revised=revised,
                )
            )

        return PipelineOutput(setting=setting, outline=outline, chunk_results=results)


def run_pipeline_to_files(
    source_text: str,
    result_output_path: Path,
    setting_output_path: Path,
    plan_output_path: Path,
    db: DatabaseManager,
    user_style: str = "",
) -> PipelineOutput:
    output = LongNovelPipeline(db).run(source_text, user_style=user_style)
    for path in (result_output_path, setting_output_path, plan_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    setting_output_path.write_text(output.setting, encoding="utf-8")
    plan_output_path.write_text(output.outline, encoding="utf-8")
    result_output_path.write_text(output.merged_text, encoding="utf-8")
    return output
