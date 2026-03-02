"""Pipeline graph orchestration for chapter adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import (
    AdaptationAgent,
    CharacterAgent,
    ContinuityAgent,
    CriticAgent,
    GenerationResult,
    RevisionAgent,
    StoryMemory,
    StyleAgent,
)
from chunker import chunk_text
from config import MAX_CRITIC_RETRIES
from database import DatabaseManager


@dataclass(slots=True)
class PipelineOutput:
    chunk_results: list[GenerationResult]

    @property
    def merged_text(self) -> str:
        return "\n\n".join(item.revised for item in self.chunk_results)


class XianxiaPipeline:
    """Coordinates chunking -> multi-agent generate -> critique -> revision."""

    def __init__(self, db: DatabaseManager):
        self.style_agent = StyleAgent()
        self.character_agent = CharacterAgent(db)
        self.adapt = AdaptationAgent(db)
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
        results: list[GenerationResult] = []
        memory = StoryMemory(style_guide=self.style_agent.create_style_guide(user_style))

        for chunk in chunks:
            role_notes = self.character_agent.update_memory(chunk, memory)
            draft = self.adapt.generate(chunk, memory, role_notes)
            critique = self.critic.evaluate(draft)
            revised = draft

            retries = 0
            while critique.startswith("NEEDS_REVISION") and retries < critic_retries:
                revised = self.revisor.revise(revised, critique, memory)
                critique = self.critic.evaluate(revised)
                retries += 1

            memory.chunk_summaries.append(self.continuity_agent.summarize(revised))
            results.append(
                GenerationResult(
                    chunk_index=chunk.index,
                    draft=draft,
                    critique=critique,
                    revised=revised,
                )
            )

        return PipelineOutput(chunk_results=results)


def run_pipeline_to_file(
    source_text: str,
    output_path: Path,
    db: DatabaseManager,
    user_style: str = "",
) -> PipelineOutput:
    output = XianxiaPipeline(db).run(source_text, user_style=user_style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.merged_text, encoding="utf-8")
    return output
