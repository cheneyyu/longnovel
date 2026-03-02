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
from llm import LLMRouter
from preprocess import clean_source_novel


@dataclass(slots=True)
class PipelineOutput:
    chunk_results: list[GenerationResult]

    @property
    def merged_text(self) -> str:
        return "\n\n".join(item.revised for item in self.chunk_results)


class XianxiaPipeline:
    """Coordinates chunking -> multi-agent generate -> critique -> revision."""

    def __init__(self, db: DatabaseManager, llm: LLMRouter | None = None):
        self.llm = llm or LLMRouter()
        self.style_agent = StyleAgent(self.llm)
        self.character_agent = CharacterAgent(db, self.llm)
        self.adapt = AdaptationAgent(db, self.llm)
        self.continuity_agent = ContinuityAgent(self.llm)
        self.critic = CriticAgent()
        self.revisor = RevisionAgent()

    def run(
        self,
        source_text: str,
        user_style: str = "",
        critic_retries: int = MAX_CRITIC_RETRIES,
        max_output_chars: int | None = None,
    ) -> PipelineOutput:
        cleaned_source = clean_source_novel(source_text)
        chunks = chunk_text(cleaned_source)
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

            if max_output_chars is not None:
                merged = "\n\n".join(item.revised for item in results)
                if len(merged) >= max_output_chars:
                    break

        return PipelineOutput(chunk_results=results)


def run_pipeline_to_file(
    source_text: str,
    output_path: Path,
    db: DatabaseManager,
    user_style: str = "",
    max_output_chars: int | None = None,
) -> PipelineOutput:
    output = XianxiaPipeline(db).run(source_text, user_style=user_style, max_output_chars=max_output_chars)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_text = output.merged_text
    if max_output_chars is not None:
        merged_text = merged_text[:max_output_chars]
    output_path.write_text(merged_text, encoding="utf-8")
    return output
