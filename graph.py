"""Pipeline graph orchestration for chapter adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import AdaptationAgent, CriticAgent, GenerationResult, RevisionAgent
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
    """Coordinates chunking -> generate -> critique -> revision."""

    def __init__(self, db: DatabaseManager):
        self.adapt = AdaptationAgent(db)
        self.critic = CriticAgent()
        self.revisor = RevisionAgent()

    def run(self, source_text: str, critic_retries: int = MAX_CRITIC_RETRIES) -> PipelineOutput:
        chunks = chunk_text(source_text)
        results: list[GenerationResult] = []

        for chunk in chunks:
            draft = self.adapt.generate(chunk)
            critique = self.critic.evaluate(chunk, draft)
            revised = draft

            retries = 0
            while critique.startswith("NEEDS_REVISION") and retries < critic_retries:
                revised = self.revisor.revise(revised, critique)
                critique = self.critic.evaluate(chunk, revised)
                retries += 1

            results.append(
                GenerationResult(
                    chunk_index=chunk.index,
                    draft=draft,
                    critique=critique,
                    revised=revised,
                )
            )

        return PipelineOutput(chunk_results=results)


def run_pipeline_to_file(source_text: str, output_path: Path, db: DatabaseManager) -> PipelineOutput:
    output = XianxiaPipeline(db).run(source_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.merged_text, encoding="utf-8")
    return output
