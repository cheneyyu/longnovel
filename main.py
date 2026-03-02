"""CLI entrypoint for the longnovel adaptation pipeline."""

from __future__ import annotations

from config import RESULT_NOVEL_PATH, SOURCE_NOVEL_PATH, ensure_project_dirs
from database import bootstrap_database
from graph import run_pipeline_to_file


def main() -> None:
    ensure_project_dirs()
    db = bootstrap_database()

    if not SOURCE_NOVEL_PATH.exists():
        sample = (
            "Edmond Dantes escaped from Chateau d'If and returned to Marseilles. "
            "He prepared for a Duel and sought hidden Treasure."
        )
        SOURCE_NOVEL_PATH.write_text(sample, encoding="utf-8")

    source_text = SOURCE_NOVEL_PATH.read_text(encoding="utf-8")
    output = run_pipeline_to_file(source_text, RESULT_NOVEL_PATH, db)
    print(f"Generated {len(output.chunk_results)} chunk(s).")
    print(f"Result saved to: {RESULT_NOVEL_PATH}")


if __name__ == "__main__":
    main()
