"""CLI entrypoint for the longnovel adaptation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_PRE_SPLIT_CHARS,
    DEFAULT_PRE_SUMMARY_CHARS,
    DEFAULT_RECURSIVE_STEPS,
    RESULT_NOVEL_PATH,
    SOURCE_NOVEL_PATH,
    STYLE_PROMPT_PATH,
    ensure_project_dirs,
)
from database import bootstrap_database
from graph import run_pipeline_to_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run longnovel adaptation pipeline.")
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=None,
        help="Optional upper bound for output characters (e.g. 20000 for a preview run).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=RESULT_NOVEL_PATH,
        help="Path to the generated output file.",
    )
    parser.add_argument(
        "--recursive-steps",
        type=int,
        default=None,
        help="Extra continuation rounds after source chunks are finished.",
    )
    parser.add_argument(
        "--pre-split-chars",
        type=int,
        default=DEFAULT_PRE_SPLIT_CHARS,
        help="When input exceeds this size, split source text into large parts before chunking.",
    )
    parser.add_argument(
        "--pre-summary-chars",
        type=int,
        default=DEFAULT_PRE_SUMMARY_CHARS,
        help="Per-part summary target size used during large-input pre-processing.",
    )
    parser.add_argument(
        "--quiet-preprocess",
        action="store_true",
        help="Disable verbose logs for large-input split/summarization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_project_dirs()
    db = bootstrap_database()

    if not SOURCE_NOVEL_PATH.exists():
        sample = (
            "Edmond Dantes escaped from Chateau d'If and returned to Marseilles. "
            "He prepared for a Duel and sought hidden Treasure."
        )
        SOURCE_NOVEL_PATH.write_text(sample, encoding="utf-8")

    if not STYLE_PROMPT_PATH.exists():
        STYLE_PROMPT_PATH.write_text("热血成长+群像权谋，节奏偏快。", encoding="utf-8")

    source_text = SOURCE_NOVEL_PATH.read_text(encoding="utf-8")
    user_style = STYLE_PROMPT_PATH.read_text(encoding="utf-8")
    output = run_pipeline_to_file(
        source_text,
        args.output_path,
        db,
        user_style=user_style,
        max_output_chars=args.max_output_chars,
        recursive_steps=args.recursive_steps if args.recursive_steps is not None else DEFAULT_RECURSIVE_STEPS,
        pre_split_chars=args.pre_split_chars,
        pre_summary_chars=args.pre_summary_chars,
        verbose_preprocess=not args.quiet_preprocess,
    )
    print(f"Generated {len(output.chunk_results)} chunk(s).")
    if args.max_output_chars is not None:
        print(f"Preview mode enabled. Output capped at {args.max_output_chars} chars.")
    print(f"Result saved to: {args.output_path}")


if __name__ == "__main__":
    main()
