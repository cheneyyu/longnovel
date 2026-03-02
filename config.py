"""Configuration for the Xianxia transformation pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Core files
SOURCE_NOVEL_PATH = INPUT_DIR / "novel.txt"
STYLE_PROMPT_PATH = INPUT_DIR / "style.txt"
RESULT_NOVEL_PATH = OUTPUT_DIR / "result_xianxia.txt"
SQLITE_DB_PATH = DATA_DIR / "xianxia_state.db"
JSON_DB_PATH = DATA_DIR / "xianxia_state.json"

# LLM / API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Chunking / generation controls
DEFAULT_CHUNK_TOKEN_LIMIT = int(os.getenv("CHUNK_TOKEN_LIMIT", "2000"))
SLIDING_WINDOW_WORDS = int(os.getenv("SLIDING_WINDOW_WORDS", "800"))
MAX_CRITIC_RETRIES = int(os.getenv("MAX_CRITIC_RETRIES", "3"))


def ensure_project_dirs() -> None:
    """Create expected local data directories if they do not exist."""
    for directory in (DATA_DIR, INPUT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
