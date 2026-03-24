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
# 优先支持本项目约定变量：ai_api / ai_url
# 同时兼容常见 OpenAI 变量：OPENAI_API_KEY / OPENAI_BASE_URL
OPENAI_API_KEY = os.getenv("ai_api", os.getenv("OPENAI_API_KEY", ""))
OPENAI_BASE_URL = os.getenv("ai_url", os.getenv("OPENAI_BASE_URL", "https://api.vveai.com/v1"))
OPENAI_FAST_MODEL_NAME = os.getenv("OPENAI_FAST_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
OPENAI_LONG_MODEL_NAME = os.getenv("OPENAI_LONG_MODEL_NAME", "gpt-4.1")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "1.0"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

# Chunking / generation controls
DEFAULT_CHUNK_TOKEN_LIMIT = int(os.getenv("CHUNK_TOKEN_LIMIT", "2000"))
SLIDING_WINDOW_WORDS = int(os.getenv("SLIDING_WINDOW_WORDS", "800"))
MAX_CRITIC_RETRIES = int(os.getenv("MAX_CRITIC_RETRIES", "3"))
DEFAULT_RECURSIVE_STEPS = int(os.getenv("RECURSIVE_CONTINUATION_STEPS", "2"))
DEFAULT_PRE_SPLIT_CHARS = int(os.getenv("PRE_SPLIT_CHARS", "50000"))
DEFAULT_PRE_SUMMARY_CHARS = int(os.getenv("PRE_SUMMARY_CHARS", "5000"))


def ensure_project_dirs() -> None:
    """Create expected local data directories if they do not exist."""
    for directory in (DATA_DIR, INPUT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
