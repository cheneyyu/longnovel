# longnovel

Multi-agent Python pipeline for transforming a Western novel into a Chinese Xianxia web novel chapter by chapter.

## Current Progress
Initial project foundation is in place:

- `config.py`: central configuration for paths, OpenAI-compatible API settings (supports custom base URLs like OpenRouter), and runtime controls.
- `database.py`: SQLite + JSON local state manager with:
  - `WorldMap` table (`original_term` -> `xianxia_term`)
  - `Characters` table (`original_name`, `xianxia_name`, `sect`, `cultivation_level`, `status`)
  - seeded mock data for *The Count of Monte Cristo* adaptation.

## Quick Start (Current Stage)

1. Use Python 3.10+
2. Optionally set environment variables:
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `OPENAI_MODEL_NAME`
3. Initialize local DB and JSON snapshot:

```bash
python database.py
```

Artifacts generated:

- `data/xianxia_state.db`
- `data/xianxia_state.json`

## Next Modules (Planned)

- `chunker.py`
- `agents.py`
- `graph.py`
- `main.py`
