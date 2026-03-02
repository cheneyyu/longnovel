# longnovel

Multi-agent Python pipeline for transforming a source novel into a Chinese web-novel adaptation chapter by chapter.

> 当前仓库默认用“西方小说 -> 仙侠风”作为示例，但**不限制仙侠**。你可以通过自定义映射词表和角色设定，把它改成你自己的世界观（科幻、赛博、蒸汽朋克、都市异能等）。

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

## Colab Example (Clone + API URL/Key + Model Name)

在 Colab 里可以直接按下面一个代码块运行：

```bash
# 1) clone
!git clone https://github.com/<your-org-or-user>/longnovel.git
%cd longnovel

# 2) 配置 API（你可以替换成 OpenAI / OpenRouter / 自建兼容服务）
import os
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"   # 例如: https://openrouter.ai/api/v1
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"               # 例如: openrouter 上的模型名

# 3) 准备输入文本（你也可以自行上传成 input/novel.txt）
from pathlib import Path
Path("input").mkdir(exist_ok=True)
Path("input/novel.txt").write_text(
    "A young detective enters a neon megacity and uncovers a conspiracy.",
    encoding="utf-8",
)

# 4) 初始化数据库并运行
!python database.py
!python main.py

# 5) 查看输出
!sed -n '1,80p' output/result_xianxia.txt
```

## 自定义背景（不只是仙侠）

默认字段名虽然叫 `xianxia_term` / `xianxia_name`，但它本质上是“改编后映射值”，你可以填任何设定。

- 方式 A（推荐）：初始化后直接改 `data/xianxia_state.db` 里的 `WorldMap` 和 `Characters`。
- 方式 B：修改 `database.py` 里的 `seed_mock_data()`，把默认 seed 换成你的背景。

例如你要做“赛博朋克”映射：

- `Treasure -> Quantum Vault`
- `Sect -> Megacorp Division`
- 角色门派/境界字段改成你自己的体系（如 `Tier-3 Augmented`）

## Implemented Modules

- `chunker.py`: sentence-based chunking with configurable token budget and sliding-window context.
- `agents.py`: deterministic adaptation / critique / revision agents as an offline baseline.
- `graph.py`: orchestration pipeline (`XianxiaPipeline`) for chunk-level processing.
- `main.py`: CLI entrypoint that boots DB, reads input, runs the pipeline, and writes output.

## Run Full Pipeline

```bash
python main.py
```
