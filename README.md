# longnovel

Multi-agent Python pipeline for transforming a source novel into a Chinese web-novel adaptation chapter by chapter.

> 当前仓库使用“跨语言小说改编”为示例，不限定题材。你可以输入任意语言小说，并通过风格提示词生成长篇连载改编文本。

## Current Progress

已实现多 agent 协作链路（可离线运行）：

- `StyleAgent`：读取简单/详细风格描述并补全为统一写作准则。
- `CharacterAgent`：从角色库检索并在每个 chunk 更新角色记忆，保证角色身份连续。
- `AdaptationAgent`：执行术语映射与正文生成。
- `ContinuityAgent`：产出分段摘要写入记忆，供后续上下文连贯。
- `CriticAgent` + `RevisionAgent`：检查并修补结构标记与叙事丰富度。

核心模块：

- `config.py`: 路径、API配置、运行参数。
- `database.py`: SQLite + JSON 状态管理（世界词表、角色设定）。
- `chunker.py`: 分块 + 滑动窗口上下文。
- `graph.py`: 多 agent 编排与执行。
- `main.py`: CLI 入口。

## Quick Start

1. Python 3.10+
2. 可选设置环境变量：
   - `OPENAI_API_KEY`
   - `OPENAI_BASE_URL`
   - `OPENAI_FAST_MODEL_NAME`
   - `OPENAI_LONG_MODEL_NAME`
   - `OPENAI_TEMPERATURE`（可选，建议 `1.0`）

```bash
python database.py
python main.py
```

首次运行会自动创建：

- `input/novel.txt`（示例原文）
- `input/style.txt`（风格描述，可写“简单描述”或“详细描述”）
- `output/result_xianxia.txt`（改编结果）

## Colab 示例

在 Google Colab 中可以直接运行以下单元：

```python
!git clone https://github.com/<your-org>/longnovel.git
%cd longnovel

!python -m pip install -U pip
!python database.py

# 1) 设置 OpenAI 兼容 API（同一个 base url，下挂两个模型）
import os
os.environ["OPENAI_API_KEY"] = "<your-api-key>"
os.environ["OPENAI_BASE_URL"] = "https://<your-openai-compatible-endpoint>/v1"
os.environ["OPENAI_FAST_MODEL_NAME"] = "<fast-cheap-model>"
os.environ["OPENAI_LONG_MODEL_NAME"] = "<long-context-pro-model>"
os.environ["OPENAI_TEMPERATURE"] = "1.0"

# 2) 你自己准备大体量小说 input/novel.txt（语言不限）
!mkdir -p input
!python - <<'PY'
from pathlib import Path
Path('input/style.txt').write_text('高张力冲突、章节结尾留钩子、人物关系持续升级。', encoding='utf-8')
print('请自行上传 input/novel.txt（可用 Colab 左侧文件面板上传）。')
PY

# 3) 运行：角色卡、情节推进、连续性摘要都通过 LLM 生成
!python main.py
!sed -n '1,120p' output/result_xianxia.txt
```

> 说明：此流程会同时使用 fast/cheap 模型（风格整理、角色卡更新、连续性摘要）和 long/pro 模型（正文改写）。

> 如果你的原文来自 Project Gutenberg 等站点，流水线会自动清理版权头、目录、页码等噪声后再分块。

## 如何满足“多个 agent + 连贯上下文”

- 章节被切分为 chunk，并附带滑动窗口上下文。
- 每个 chunk 按顺序经过 `Style -> Character -> Adapt -> Critic/Revision -> Continuity Summary`。
- 角色状态由数据库与 `StoryMemory.known_characters` 共同维护。
- 连续性由 `[Context Window]` 和 `chunk_summaries` 记忆保证。

## 自定义背景

虽然字段名为 `xianxia_term` / `xianxia_name`，本质是“改编后映射值”：

- 直接修改 `data/xianxia_state.db` 中的 `WorldMap` 与 `Characters`
- 或修改 `database.py` 中 `seed_mock_data()` 的种子数据

例如赛博朋克：

- `Treasure -> Quantum Vault`
- `Sect -> Megacorp Division`

## Run Full Pipeline

```bash
python main.py
```
