# longnovel

Multi-agent Python pipeline for transforming a source novel into a Chinese web-novel adaptation chapter by chapter.

## 当前版本管线（你提出的三段式）

1. **novel in**：输入原著 `input/novel.txt` + 用户描述 `input/style.txt`。
2. **设定生成**：`SettingAgent` 综合原著与用户要求，输出统一设定到 `output/setting.txt`。
3. **计划生成**：`PlanningAgent` 基于原著+设定输出总纲，并按“每 1 万字约 500 字细纲”生成到 `output/plan.txt`。
4. **novel out**：`ChapterWriterAgent` 按 chunk 逐章生成正文，`CriticAgent`/`RevisionAgent` 做一致性与质量兜底，输出 `output/result_novel.txt`。

## OpenAI 调用与依赖

项目已接入 `openai` 官方 Python SDK（见 `llm.py`）。

- 环境变量：
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`（可选）
  - `OPENAI_MODEL_NAME`（默认 `gpt-4o-mini`）
- 依赖安装：

```bash
pip install -r requirements.txt
```

> 若未提供 API key，系统会走 deterministic fallback（可运行但仅用于调试，不代表真实写作质量）。

## Quick Start

```bash
python database.py
python main.py
```

首次运行会自动创建：

- `input/novel.txt`
- `input/style.txt`
- `output/setting.txt`
- `output/plan.txt`
- `output/result_novel.txt`

## 代码结构

- `llm.py`: OpenAI SDK 封装与 fallback。
- `agents.py`: 设定/计划/写作/角色连续性/审校 Agent。
- `graph.py`: 三段式管线编排。
- `config.py`: 路径、API配置、运行参数。
- `database.py`: SQLite + JSON 状态管理（术语映射、角色设定）。
- `chunker.py`: 分块 + 滑动窗口上下文。
- `main.py`: CLI 入口。
