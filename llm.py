"""OpenAI client wrapper with deterministic fallback for offline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


@dataclass(slots=True)
class LLMClient:
    """Thin chat-completion helper used by all high-level agents."""

    model_name: str = OPENAI_MODEL_NAME
    enabled: bool = field(init=False, default=False)
    _client: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.enabled = bool(OPENAI_API_KEY and OpenAI is not None)
        if self.enabled:
            self._client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        if not self.enabled or self._client is None:
            return self._fallback(system_prompt, user_prompt)

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _fallback(system_prompt: str, user_prompt: str) -> str:
        """Provide deterministic output so pipeline remains runnable without API key."""
        shortened = user_prompt.strip().replace("\n", " ")
        if len(shortened) > 600:
            shortened = shortened[:600] + "..."
        return f"[LLM_FALLBACK]\nSYSTEM: {system_prompt[:120]}\nUSER: {shortened}"
