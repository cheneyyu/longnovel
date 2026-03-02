"""OpenAI-compatible chat client with fast/long model routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from config import (
    LLM_REQUEST_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_FAST_MODEL_NAME,
    OPENAI_LONG_MODEL_NAME,
    OPENAI_TEMPERATURE,
)


@dataclass(slots=True)
class LLMRouter:
    """Small helper for calling OpenAI-compatible chat completion APIs."""

    api_key: str = OPENAI_API_KEY
    base_url: str = OPENAI_BASE_URL
    fast_model: str = OPENAI_FAST_MODEL_NAME
    long_model: str = OPENAI_LONG_MODEL_NAME
    temperature: float = OPENAI_TEMPERATURE
    timeout_s: int = LLM_REQUEST_TIMEOUT

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def fast_chat(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        return self._chat(self.fast_model, system_prompt, user_prompt, fallback)

    def long_chat(self, system_prompt: str, user_prompt: str, fallback: str) -> str:
        return self._chat(self.long_model, system_prompt, user_prompt, fallback)

    def _chat(self, model: str, system_prompt: str, user_prompt: str, fallback: str) -> str:
        if not self.enabled:
            return fallback

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return content or fallback
        except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            return fallback
