"""OpenAI service helpers for story generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class OpenAIService:
    """Wrapper around the OpenAI Chat Completions API."""

    api_key: str | None = None
    model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY") or None
        self.model = os.getenv("OPENAI_STORY_MODEL", self.model)

    @property
    def is_configured(self) -> bool:
        """Return True when an OpenAI API key and SDK are available."""

        return bool(self.api_key and OpenAI is not None)

    def _client(self) -> Any:
        if not self.is_configured:
            raise RuntimeError("OpenAI API is not configured.")
        return OpenAI(api_key=self.api_key)

    def generate_story_json(self, prompt: str) -> str:
        """Generate JSON text using OpenAI."""

        if not self.is_configured:
            raise RuntimeError("OpenAI API is not configured.")

        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return JSON only. No markdown. No code fences.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content if response.choices else ""
        if not text:
            raise RuntimeError("OpenAI story generation returned an empty response.")
        return text.strip()

