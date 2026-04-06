"""동화 생성을 위한 OpenAI 서비스 래퍼."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class OpenAIService:
    """OpenAI JSON 동화 생성 호출을 감싸는 얇은 래퍼."""

    api_key: str | None = None
    model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        load_dotenv()
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY") or None
        self.model = os.getenv("OPENAI_STORY_MODEL", self.model)

    @property
    def is_configured(self) -> bool:
        """OpenAI API 키와 SDK가 준비되었는지 반환한다."""

        return bool(self.api_key and OpenAI is not None)

    def _client(self) -> Any:
        """OpenAI 클라이언트를 만들고, 불가능하면 설정 오류를 낸다."""

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the project .env file before generating stories.")
        if OpenAI is None:
            raise RuntimeError("The openai package is not installed or could not be imported.")
        return OpenAI(api_key=self.api_key)

    def generate_story_json(self, prompt: str) -> str:
        """주어진 프롬프트로 JSON story package 문자열을 생성한다."""

        if not prompt.strip():
            raise ValueError("Story prompt must not be empty.")

        client = self._client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return JSON only. No markdown. No code fences. Follow the requested schema exactly.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - network/API dependent
            raise RuntimeError(f"OpenAI story generation request failed: {exc}") from exc

        text = response.choices[0].message.content if response.choices else ""
        if not text:
            raise RuntimeError("OpenAI story generation returned an empty response.")
        return text.strip()
