"""기존 호출부와 호환되는 story 헬퍼."""

from __future__ import annotations

from typing import Any

from app.story_pipeline import build_story_prompt, generate_story_package


def build_story(character_sheet: dict[str, Any], openai_service: Any | None = None) -> dict[str, Any]:
    """새 story 파이프라인을 감싸는 하위 호환용 래퍼."""

    _ = openai_service
    return generate_story_package(character_sheet)


__all__ = ["build_story", "build_story_prompt", "generate_story_package"]
