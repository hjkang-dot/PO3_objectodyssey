"""스토리북 파이프라인의 공개 호환 레이어.

실제 구현은 `app.storybook_core`에 모두 모아 두고,
이 파일은 기존 import 경로를 유지하기 위한 얇은 중간 레이어만 담당한다.
"""

from __future__ import annotations

from typing import Any

from app.storybook_core import build_story_prompt, generate_story_package

__all__ = ["build_story_prompt", "generate_story_package"]


def build_story(character_sheet: dict[str, Any], extra_prompt: str = "", story_tone: str | None = None) -> str:
    """예전 호출부에서 쓰던 이름과의 호환을 위한 보조 함수."""

    return build_story_prompt(character_sheet, extra_prompt=extra_prompt, story_tone=story_tone)
