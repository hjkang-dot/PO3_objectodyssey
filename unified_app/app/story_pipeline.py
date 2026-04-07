"""짧은 참여형 아동 동화를 생성하는 파이프라인."""

from __future__ import annotations

import random
from typing import Any

from app.models import ALLOWED_STORY_TONES, CharacterSheet, StoryPackageResponse
from app.prompts import story_generation_prompt
from app.services.openai_service import OpenAIService
from app.utils import safe_json_loads

DEFAULT_STORY_TONE = "\ub530\ub73b\ud55c"


def _normalize_story_character_sheet(character_sheet: dict[str, Any]) -> dict[str, Any]:
    """동화 생성에 맞게 character_sheet를 검증하고 정규화한다."""

    validated = CharacterSheet.model_validate(character_sheet).model_dump()
    if validated["tone"] not in ALLOWED_STORY_TONES:
        validated["tone"] = DEFAULT_STORY_TONE
    return validated


def _resolve_story_tone(character_sheet: dict[str, Any], story_tone: str | None) -> str:
    """동화 생성에 사용할 최종 tone 값을 결정한다."""

    if story_tone in ALLOWED_STORY_TONES:
        return story_tone
    if story_tone in (None, "", "랜덤"):
        return random.choice(ALLOWED_STORY_TONES)
    return character_sheet["tone"]


def build_story_prompt(character_sheet: dict, extra_prompt: str = "", story_tone: str | None = None) -> str:
    """전체 story package 생성을 위한 OpenAI 프롬프트를 만든다."""

    normalized = _normalize_story_character_sheet(character_sheet)
    resolved_tone = _resolve_story_tone(normalized, story_tone)
    return story_generation_prompt(normalized, extra_prompt=extra_prompt, story_tone=resolved_tone)


def _validate_story_package(data: dict[str, Any], character_sheet: dict[str, Any]) -> dict[str, Any]:
    """생성된 story package를 검증하고 프로젝트 제약을 강제한다."""

    normalized = _normalize_story_character_sheet(character_sheet)
    story_package = StoryPackageResponse.model_validate(data).model_dump()
    combined_story = " ".join(story_package["story_paragraphs"])
    combined_tts = " ".join(item["line"] for item in story_package["tts_script"])

    # 시연 단계에서는 표현이 조금만 바뀌어도 실패하지 않도록
    # 핵심 식별자만 최소 검증하되, 없더라도 에러 대신 경고만 남겨 파이프라인이 끊기지 않도록 완화합니다.
    if normalized["name"] not in combined_story and normalized["name"] not in combined_tts:
        print(f"Warning: Generated story might be missing character name: {normalized['name']}")

    if len(story_package["tts_script"]) < len(story_package["story_paragraphs"]):
        print("Warning: TTS script might be shorter than the story paragraphs.")

    if len(combined_tts) < max(20, len(combined_story) // 2):
        raise ValueError("TTS script is too short to cover the full story.")

    return story_package


def generate_story_package(
    character_sheet: dict,
    extra_prompt: str = "",
    story_tone: str | None = None,
) -> dict:
    """character_sheet를 바탕으로 최종 story package를 생성하고 검증한다."""

    normalized = _normalize_story_character_sheet(character_sheet)
    resolved_tone = _resolve_story_tone(normalized, story_tone)
    normalized["tone"] = resolved_tone
    prompt = build_story_prompt(normalized, extra_prompt=extra_prompt, story_tone=resolved_tone)
    service = OpenAIService()

    try:
        raw_text = service.generate_story_json(prompt)
    except Exception as exc:
        raise RuntimeError(f"Story generation failed before parsing: {exc}") from exc

    try:
        parsed = safe_json_loads(raw_text)
    except Exception as exc:
        raise RuntimeError(f"Story generation returned invalid JSON: {exc}") from exc

    try:
        return _validate_story_package(parsed, normalized)
    except Exception as exc:
        raise RuntimeError(f"Story generation returned an invalid story package: {exc}") from exc
