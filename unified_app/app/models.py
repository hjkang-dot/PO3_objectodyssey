"""Pydantic models for the prototype API."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_STORY_TONES = ("따뜻한", "모험적인", "교훈적인")


class VisionResult(BaseModel):
    """Vision model output passed in from the demo UI or future vision service."""

    model_config = ConfigDict(extra="allow")
    objects: list[str] = Field(default_factory=list)


class ParentInput(BaseModel):
    """Parent-provided character direction."""

    model_config = ConfigDict(extra="allow")
    name: str = ""
    job: str = ""
    personality: str = ""
    goal: str = ""
    extra_description: str = ""


class CharacterSheet(BaseModel):
    """Structured character sheet returned by the LLM."""

    original_object: str
    name: str
    job: str
    personality: str
    goal: str
    core_visual_traits: list[str] = Field(default_factory=list)
    tone: str


class StylePrompts(BaseModel):
    """Two style-specific image prompts for the same character."""

    active_style: str
    soft_style: str


class GeneratedImages(BaseModel):
    """Paths or base64 payloads for generated images."""

    active_style: str | None = None
    soft_style: str | None = None


class TtsScriptLine(BaseModel):
    """A single TTS-ready line with speaking direction."""

    line: str
    tone: str

    @field_validator("line", "tone")
    @classmethod
    def _ensure_not_blank(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("TTS script values must not be empty.")
        return cleaned


class StoryChoice(BaseModel):
    """Interactive choice shown after the story ends."""

    id: str
    text: str

    @field_validator("id")
    @classmethod
    def _validate_choice_id(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not re.fullmatch(r"[a-z]+(?:_[a-z]+)*", cleaned):
            raise ValueError("Choice id must be snake_case.")
        return cleaned

    @field_validator("text")
    @classmethod
    def _validate_choice_text(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Choice text must not be empty.")
        return cleaned


class StoryPackageResponse(BaseModel):
    """Full story generation payload for reading and TTS."""

    title: str
    story_paragraphs: list[str]
    tts_script: list[TtsScriptLine]
    choices: list[StoryChoice]

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Title must not be empty.")
        return cleaned

    @field_validator("story_paragraphs")
    @classmethod
    def _validate_story_paragraphs(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("Story must contain at least one paragraph.")
        if len(cleaned) > 5:  # 약간 완화
            raise ValueError("Story must contain at most 5 paragraphs.")
        for paragraph in cleaned:
            if len(paragraph) > 400:  # 200자는 너무 짧아 LLM이 초과하기 쉽습니다
                raise ValueError(f"Each story paragraph must be 400 characters or fewer (Got {len(paragraph)}).")
        return cleaned

    @field_validator("tts_script")
    @classmethod
    def _validate_tts_script(cls, value: list[TtsScriptLine]) -> list[TtsScriptLine]:
        if not value:
            raise ValueError("TTS script must contain at least one line.")
        return value

    @field_validator("choices")
    @classmethod
    def _validate_choices(cls, value: list[StoryChoice]) -> list[StoryChoice]:
        if len(value) != 2:
            raise ValueError("Choices must contain exactly 2 items.")
        return value


class StoryRequest(BaseModel):
    """Request body for story generation."""

    character_sheet: CharacterSheet
    extra_prompt: str = ""
    story_tone: str | None = None

    @field_validator("character_sheet")
    @classmethod
    def _validate_tone(cls, value: CharacterSheet) -> CharacterSheet:
        if value.tone not in ALLOWED_STORY_TONES:
            raise ValueError(f"tone must be one of: {', '.join(ALLOWED_STORY_TONES)}")
        return value

    @field_validator("story_tone")
    @classmethod
    def _validate_story_tone(cls, value: str | None) -> str | None:
        if value in (None, "", "랜덤"):
            return None
        if value not in ALLOWED_STORY_TONES:
            raise ValueError(f"story_tone must be one of: {', '.join(ALLOWED_STORY_TONES)} or 랜덤")
        return value


class CharacterSheetRequest(BaseModel):
    """Request body for character sheet generation."""

    vision_result: VisionResult
    parent_input: ParentInput


class StylePromptsRequest(BaseModel):
    """Request body for style prompt generation."""

    character_sheet: CharacterSheet


class GenerateImagesRequest(BaseModel):
    """Request body for image generation."""

    style_prompts: StylePrompts
    reference_image: str
    image_style: str = "active_style"


class GenerateStoryRequest(BaseModel):
    """Backward-compatible request body for story generation."""

    character_sheet: CharacterSheet


class PipelineRequest(BaseModel):
    """End-to-end request body."""

    vision_result: VisionResult
    parent_input: ParentInput
    reference_image: str


class PipelineResponse(BaseModel):
    """End-to-end response body."""

    character_sheet: CharacterSheet
    style_prompts: StylePrompts
    generated_images: GeneratedImages
    story: StoryPackageResponse


class ReferenceImagesResponse(BaseModel):
    """Available reference images inside the nukki folder."""

    reference_images: list[str]


class ErrorResponse(BaseModel):
    """Simple error payload."""

    detail: str
    extra: dict[str, Any] | None = None
