"""Pydantic models for the prototype API."""

from __future__ import annotations

import re
from typing import Any, Literal

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
    original_object_hint: str = ""
    traits_input: str = ""


class PromptOptions(BaseModel):
    """Prompt-selection values used by the image prompt builder."""

    gender: Literal["boy", "girl"] = "girl"
    base_style: Literal["active", "soft"] = "active"
    category: str = "default"


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

    active_style: str
    soft_style: str


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


class StoryPage(BaseModel):
    """A single illustrated page in the storybook."""

    page_number: int
    sentences: list[str]
    page_text: str = ""
    image_prompt: str
    image_path: str | None = None

    @field_validator("page_number")
    @classmethod
    def _validate_page_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must start at 1.")
        return value

    @field_validator("sentences")
    @classmethod
    def _validate_sentences(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if len(cleaned) != 3:
            raise ValueError("Each story page must contain exactly 3 sentences.")
        return cleaned

    @field_validator("image_prompt")
    @classmethod
    def _validate_image_prompt(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Each story page must include an image_prompt.")
        return cleaned


class StoryPackageResponse(BaseModel):
    """Full story generation payload for reading and TTS."""

    title: str
    story_paragraphs: list[str] = Field(default_factory=list)
    story_pages: list[StoryPage] = Field(default_factory=list)
    cover_image_path: str | None = None
    cover_prompt: str | None = None
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
        return cleaned

    @field_validator("story_pages")
    @classmethod
    def _validate_story_pages(cls, value: list[StoryPage]) -> list[StoryPage]:
        if len(value) != 5:
            raise ValueError("Story must contain exactly 5 pages.")
        expected_page_numbers = list(range(1, 6))
        actual_page_numbers = [page.page_number for page in value]
        if actual_page_numbers != expected_page_numbers:
            raise ValueError("Story pages must be ordered from page 1 to page 5.")
        return value

    @field_validator("tts_script")
    @classmethod
    def _validate_tts_script(cls, value: list[TtsScriptLine]) -> list[TtsScriptLine]:
        if not value:
            raise ValueError("TTS script must contain at least one line.")
        if len(value) < 15:
            raise ValueError("TTS script must cover all 15 story sentences.")
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
    style_prompts: StylePrompts | None = None
    reference_image: str | None = None

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
    prompt_options: PromptOptions | None = None


class GenerateImagesRequest(BaseModel):
    """Request body for image generation."""

    style_prompts: StylePrompts
    reference_image: str


class GenerateStoryRequest(BaseModel):
    """Backward-compatible request body for story generation."""

    character_sheet: CharacterSheet


class PipelineRequest(BaseModel):
    """End-to-end request body."""

    vision_result: VisionResult
    parent_input: ParentInput
    reference_image: str
    prompt_options: PromptOptions | None = None


class PipelineResponse(BaseModel):
    """End-to-end response body."""

    character_sheet: CharacterSheet
    style_prompts: StylePrompts
    generated_images: GeneratedImages
    story: StoryPackageResponse | None = None
    story_error: str | None = None


class ReferenceImagesResponse(BaseModel):
    """Available reference images inside the nukki folder."""

    reference_images: list[str]


class ErrorResponse(BaseModel):
    """Simple error payload."""

    detail: str
    extra: dict[str, Any] | None = None
