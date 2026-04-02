"""Pydantic models for the prototype API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    active_style: str
    soft_style: str


class StoryResponse(BaseModel):
    """Child-friendly story output."""

    title: str
    story: list[str]


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


class GenerateStoryRequest(BaseModel):
    """Request body for story generation."""

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
    story: StoryResponse


class ReferenceImagesResponse(BaseModel):
    """Available reference images inside the nukki folder."""

    reference_images: list[str]


class ErrorResponse(BaseModel):
    """Simple error payload."""

    detail: str
    extra: dict[str, Any] | None = None

