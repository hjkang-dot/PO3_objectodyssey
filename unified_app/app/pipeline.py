"""High-level pipeline functions for the PO3 prototype."""

from __future__ import annotations

from typing import Any

from app.character import build_character_sheet as _build_character_sheet
from app.character import build_style_prompts as _build_style_prompts
from app.image_flow import generate_images as _generate_images
from app.services.gemini_service import GeminiService
from app.story_pipeline import generate_story_package as _generate_story_package

gemini_service = GeminiService()


def build_character_sheet(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> dict[str, Any]:
    """Build a structured character sheet from vision and parent inputs."""

    return _build_character_sheet(vision_result, parent_input, gemini_service)


def build_style_prompts(character_sheet: dict[str, Any]) -> dict[str, Any]:
    """Build active and soft image prompts for the same character."""

    return _build_style_prompts(character_sheet, gemini_service)


def generate_images(style_prompts: dict[str, Any], reference_image: str, image_style: str = "active_style") -> dict[str, Any]:
    """Generate active and soft style images from a reference image."""

    return _generate_images(style_prompts, reference_image, gemini_service, image_style)


def generate_story(
    character_sheet: dict[str, Any],
    extra_prompt: str = "",
    story_tone: str | None = None,
) -> dict[str, Any]:
    """Generate a short story package from the final character sheet."""

    return _generate_story_package(character_sheet, extra_prompt=extra_prompt, story_tone=story_tone)


def run_pipeline(vision_result: dict[str, Any], parent_input: dict[str, Any], reference_image: str, image_style: str = "active_style") -> dict[str, Any]:
    """Run the full prototype pipeline and return the final demo payload."""

    character_sheet = build_character_sheet(vision_result, parent_input)
    style_prompts = build_style_prompts(character_sheet)
    generated_images = generate_images(style_prompts, reference_image, image_style)
    story = generate_story(character_sheet)
    return {
        "character_sheet": character_sheet,
        "style_prompts": style_prompts,
        "generated_images": generated_images,
        "story": story,
    }
