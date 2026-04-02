"""Image prompt seed and image generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.prompts import reference_image_prompt
from app.services.gemini_service import GeminiService
from app.utils import resolve_reference_image_path


def build_reference_prompt_seed(reference_path: Path, style_label: str, gemini_service: GeminiService) -> dict[str, Any]:
    """Describe a reference image as a prompt seed."""

    if not gemini_service.is_configured:
        stem = reference_path.stem
        return {
            "prompt": f"An illustration inspired by {stem}.",
            "reference_description": f"An object-based character inspired by {stem}.",
            "key_visual_facts": [stem],
        }
    return gemini_service.describe_reference_image(str(reference_path), style_label)


def compose_final_image_prompt(reference_seed: dict[str, Any], style_prompt: str, style_label: str) -> str:
    """Combine reference description and style prompt into the final image prompt."""

    suffix = {
        "active_style": "bold active composition",
        "soft_style": "soft emotional composition",
    }.get(style_label, "illustrated composition")

    return (
        f"{reference_seed['prompt']}\n\n"
        f"Style direction: {style_prompt}\n\n"
        f"Create a fresh illustration based on the prompt above. "
        f"Do not apply a filter, recolor, or texture overlay. "
        f"Keep the character identity consistent, but redraw it as a new image with a {suffix}."
    )


def generate_images(style_prompts: dict[str, Any], reference_image: str, gemini_service: GeminiService) -> dict[str, Any]:
    """Generate active and soft style images from a reference image."""

    reference_path = resolve_reference_image_path(reference_image)
    active_seed = build_reference_prompt_seed(reference_path, "active_style", gemini_service)
    soft_seed = build_reference_prompt_seed(reference_path, "soft_style", gemini_service)

    active_prompt = compose_final_image_prompt(active_seed, str(style_prompts.get("active_style") or ""), "active_style")
    soft_prompt = compose_final_image_prompt(soft_seed, str(style_prompts.get("soft_style") or ""), "soft_style")

    try:
        active_output = gemini_service.generate_image_from_prompt(active_prompt, "active_style")
        soft_output = gemini_service.generate_image_from_prompt(soft_prompt, "soft_style")
    except Exception:
        active_output = gemini_service.generate_image(active_prompt, str(reference_path), "active_style")
        soft_output = gemini_service.generate_image(soft_prompt, str(reference_path), "soft_style")

    return {"active_style": active_output, "soft_style": soft_output}

