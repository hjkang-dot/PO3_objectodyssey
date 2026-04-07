"""Image prompt seed and image generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    """Combine reference description and style prompt into a character-redesign prompt."""

    suffix = {
        "active_style": "high-energy cinematic picture-book scene",
        "soft_style": "warm gentle picture-book scene",
    }.get(style_label, "storybook illustration")

    reference_description = str(reference_seed.get("reference_description") or "").strip()
    key_visual_facts = ", ".join(reference_seed.get("key_visual_facts") or [])

    return (
        "Use the attached reference image as identity source material, not as a final canvas.\n"
        "Transform the referenced object or figure into a fully illustrated children's-book character for children ages 6 to 8.\n"
        "Keep only the recognizable identity cues from the reference image such as silhouette, key shapes, "
        "face placement, standout accessories, and memorable proportions.\n"
        "Do not preserve the original photo background, lighting, texture, or camera look.\n"
        "Do not apply a simple filter, recolor, photo effect, or overlay.\n"
        "Create a brand-new drawn image with clean illustration edges, fresh composition, a new background, and a strong toy-like character presence.\n"
        "Make the design feel iconic and easy for a child to recognize as one recurring character.\n"
        "Avoid washed-out pastel color treatment; use richer child-friendly colors with clear silhouette readability.\n\n"
        f"Reference analysis: {reference_seed['prompt']}\n"
        f"Reference description: {reference_description}\n"
        f"Key visual facts to preserve: {key_visual_facts}\n\n"
        f"Character direction: {style_prompt}\n\n"
        f"Final goal: a {suffix} that clearly feels like the same character reimagined from the reference image."
    )


def generate_images(style_prompts: dict[str, Any], reference_image: str, gemini_service: GeminiService, image_style: str = "active_style") -> dict[str, Any]:
    """Generate a single style image from a reference image."""

    reference_path = resolve_reference_image_path(reference_image)
    
    if image_style == "active_style":
        active_seed = build_reference_prompt_seed(reference_path, "active_style", gemini_service)
        active_prompt = compose_final_image_prompt(active_seed, str(style_prompts.get("active_style") or ""), "active_style")
        active_output = gemini_service.generate_image(active_prompt, str(reference_path), "active_style")
        return {"active_style": active_output}
    else:
        soft_seed = build_reference_prompt_seed(reference_path, "soft_style", gemini_service)
        soft_prompt = compose_final_image_prompt(soft_seed, str(style_prompts.get("soft_style") or ""), "soft_style")
        soft_output = gemini_service.generate_image(soft_prompt, str(reference_path), "soft_style")
        return {"soft_style": soft_output}
