"""Character sheet and style prompt helpers."""

from __future__ import annotations

from typing import Any

from app.models import CharacterSheet
from app.prompts import character_sheet_prompt, style_prompts_prompt
from app.services.gemini_service import GeminiService
from app.utils import safe_json_loads


def fallback_character_sheet(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic character sheet when the text model is unavailable."""

    objects = vision_result.get("objects") or []
    original_object = str(objects[0]) if objects else "unknown object"
    name = str(parent_input.get("name") or "코코")
    job = str(parent_input.get("job") or "friend of the stars")
    personality = str(parent_input.get("personality") or "warm and brave")
    goal = str(parent_input.get("goal") or "wants to discover something new")
    extra = str(parent_input.get("extra_description") or "").strip()

    traits = [
        f"based on the appearance of {original_object}",
        "keeps the same friendly silhouette as the reference image",
        f"wears visual hints of the job: {job}",
    ]
    if extra:
        traits.append(extra)

    return {
        "original_object": original_object,
        "name": name,
        "job": job,
        "personality": personality,
        "goal": goal,
        "core_visual_traits": traits[:5],
        "tone": "warm, adventurous, and child-friendly",
    }


def validate_character_sheet(data: dict[str, Any], vision_result: dict[str, Any], parent_input: dict[str, Any]) -> dict[str, Any]:
    """Merge model output with defaults and validate the final schema."""

    fallback = fallback_character_sheet(vision_result, parent_input)
    merged = {**fallback, **data}

    core_traits = merged.get("core_visual_traits")
    if not isinstance(core_traits, list):
        core_traits = fallback["core_visual_traits"]

    cleaned = {
        "original_object": str(merged.get("original_object") or fallback["original_object"]),
        "name": str(merged.get("name") or fallback["name"]),
        "job": str(merged.get("job") or fallback["job"]),
        "personality": str(merged.get("personality") or fallback["personality"]),
        "goal": str(merged.get("goal") or fallback["goal"]),
        "core_visual_traits": [str(item).strip() for item in core_traits if str(item).strip()] or fallback["core_visual_traits"],
        "tone": str(merged.get("tone") or fallback["tone"]),
    }
    return CharacterSheet.model_validate(cleaned).model_dump()


def build_character_sheet(vision_result: dict[str, Any], parent_input: dict[str, Any], gemini_service: GeminiService) -> dict[str, Any]:
    """Build a structured character sheet from vision and parent inputs."""

    prompt = character_sheet_prompt(vision_result, parent_input)
    try:
        raw_text = gemini_service.generate_text(prompt)
        parsed = safe_json_loads(raw_text)
    except Exception:
        parsed = {}
    return validate_character_sheet(parsed, vision_result, parent_input)


def fallback_style_prompts(character_sheet: dict[str, Any]) -> dict[str, Any]:
    """Create style prompts locally when Gemini is unavailable."""

    name = character_sheet["name"]
    job = character_sheet["job"]
    original_object = character_sheet["original_object"]
    traits = ", ".join(character_sheet.get("core_visual_traits", []))
    base_instruction = (
        "Use the reference image only to extract character identity cues. Reimagine the subject as a "
        "storybook character and paint a completely new illustration. Do not apply a simple filter, "
        "recolor, texture overlay, or photo retouch."
    )

    active = (
        f"{base_instruction} "
        f"{name}, a {job}, should preserve the same silhouette, face placement, body shape, and "
        f"distinctive details of the {original_object}, while becoming a clearly illustrated character. "
        f"Active and adventurous pose, vivid colors, dynamic motion, bright lighting, "
        f"dramatic 3/4 angle, motion lines, bold background, high-energy children's picture book "
        f"illustration, energetic composition, {traits}."
    )
    soft = (
        f"{base_instruction} "
        f"{name}, a {job}, should preserve the same silhouette, face placement, body shape, and "
        f"distinctive details of the {original_object}, while becoming a clearly illustrated character. "
        f"Soft and warm pose, pastel colors, gentle lighting, cozy atmosphere, front-facing or "
        f"seated pose, dreamy background, soft brush texture, children's picture book illustration, "
        f"tender composition, {traits}."
    )
    return {"active_style": active, "soft_style": soft}


def validate_style_prompts(data: dict[str, Any], character_sheet: dict[str, Any]) -> dict[str, Any]:
    """Merge model output with defaults for style prompts."""

    fallback = fallback_style_prompts(character_sheet)
    merged = {**fallback, **data}
    return {
        "active_style": str(merged.get("active_style") or fallback["active_style"]).strip(),
        "soft_style": str(merged.get("soft_style") or fallback["soft_style"]).strip(),
    }


def build_style_prompts(character_sheet: dict[str, Any], gemini_service: GeminiService) -> dict[str, Any]:
    """Build active and soft image prompts for the same character."""

    prompt = style_prompts_prompt(character_sheet)
    try:
        raw_text = gemini_service.generate_text(prompt)
        parsed = safe_json_loads(raw_text)
    except Exception:
        parsed = {}
    return validate_style_prompts(parsed, character_sheet)
