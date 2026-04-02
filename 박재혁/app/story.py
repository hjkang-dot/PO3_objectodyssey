"""Story generation helpers."""

from __future__ import annotations

from typing import Any

from app.prompts import story_prompt
from app.services.openai_service import OpenAIService
from app.utils import normalize_story_list, safe_json_loads


def fallback_story(character_sheet: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic child-friendly story when OpenAI is unavailable."""

    name = character_sheet["name"]
    original_object = character_sheet["original_object"]
    job = character_sheet["job"]
    personality = character_sheet["personality"]
    goal = character_sheet["goal"]

    title = f"{name} and the Little {original_object}"
    story = [
        f"{name} was a kind {job} with a brave heart.",
        f"{name} looked at the little {original_object} and smiled.",
        f"Together, they set out to reach a dream that matched the goal: {goal}.",
        f"Because {name} was {personality}, the journey felt safe and fun.",
        f"In the end, {name} found a new friend and a happy little star to follow home.",
    ]
    return {"title": title, "story": story}


def validate_story(data: dict[str, Any], character_sheet: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the story output."""

    fallback = fallback_story(character_sheet)
    merged = {**fallback, **data}
    title = str(merged.get("title") or fallback["title"]).strip()
    story = normalize_story_list(merged.get("story"))
    if len(story) < 4:
        story = fallback["story"]
    return {"title": title, "story": story[:6]}


def build_story(character_sheet: dict[str, Any], openai_service: OpenAIService) -> dict[str, Any]:
    """Generate a short children's story from the final character sheet."""

    prompt = story_prompt(character_sheet)
    try:
        raw_text = openai_service.generate_story_json(prompt)
        parsed = safe_json_loads(raw_text)
    except Exception:
        parsed = {}
    return validate_story(parsed, character_sheet)

