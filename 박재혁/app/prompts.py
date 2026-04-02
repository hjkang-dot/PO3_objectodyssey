"""Prompt builders for the Gemini and OpenAI calls."""

from __future__ import annotations

from typing import Any


def character_sheet_prompt(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> str:
    """Build a strict JSON prompt for character-sheet generation."""

    return f"""
You are creating a children's character sheet from vision data and parent guidance.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys:
  original_object, name, job, personality, goal, core_visual_traits, tone
- Keep the character age-appropriate, friendly, and easy for ages 4-6 to understand.
- Preserve the first detected object as the core identity of the character.
- Make the character visually consistent with the reference object.
- core_visual_traits must be a JSON array of 2 to 5 short strings.
- tone must be a short descriptive phrase.

Vision result:
{vision_result}

Parent input:
{parent_input}
""".strip()


def style_prompts_prompt(character_sheet: dict[str, Any]) -> str:
    """Build a prompt that yields two English image prompts for the same character."""

    return f"""
Create two English image prompts for the same character.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys: active_style, soft_style
- Both prompts must preserve the same character identity, face, proportions, and core outfit details.
- Both prompts must explicitly say the reference image is only for identity cues and should be reinterpreted as a fresh illustration.
- The prompts must ask for a full redraw or character redesign, not a simple filter, recolor, retouch, or texture overlay.
- The prompts must tell the model to preserve recognizable identity cues while replacing the original photo look with a new illustrated scene.
- active_style should feel adventurous, energetic, dynamic, vivid, cinematic, with a dramatic 3/4 view, motion lines, and a bold background.
- soft_style should feel warm, gentle, pastel, cozy, and emotionally tender, with a calm front-facing or seated pose and a soft dreamy background.
- Keep the character name, job, and key visual traits in both prompts.
- Write the prompts in English and make them directly usable for image generation.

Character sheet:
{character_sheet}
""".strip()


def story_prompt(character_sheet: dict[str, Any]) -> str:
    """Build a prompt for a JSON-formatted children's story."""

    return f"""
Write a short children's story for ages 4-6.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys: title, story
- story must be a JSON array with 4 to 6 short sentences.
- Use simple words, warm feelings, and a positive ending.
- The character's name, original_object, job, personality, and goal must all appear.
- Make the story easy to read aloud.

Character sheet:
{character_sheet}
""".strip()


def reference_image_prompt(reference_hint: str, style_label: str) -> str:
    """Build a prompt that turns a reference image into a reusable generation prompt."""

    style_rules = {
        "active_style": (
            "Make it feel adventurous, energetic, dynamic, vivid, and cinematic. "
            "Use a dramatic 3/4 angle, strong motion cues, and a bold background."
        ),
        "soft_style": (
            "Make it feel warm, gentle, pastel, cozy, and emotionally tender. "
            "Use a calm pose, soft brush texture, and a dreamy background."
        ),
    }

    return f"""
You are converting a reference image into a detailed English image-generation prompt.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys: prompt, reference_description, key_visual_facts
- Describe the visual identity cues that should survive a full character redesign.
- Do not describe the image as something to filter, edit, retouch, or lightly modify.
- Focus on a prompt that another image model can use to turn the subject into a fresh character illustration.
- {style_rules.get(style_label, style_rules["active_style"])}
- The prompt must be a single English paragraph.

Reference hint:
{reference_hint}
""".strip()
