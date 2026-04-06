"""Prompt builders for the Gemini and OpenAI calls."""

from __future__ import annotations

from typing import Any

from app.models import ALLOWED_STORY_TONES


def character_sheet_prompt(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> str:
    """Build a strict JSON prompt for character-sheet generation."""

    return f"""
You are creating a children's character sheet from vision data and parent guidance.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys:
  original_object, name, job, personality, goal, core_visual_traits, tone
- Keep the character age-appropriate, friendly, and appealing for children ages 6-8.
- Preserve the first detected object as the core identity of the character.
- Make the character visually consistent with the reference object.
- Favor bold, readable, toy-like visual traits over vague pastel decoration.
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
- Both prompts should feel suitable for children ages 6-8 and should make the character instantly recognizable as a toy-like story hero.
- Avoid washed-out pastel palettes; prefer richer, cleaner, child-friendly colors with strong shape readability.
- active_style should feel adventurous, energetic, dynamic, vivid, cinematic, with a dramatic 3/4 view, motion lines, a bold background, and strong hero-character presence.
- soft_style should feel warm, gentle, cozy, and emotionally tender, with a calm front-facing or seated pose, a dreamy but readable background, and a lovable character presence.
- Keep the character name, job, and key visual traits in both prompts.
- Write the prompts in English and make them directly usable for image generation.

Character sheet:
{character_sheet}
""".strip()


def story_generation_prompt(
    character_sheet: dict[str, Any],
    extra_prompt: str = "",
    story_tone: str | None = None,
) -> str:
    """Build a strict JSON prompt for story, TTS lines, and interactive choices."""

    tone_instruction = story_tone or "character_sheet의 tone을 유지하되, 지정되지 않았다면 허용된 tone 중 하나를 선택"
    extra_instruction = extra_prompt.strip() or "없음"

    return f"""
Create a short original children's story package from this character sheet.

Target reader:
- Children ages 6 to 8

Allowed tone values:
- {", ".join(ALLOWED_STORY_TONES)}

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys:
  title, story_paragraphs, tts_script, choices
- Use the existing main character only. Do not invent a new protagonist.
- Reflect the character's name, original_object, job, personality, goal, and tone.
- Use this story tone as the overall mood of the story and choices: {tone_instruction}
- story_paragraphs must contain exactly 3 short paragraphs in this order:
  introduction, action, ending
- Each paragraph must be 200 characters or fewer.
- Keep the total story simple and child-friendly.
- Include at most 2 core events in the story.
- Use easy Korean suitable for ages 6 to 8.
- Avoid abstract language, difficult vocabulary, and long sentences.
- title must be short and clear.
- tts_script must be a JSON array of sentence-level items.
- Each tts_script item must contain exactly these keys: line, tone
- TTS lines can be slightly shorter than the story, but must match the same content.
- The full tts_script must cover the entire story from beginning to end.
- The tts_script tone is NOT the same as the overall story tone.
- Each tts_script tone must describe how the line should be read aloud in narration.
- Use practical narration guide tones such as:
  따뜻한 톤, 설레는 톤, 신나는 톤, 조용한 톤, 비장한 톤, 다정한 톤, 놀라는 톤
- Change the TTS tone line by line when the scene changes.
- choices must contain exactly 2 items.
- Each choice item must contain exactly these keys: id, text
- choice ids must be snake_case English.
- choice text must be easy for a child to choose from after the ending.
- The 2 choices should work well as dropdown options after the story ends.
- Make the story, TTS script, and choices all consistent with the tone.

Additional user prompt:
{extra_instruction}

Character sheet:
{character_sheet}
""".strip()


def story_prompt(character_sheet: dict[str, Any], extra_prompt: str = "", story_tone: str | None = None) -> str:
    """Backward-compatible alias for story generation prompt building."""

    return story_generation_prompt(character_sheet, extra_prompt=extra_prompt, story_tone=story_tone)


def reference_image_prompt(reference_hint: str, style_label: str) -> str:
    """Build a prompt that turns a reference image into a reusable generation prompt."""

    style_rules = {
        "active_style": (
            "Make it feel adventurous, energetic, dynamic, vivid, cinematic, and appealing to children ages 6-8. "
            "Use a dramatic 3/4 angle, strong motion cues, richer child-friendly colors, and a bold background. "
            "Give the character a memorable toy-like silhouette and hero-character presence."
        ),
        "soft_style": (
            "Make it feel warm, gentle, cozy, and emotionally tender while still feeling like a memorable hero character for children ages 6-8. "
            "Use a calm pose, stylized texture, richer friendly colors instead of washed-out pastel tones, and a dreamy but readable background."
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
