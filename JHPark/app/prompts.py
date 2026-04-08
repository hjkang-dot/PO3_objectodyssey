"""프롬프트 관련 공개 진입점.

스토리북 관련 실제 구현은 `app.storybook_core`에 모아 두고,
이 파일은 기존 import 경로를 유지하기 위한 호환 레이어 역할을 한다.
"""

from __future__ import annotations

from typing import Any

from app.storybook_core import (
    build_adventure_story_style,
    build_common_story_rules,
    build_lesson_story_style,
    build_story_style_guide,
    build_warm_story_style,
    format_character_sheet,
    normalize_extra_prompt,
    story_generation_prompt,
    story_prompt,
)


def character_sheet_prompt(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> str:
    """비전 결과와 사용자 입력을 바탕으로 캐릭터 시트를 만들기 위한 프롬프트."""

    return f"""
You are creating a children's character sheet from vision data and parent guidance.

Rules:
- Return JSON only. No markdown, no code fences, no explanation.
- The JSON must contain exactly these keys:
  original_object, name, job, personality, goal, core_visual_traits, tone
- Keep the character age-appropriate, friendly, and appealing for children ages 6-8.
- Preserve the first detected object as the core identity of the character unless parent_input provides a clearer original_object_hint.
- Make the character visually consistent with the reference object.
- Favor bold, readable, toy-like visual traits over vague decoration.
- core_visual_traits must be a JSON array of 2 to 5 short strings.
- tone must be a short descriptive phrase.

Vision result:
{vision_result}

Parent input:
{parent_input}
""".strip()


def reference_image_prompt(reference_hint: str, style_label: str) -> str:
    """레퍼런스 이미지를 캐릭터 이미지 생성용 prompt seed로 바꾸는 프롬프트."""

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


__all__ = [
    "build_adventure_story_style",
    "build_common_story_rules",
    "build_lesson_story_style",
    "build_story_style_guide",
    "build_warm_story_style",
    "character_sheet_prompt",
    "format_character_sheet",
    "normalize_extra_prompt",
    "reference_image_prompt",
    "story_generation_prompt",
    "story_prompt",
]
