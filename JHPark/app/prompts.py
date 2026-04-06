"""Gemini와 OpenAI 호출에 사용하는 프롬프트 빌더 모음."""

from __future__ import annotations

from typing import Any

from app.models import ALLOWED_STORY_TONES


def character_sheet_prompt(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> str:
    """캐릭터 시트 생성을 위한 엄격한 JSON 프롬프트를 만든다."""

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
    """같은 캐릭터의 두 가지 이미지 스타일 프롬프트를 생성한다."""

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


def build_common_story_rules() -> str:
    """모든 동화 스타일에 공통으로 적용할 규칙을 반환한다."""

    return """
[공통 규칙]
- 반드시 JSON만 반환한다. 마크다운, 코드블록, 설명문은 금지한다.
- JSON 키는 정확히 다음 4개만 사용한다:
  title, story_paragraphs, tts_script, choices
- 주인공은 character_sheet의 기존 캐릭터 한 명만 사용한다.
- 캐릭터의 name, original_object, job, personality, goal 정보를 이야기 안에 자연스럽게 반영한다.
- 대상 독자는 6세에서 8세 아동이다.
- story_paragraphs는 정확히 3개 문단이어야 한다.
- 문단 순서는 도입, 전개, 마무리다.
- 각 문단은 200자 이하의 쉬운 한국어로 작성한다.
- 이야기 전체에는 핵심 사건을 최대 2개까지만 넣는다.
- 어렵거나 추상적인 표현, 긴 문장은 피한다.
- title은 짧고 분명하게 쓴다.
- tts_script는 문장 단위 배열이다.
- 각 tts_script 항목은 정확히 line, tone 키만 가진다.
- tts_script는 이야기 처음부터 끝까지 빠짐없이 덮어야 한다.
- tts_script의 tone은 동화 스타일명이 아니라 낭독 방식이어야 한다.
- 예시 tone: 또박또박, 신나게, 조용하게, 비밀스럽게, 다정하게, 긴장감 있게
- 장면이 바뀌면 tts_script의 tone도 적절히 바꾼다.
- choices는 정확히 2개여야 한다.
- 각 choice 항목은 정확히 id, text 키만 가진다.
- choice id는 snake_case 영어로 작성한다.
- choice text는 아이가 이야기 끝에서 쉽게 고를 수 있게 짧고 분명하게 쓴다.
""".strip()


def normalize_extra_prompt(extra_prompt: str) -> str:
    """추가 요청 문구를 정리한다."""

    return extra_prompt.strip() or "없음"


def format_character_sheet(character_sheet: dict[str, Any]) -> str:
    """프롬프트에 넣기 좋은 형태로 캐릭터 정보를 정리한다."""

    return f"""
[캐릭터 정보]
{character_sheet}
""".strip()


def build_warm_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """따뜻한 스타일 동화를 위한 지침을 반환한다."""

    return f"""
[스타일]
- 전체 분위기는 다정하고 포근한 "따뜻한" 스타일로 만든다.
- 갈등보다 위로, 배려, 안심, 작은 기쁨을 중심으로 전개한다.
- 배경은 집, 마을, 공원, 해질녘 길처럼 편안한 공간을 우선 활용한다.
- 사건은 놀라움보다 정서적 공감이 잘 느껴지게 구성한다.
- 마지막은 아이가 마음이 편안해지는 결말로 마무리한다.
- choices도 다정한 다음 행동처럼 느껴지게 만든다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_adventure_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """모험적인 스타일 동화를 위한 지침을 반환한다."""

    return f"""
[스타일]
- 전체 분위기는 활기차고 호기심이 살아 있는 "모험적인" 스타일로 만든다.
- 출발, 발견, 해결의 흐름이 분명하게 느껴지도록 쓴다.
- 배경은 숲길, 하늘길, 바닷가, 우주 정거장처럼 상상력이 살아나는 장소를 활용한다.
- 위험을 과하게 키우지 말고, 아이가 즐겁게 따라갈 수 있는 수준의 긴장감만 준다.
- 주인공이 자신의 장점으로 문제를 해결하는 장면을 꼭 넣는다.
- 마지막은 성취감과 다음 탐험의 기대가 남도록 마무리한다.
- choices도 탐험을 이어 가는 선택처럼 보이게 만든다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_lesson_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """교훈적인 스타일 동화를 위한 지침을 반환한다."""

    return f"""
[스타일]
- 전체 분위기는 쉽고 분명한 배움을 담은 "교훈적인" 스타일로 만든다.
- 설교처럼 딱딱하게 쓰지 말고, 사건을 통해 자연스럽게 깨달음을 보여 준다.
- 약속, 책임, 협동, 정직, 배려 중 하나가 드러나도록 구성한다.
- 교훈은 문장으로 직접 설명하기보다 행동과 결과로 이해되게 만든다.
- 주인공이 작은 실수나 고민을 겪고 스스로 더 나은 선택을 하게 만든다.
- 마지막은 뿌듯함과 배움이 함께 남도록 마무리한다.
- choices도 배운 점을 이어 가거나 실천해 보는 방향으로 만든다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_story_style_guide(character_sheet: dict[str, Any], story_tone: str, extra_prompt: str = "") -> str:
    """선택한 스타일에 맞는 동화 지침을 반환한다."""

    style_map = {
        "따뜻한": build_warm_story_style,
        "모험적인": build_adventure_story_style,
        "교훈적인": build_lesson_story_style,
    }
    guide_builder = style_map.get(story_tone, build_warm_story_style)
    return guide_builder(character_sheet, extra_prompt=extra_prompt)


def story_generation_prompt(
    character_sheet: dict[str, Any],
    extra_prompt: str = "",
    story_tone: str | None = None,
) -> str:
    """동화, TTS, 선택지를 한 번에 생성하는 한국어 프롬프트를 만든다."""

    selected_style = story_tone or character_sheet.get("tone") or ALLOWED_STORY_TONES[0]
    style_guide = build_story_style_guide(character_sheet, selected_style, extra_prompt=extra_prompt)

    return f"""
너는 6세에서 8세 아이를 위한 참여형 동화를 만드는 작가야.

[허용 스타일]
- {", ".join(ALLOWED_STORY_TONES)}

{build_common_story_rules()}

{style_guide}

[출력 점검]
- story_paragraphs, tts_script, choices가 서로 같은 내용을 가리키도록 일관성을 맞춘다.
- 캐릭터 이름이 이야기나 TTS에 반드시 드러나야 한다.
- 이야기 끝을 읽은 뒤 아이가 바로 choices를 고르고 싶어지게 만든다.
""".strip()


def story_prompt(character_sheet: dict[str, Any], extra_prompt: str = "", story_tone: str | None = None) -> str:
    """기존 호출부와 호환되는 별칭 함수."""

    return story_generation_prompt(character_sheet, extra_prompt=extra_prompt, story_tone=story_tone)


def reference_image_prompt(reference_hint: str, style_label: str) -> str:
    """참조 이미지를 재사용 가능한 생성 프롬프트로 바꾸는 프롬프트를 만든다."""

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
