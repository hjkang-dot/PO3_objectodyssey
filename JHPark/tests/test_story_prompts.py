from app.prompts import (
    build_adventure_story_style,
    build_lesson_story_style,
    build_story_style_guide,
    build_warm_story_style,
    story_generation_prompt,
)


CHARACTER_SHEET = {
    "original_object": "장난감 기차",
    "name": "코코",
    "job": "별빛 배달원",
    "personality": "용감하고 다정함",
    "goal": "밤하늘 친구들에게 별빛 전하기",
    "core_visual_traits": ["파란 기차 몸체", "별 모양 전조등"],
    "tone": "따뜻한",
}


def test_story_style_builders_include_distinct_traits() -> None:
    assert "위로, 배려, 안심" in build_warm_story_style(CHARACTER_SHEET)
    assert "출발, 발견, 해결" in build_adventure_story_style(CHARACTER_SHEET)
    assert "약속, 책임, 협동, 정직, 배려" in build_lesson_story_style(CHARACTER_SHEET)


def test_story_style_guide_returns_selected_style() -> None:
    adventure = build_story_style_guide(CHARACTER_SHEET, "모험적인", extra_prompt="별 언덕에 가기")
    assert "활기차고 호기심이 살아 있는" in adventure
    assert "별 언덕에 가기" in adventure


def test_story_generation_prompt_contains_style_guide() -> None:
    prompt = story_generation_prompt(CHARACTER_SHEET, extra_prompt="친구를 만나기", story_tone="교훈적인")
    assert "쉽고 분명한 배움을 담은" in prompt
    assert "친구를 만나기" in prompt
    assert "title, story_paragraphs, tts_script, choices" in prompt
