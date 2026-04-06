import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
GENERATE_STORY_URL = f"{BACKEND_BASE_URL}/generate-story"
STORY_TONE_OPTIONS = ["랜덤", "따뜻한", "모험적인", "교훈적인"]


def call_generate_story(
    character_sheet: dict[str, Any],
    extra_prompt: str,
    story_tone: str | None,
) -> dict[str, Any]:
    """동화 생성 엔드포인트를 호출한다."""

    response = requests.post(
        GENERATE_STORY_URL,
        json={
            "character_sheet": character_sheet,
            "extra_prompt": extra_prompt,
            "story_tone": story_tone,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_story_package(story_result: dict[str, Any]) -> None:
    """동화 생성 결과를 화면에 출력한다."""

    st.subheader(story_result.get("title", ""))
    for paragraph in story_result.get("story_paragraphs", []):
        st.write(paragraph)

    choices = story_result.get("choices", [])
    if choices:
        st.write("다음 행동")
        button_columns = st.columns(len(choices))
        for idx, choice in enumerate(choices):
            with button_columns[idx]:
                if st.button(choice.get("text", ""), key=f"story_choice_{choice.get('id', idx)}", use_container_width=True):
                    st.session_state.selected_story_choice = choice
        selected_choice = st.session_state.get("selected_story_choice")
        if selected_choice:
            st.info(f"선택한 행동: {selected_choice['text']} ({selected_choice['id']})")


default_story_character_sheet = {
    "original_object": "곰인형",
    "name": "코코",
    "job": "우주 탐험가",
    "personality": "용감하고 다정함",
    "goal": "새로운 별을 찾고 싶어함",
    "core_visual_traits": ["작은 별가방", "반짝이는 우주복"],
    "tone": "모험적인",
}

if "story_character_sheet_json" not in st.session_state:
    st.session_state.story_character_sheet_json = json.dumps(
        default_story_character_sheet, ensure_ascii=False, indent=2
    )

st.title("Story Generation Demo")
st.caption("기존 캐릭터 이미지는 유지하고, 이 페이지에서 동화만 따로 생성합니다.")

story_character_sheet_json = st.text_area(
    "character_sheet JSON",
    value=st.session_state.story_character_sheet_json,
    height=260,
)

extra_prompt = st.text_area(
    "추가 프롬프트",
    value="",
    height=120,
    help="기존 프롬프트 내용은 숨겨져 있고, 여기 적은 추가 정보만 반영됩니다.",
)

story_tone_option = st.selectbox(
    "동화 분위기",
    STORY_TONE_OPTIONS,
    index=0,
    help="이 값은 동화 전체 분위기를 뜻합니다. TTS 읽기 톤은 문장별로 따로 생성됩니다.",
)

if st.button("동화 생성", type="primary", use_container_width=True):
    try:
        character_sheet = json.loads(story_character_sheet_json)
    except json.JSONDecodeError as exc:
        st.error(f"character_sheet JSON 오류: {exc}")
    else:
        try:
            with st.spinner("동화를 생성하는 중입니다..."):
                story_result = call_generate_story(
                    character_sheet,
                    extra_prompt=extra_prompt,
                    story_tone=None if story_tone_option == "랜덤" else story_tone_option,
                )

            st.session_state.latest_story_package = story_result
            st.success("동화 생성이 완료되었습니다.")
            render_story_package(story_result)
            st.info("TTS Script 페이지에서 전체 대본을 확인할 수 있습니다.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"백엔드 오류: {detail or exc}")
        except Exception as exc:
            st.error(f"실행 중 오류가 발생했습니다: {exc}")

if st.session_state.get("latest_story_package"):
    st.divider()
    st.caption("가장 최근 생성 결과")
    render_story_package(st.session_state["latest_story_package"])
