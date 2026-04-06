"""동화 생성 전용 페이지."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
STORY_TONE_OPTIONS = ["랜덤", "따뜻한", "모험적인", "교훈적인"]


def _call_story_generation(
    character_sheet: dict[str, Any],
    extra_prompt: str,
    story_tone: str | None,
) -> dict[str, Any]:
    """동화 생성 엔드포인트를 호출한다."""

    response = requests.post(
        f"{FASTAPI_BASE_URL}/generate-story",
        json={
            "character_sheet": character_sheet,
            "extra_prompt": extra_prompt,
            "story_tone": story_tone,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _render_story_package(story_package: dict[str, Any]) -> None:
    """동화 생성 결과를 보기 좋게 출력한다."""

    st.subheader(story_package.get("title", ""))

    st.write("동화")
    for idx, paragraph in enumerate(story_package.get("story_paragraphs", []), start=1):
        st.write(f"{idx}. {paragraph}")

    st.write("다음 행동")
    choices = story_package.get("choices", [])
    if choices:
        button_columns = st.columns(len(choices))
        for idx, choice in enumerate(choices):
            with button_columns[idx]:
                if st.button(choice.get("text", ""), key=f"story_choice_{choice.get('id', idx)}", use_container_width=True):
                    st.session_state.selected_story_choice = choice
        selected_choice = st.session_state.get("selected_story_choice")
        if selected_choice:
            st.info(f"선택한 행동: {selected_choice['text']} ({selected_choice['id']})")


default_character_sheet = {
    "original_object": "곰인형",
    "name": "코코",
    "job": "우주 탐험가",
    "personality": "용감하고 다정함",
    "goal": "새로운 별을 찾고 싶어함",
    "core_visual_traits": ["작은 별가방", "반짝이는 우주복"],
    "tone": "모험적인",
}

if "story_character_sheet_json" not in st.session_state:
    st.session_state.story_character_sheet_json = json.dumps(default_character_sheet, ensure_ascii=False, indent=2)

st.set_page_config(page_title="Story Generation", page_icon="OO", layout="wide")

st.title("동화 생성")
st.caption("기존 캐릭터 이미지는 유지하고, 이 페이지에서 동화만 따로 생성합니다.")

story_character_sheet_json = st.text_area(
    "character_sheet JSON",
    value=st.session_state.story_character_sheet_json,
    height=260,
    help="메인 페이지에서 생성한 character_sheet를 자동으로 가져오며, 필요하면 직접 수정할 수 있습니다.",
)

extra_prompt = st.text_area(
    "추가 프롬프트",
    value="",
    height=120,
    help="기존 프롬프트 내용은 숨겨져 있습니다. 여기에 적은 추가 정보만 동화에 반영됩니다.",
)

story_tone_option = st.selectbox(
    "동화 분위기 선택",
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
                story_package = _call_story_generation(
                    character_sheet,
                    extra_prompt=extra_prompt,
                    story_tone=None if story_tone_option == "랜덤" else story_tone_option,
                )
            st.session_state.latest_story_package = story_package
            st.success("동화 생성이 완료되었습니다.")
            _render_story_package(story_package)
            st.info("TTS Script 페이지에서 전체 대본을 바로 확인할 수 있습니다.")
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
    _render_story_package(st.session_state["latest_story_package"])
