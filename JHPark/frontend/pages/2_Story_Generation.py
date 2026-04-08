"""Story generation page with 5-page book-style navigation."""

from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
STORY_TONE_OPTIONS = ["랜덤", "따뜻한", "모험적인", "교훈적인"]


def _display_image_payload(value: str, caption: str) -> None:
    """Render a generated page image from path, URL, or base64 payload."""

    if not value:
        st.info("이 페이지에는 아직 생성된 이미지가 없습니다.")
        return

    if value.startswith("data:image/"):
        _, encoded = value.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        st.image(Image.open(BytesIO(image_bytes)), caption=caption, width="stretch")
        return

    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / value

    if image_path.exists():
        st.image(str(image_path), caption=caption, width="stretch")
        return

    if value.startswith("http://") or value.startswith("https://"):
        st.image(value, caption=caption, width="stretch")
        return

    st.code(value)


def _call_story_generation(
    character_sheet: dict[str, Any],
    extra_prompt: str,
    story_tone: str | None,
    style_prompts: dict[str, Any] | None,
    reference_image: str | None,
) -> dict[str, Any]:
    """Call the story endpoint with optional illustration context."""

    response = requests.post(
        f"{FASTAPI_BASE_URL}/generate-story",
        json={
            "character_sheet": character_sheet,
            "extra_prompt": extra_prompt,
            "story_tone": story_tone,
            "style_prompts": style_prompts,
            "reference_image": reference_image,
        },
        timeout=1200,
    )
    response.raise_for_status()
    return response.json()


def _render_story_book(story_package: dict[str, Any]) -> None:
    """Render the story as a 5-page picture book with next/previous buttons."""

    pages = story_package.get("story_pages", [])
    if not pages:
        st.warning("페이지형 스토리 데이터가 없습니다.")
        return

    current_page = int(st.session_state.get("current_story_page", 0))
    current_page = max(0, min(current_page, len(pages) - 1))
    st.session_state.current_story_page = current_page
    page = pages[current_page]

    st.subheader(story_package.get("title", ""))
    st.caption(f"{current_page + 1} / {len(pages)} 페이지")

    page_left, page_right = st.columns([1.05, 0.95], gap="large")
    with page_left:
        _display_image_payload(page.get("image_path") or "", f"Page {page.get('page_number', current_page + 1)}")

    with page_right:
        st.markdown(f"### Page {page.get('page_number', current_page + 1)}")
        for sentence in page.get("sentences", []):
            st.write(sentence)

        with st.expander("이 페이지 이미지 프롬프트 보기", expanded=False):
            st.code(page.get("image_prompt", ""), language="text")

    nav_left, nav_center, nav_right = st.columns([1, 1, 1])
    with nav_left:
        if st.button("이전 페이지", width="stretch", disabled=current_page == 0):
            st.session_state.current_story_page = max(0, current_page - 1)
            st.rerun()
    with nav_center:
        st.markdown(
            f"<div style='text-align:center; padding-top:0.5rem;'>책장을 넘기듯 페이지를 이동하세요.</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button("다음 페이지", width="stretch", disabled=current_page == len(pages) - 1):
            st.session_state.current_story_page = min(len(pages) - 1, current_page + 1)
            st.rerun()

    st.divider()
    st.write("마지막 선택")
    choices = story_package.get("choices", [])
    if choices:
        button_columns = st.columns(len(choices))
        for idx, choice in enumerate(choices):
            with button_columns[idx]:
                if st.button(
                    choice.get("text", ""),
                    key=f"story_choice_{choice.get('id', idx)}",
                    width="stretch",
                ):
                    st.session_state.selected_story_choice = choice
        selected_choice = st.session_state.get("selected_story_choice")
        if selected_choice:
            st.info(f"선택한 이야기: {selected_choice['text']} ({selected_choice['id']})")


default_character_sheet = {
    "original_object": "toy bus",
    "name": "Coco",
    "job": "space explorer",
    "personality": "warm and brave",
    "goal": "wants to discover a new star",
    "core_visual_traits": ["round toy-bus body", "bright window eyes", "friendly hero silhouette"],
    "tone": "모험적인",
}

if "story_character_sheet_json" not in st.session_state:
    st.session_state.story_character_sheet_json = json.dumps(default_character_sheet, ensure_ascii=False, indent=2)
if "current_story_page" not in st.session_state:
    st.session_state.current_story_page = 0

st.set_page_config(page_title="Story Generation", page_icon="📖", layout="wide")

st.title("동화 생성")
st.caption("15문장, 5페이지 그림책 구조로 동화를 만들고 페이지별 이미지를 함께 확인할 수 있습니다.")

story_character_sheet_json = st.text_area(
    "character_sheet JSON",
    value=st.session_state.story_character_sheet_json,
    height=260,
    help="메인 페이지에서 생성한 character_sheet를 자동으로 가져옵니다. 필요하면 직접 수정할 수 있습니다.",
)

extra_prompt = st.text_area(
    "추가 요청",
    value="의성어와 의태어를 충분히 넣고, 아이가 다음 페이지가 궁금해지게 만들어줘.",
    height=120,
    help="스토리 분위기, 장면, 교육 포인트 등을 더하고 싶을 때 입력합니다.",
)

story_tone_option = st.selectbox(
    "동화 스타일 선택",
    STORY_TONE_OPTIONS,
    index=0,
    help="랜덤 또는 따뜻한, 모험적인, 교훈적인 스타일 중 하나를 선택할 수 있습니다.",
)

style_prompts = st.session_state.get("latest_style_prompts")
story_reference_image = st.session_state.get("latest_story_reference_image")

if story_reference_image:
    st.caption(f"스토리 일러스트 기준 캐릭터 이미지: {story_reference_image}")
else:
    st.warning("메인 페이지에서 캐릭터 이미지를 먼저 생성하면 같은 캐릭터/그림체로 페이지 이미지를 만들 수 있습니다.")

if st.button("동화와 페이지 이미지 생성", type="primary", width="stretch"):
    try:
        character_sheet = json.loads(story_character_sheet_json)
    except json.JSONDecodeError as exc:
        st.error(f"character_sheet JSON 오류: {exc}")
    else:
        try:
            with st.spinner("5페이지 그림책과 페이지별 이미지를 생성하는 중입니다..."):
                story_package = _call_story_generation(
                    character_sheet,
                    extra_prompt=extra_prompt,
                    story_tone=None if story_tone_option == "랜덤" else story_tone_option,
                    style_prompts=style_prompts,
                    reference_image=story_reference_image,
                )
            st.session_state.latest_story_package = story_package
            st.session_state.current_story_page = 0
            st.success("동화 생성이 완료되었습니다.")
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
    _render_story_book(st.session_state["latest_story_package"])
