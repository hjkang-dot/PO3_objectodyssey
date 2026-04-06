"""메인 캐릭터/이미지 생성 페이지."""

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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")


def _reference_image_path(filename: str) -> Path:
    """nukki 내부의 기준 이미지 경로를 만든다."""

    return ROOT_DIR / "nukki" / filename


def _load_reference_images() -> list[str]:
    """백엔드에서 기준 이미지 목록을 불러온다."""

    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/reference-images", timeout=20)
        response.raise_for_status()
        data = response.json()
        return list(data.get("reference_images", []))
    except Exception as exc:
        st.error(f"기준 이미지 목록을 불러오지 못했습니다: {exc}")
        return []


def _display_image_payload(value: str, caption: str) -> None:
    """경로 또는 base64 형태의 이미지를 화면에 보여준다."""

    if not value:
        st.warning(f"{caption} 이미지가 비어 있습니다.")
        return

    if value.startswith("data:image/"):
        _, encoded = value.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        st.image(Image.open(BytesIO(image_bytes)), caption=caption, use_container_width=True)
        return

    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / value

    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
        return

    if value.startswith("http://") or value.startswith("https://"):
        st.image(value, caption=caption, use_container_width=True)
        return

    st.code(value)


def _call_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """캐릭터, 이미지, 동화를 함께 생성하는 파이프라인을 호출한다."""

    response = requests.post(f"{FASTAPI_BASE_URL}/pipeline", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


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

st.set_page_config(page_title="PO3 Character Studio", page_icon="OO", layout="wide")

st.title("캐릭터/이미지 생성")
st.caption("이 페이지에서는 캐릭터 시트와 이미지를 생성합니다. 동화 생성은 별도 페이지에서 진행합니다.")

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("1. 비전 입력")
    objects_text = st.text_input("감지 객체", value="곰인형", help="쉼표로 여러 객체를 입력할 수 있습니다.")

    st.subheader("2. 부모 입력")
    name = st.text_input("이름", value="코코")
    job = st.text_input("직업", value="우주 탐험가")
    personality = st.text_input("성격", value="용감하고 다정함")
    goal = st.text_input("목표", value="새로운 별을 찾고 싶어함")
    tone = st.selectbox("기본 톤", ["따뜻한", "모험적인", "교훈적인"], index=1)
    extra_description = st.text_area(
        "추가 프롬프트",
        value="작은 별가방과 반짝이는 우주복을 입고 있어요.",
        height=100,
        help="기존 내부 프롬프트는 화면에 보이지 않으며, 여기에 적은 내용만 추가 반영됩니다.",
    )

with col_right:
    st.subheader("3. 기준 이미지")
    reference_images = _load_reference_images()
    if not reference_images:
        st.error("nukki 폴더에서 사용할 수 있는 이미지가 없습니다.")
        selected_reference = ""
    else:
        selected_reference = st.selectbox("기준 이미지 선택", reference_images, index=0)
        preview_path = _reference_image_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"기준 이미지: {selected_reference}", use_container_width=True)

run_clicked = st.button("캐릭터/이미지 생성", type="primary", use_container_width=True)

if run_clicked:
    objects = [item.strip() for item in objects_text.split(",") if item.strip()]
    if not selected_reference:
        st.error("기준 이미지를 먼저 선택해 주세요.")
    else:
        payload = {
            "vision_result": {"objects": objects},
            "parent_input": {
                "name": name,
                "job": job,
                "personality": personality,
                "goal": goal,
                "extra_description": extra_description,
                "tone": tone,
            },
            "reference_image": selected_reference,
        }

        try:
            with st.spinner("캐릭터와 이미지를 생성하는 중입니다..."):
                result = _call_pipeline(payload)

            result["character_sheet"]["tone"] = tone
            st.session_state.story_character_sheet_json = json.dumps(
                result["character_sheet"], ensure_ascii=False, indent=2
            )
            st.session_state.latest_story_package = result.get("story", {})

            st.success("캐릭터와 이미지 생성이 완료되었습니다.")

            left, right = st.columns([1, 1], gap="large")
            with left:
                st.subheader("캐릭터 시트")
                st.json(result["character_sheet"])
                st.subheader("스타일 프롬프트")
                st.write("활동형")
                st.code(result["style_prompts"]["active_style"])
                st.write("부드러운형")
                st.code(result["style_prompts"]["soft_style"])

            with right:
                st.subheader("생성 이미지")
                _display_image_payload(result["generated_images"]["active_style"], "활동형")
                _display_image_payload(result["generated_images"]["soft_style"], "부드러운형")

            st.info("동화 생성은 왼쪽 사이드바의 Story Generation 페이지에서 이어서 진행할 수 있습니다.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"백엔드 오류: {detail or exc}")
        except Exception as exc:
            st.error(f"실행 중 오류가 발생했습니다: {exc}")
