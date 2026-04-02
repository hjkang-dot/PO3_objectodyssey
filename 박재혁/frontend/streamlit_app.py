"""Streamlit demo UI for the PO3 prototype."""

from __future__ import annotations

import base64
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
    """Resolve a selected reference image path inside nukki."""

    return ROOT_DIR / "nukki" / filename


def _load_reference_images() -> list[str]:
    """Load the reference image list from the backend."""

    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/reference-images", timeout=20)
        response.raise_for_status()
        data = response.json()
        return list(data.get("reference_images", []))
    except Exception as exc:
        st.error(f"기준 이미지 목록을 불러오지 못했습니다: {exc}")
        return []


def _display_image_payload(value: str, caption: str) -> None:
    """Render a generated image regardless of whether it is a path or base64."""

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
    """Call the backend pipeline endpoint."""

    response = requests.post(f"{FASTAPI_BASE_URL}/pipeline", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="PO3 Object Odyssey", page_icon="✨", layout="wide")

st.title("PO3 Object Odyssey Prototype")
st.caption("FastAPI + Streamlit demo for character sheet, styled images, and a children's story.")

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("1. 비전 결과 입력")
    objects_text = st.text_input("객체 이름들", value="곰인형", help="쉼표로 여러 객체를 입력하세요. 예: 곰인형, 컵, 책")

    st.subheader("2. 부모 프롬프트 입력")
    name = st.text_input("이름", value="코코")
    job = st.text_input("직업", value="우주 탐험가")
    personality = st.text_input("성격", value="용감하고 다정함")
    goal = st.text_input("하고 싶은 것", value="새로운 별을 찾고 싶어함")
    extra_description = st.text_area(
        "추가 설명",
        value="작은 별 모양 가방을 메고 다녔으면 좋겠어",
        height=100,
    )

with col_right:
    st.subheader("3. 기준 이미지 선택")
    reference_images = _load_reference_images()
    if not reference_images:
        st.error("박재혁/nukki 폴더에서 사용할 수 있는 jpg, jpeg, png 이미지가 없습니다.")
        selected_reference = ""
    else:
        selected_reference = st.selectbox("기준 이미지", reference_images, index=0)
        preview_path = _reference_image_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"기준 이미지: {selected_reference}", use_container_width=True)

st.divider()

run_clicked = st.button("캐릭터 생성 시작", type="primary", use_container_width=True)

if run_clicked:
    objects = [item.strip() for item in objects_text.split(",") if item.strip()]
    if not selected_reference:
        st.error("기준 이미지를 선택해야 합니다.")
    else:
        payload = {
            "vision_result": {"objects": objects},
            "parent_input": {
                "name": name,
                "job": job,
                "personality": personality,
                "goal": goal,
                "extra_description": extra_description,
            },
            "reference_image": selected_reference,
        }

        try:
            with st.spinner("캐릭터, 이미지, 동화를 생성하는 중..."):
                result = _call_pipeline(payload)

            st.success("생성이 완료되었습니다.")

            left, right = st.columns([1, 1], gap="large")
            with left:
                st.subheader("Character Sheet")
                st.json(result["character_sheet"])
                st.subheader("Style Prompts")
                st.write("Active Style")
                st.code(result["style_prompts"]["active_style"])
                st.write("Soft Style")
                st.code(result["style_prompts"]["soft_style"])

            with right:
                st.subheader("Generated Images")
                _display_image_payload(result["generated_images"]["active_style"], "Active Style")
                _display_image_payload(result["generated_images"]["soft_style"], "Soft Style")

            st.subheader("Story")
            st.markdown(f"### {result['story']['title']}")
            for sentence in result["story"]["story"]:
                st.write(f"- {sentence}")

            st.subheader("Raw Pipeline Result")
            st.json(result)
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"백엔드 오류: {detail or exc}")
        except Exception as exc:
            st.error(f"실행 중 오류가 발생했습니다: {exc}")

