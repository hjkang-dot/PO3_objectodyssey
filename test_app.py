import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000/extract"
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
REFERENCE_IMAGES_URL = f"{BACKEND_BASE_URL}/reference-images"
CREATE_ART_URL = f"{BACKEND_BASE_URL}/create-art"


def request_nukki_api(image_file):
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        response = requests.post(API_URL, files=files, timeout=60)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"Server error: {response.status_code}"}
    except Exception as exc:
        return {"status": "error", "message": f"Request failed: {exc}"}


def load_reference_images() -> list[str]:
    try:
        response = requests.get(REFERENCE_IMAGES_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("reference_images", []))
    except Exception as exc:
        st.error(f"기준 이미지 목록을 불러오지 못했습니다: {exc}")
        return []


def get_reference_preview_path(filename: str) -> Path:
    return ROOT_DIR / "nukki" / filename


def render_generated_image(value: str, caption: str) -> None:
    if not value:
        st.warning(f"{caption} 이미지가 없습니다.")
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


def call_create_art(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(CREATE_ART_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


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

st.title("Object Odyssey Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", width=300)

    if st.button("Run Extraction"):
        with st.spinner("Processing the uploaded image..."):
            result = request_nukki_api(uploaded_file)

        if result["status"] == "success":
            st.success(f"Detected target: {result['target']}")
            if "output_path" in result:
                st.image(result["output_path"], caption="Extraction result")
        else:
            st.error(result["message"])


st.divider()
st.title("Character Art Demo")
st.caption("이 페이지는 캐릭터/이미지 생성 전용입니다. 동화는 별도 페이지에서 생성합니다.")

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("Vision Input")
    objects_text = st.text_input(
        "Detected objects",
        value="곰인형",
        help="Comma-separated list of detected objects.",
    )

    st.subheader("Parent Input")
    art_name = st.text_input("Name", value="코코")
    art_job = st.text_input("Job", value="우주 탐험가")
    art_personality = st.text_input("Personality", value="용감하고 다정함")
    art_goal = st.text_input("Goal", value="새로운 별을 찾고 싶어함")
    art_tone = st.selectbox("Base tone", ["따뜻한", "모험적인", "교훈적인"], index=1)
    art_extra = st.text_area(
        "추가 프롬프트",
        value="작은 별가방과 반짝이는 우주복을 입고 있어요.",
        height=100,
        help="기존 내부 프롬프트는 숨겨져 있고, 여기 입력한 내용만 추가 반영됩니다.",
    )

with right_col:
    st.subheader("Reference Image")
    reference_images = load_reference_images()
    if not reference_images:
        st.error("No usable jpg, jpeg, or png images were found in the root nukki folder.")
        selected_reference = ""
    else:
        selected_reference = st.selectbox("Reference image", reference_images, index=0)
        preview_path = get_reference_preview_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"Reference image: {selected_reference}", use_container_width=True)

run_create_art = st.button("Generate Character Art", type="primary", use_container_width=True)

if run_create_art:
    objects = [item.strip() for item in objects_text.split(",") if item.strip()]
    if not selected_reference:
        st.error("Select a reference image first.")
    else:
        payload = {
            "vision_result": {"objects": objects},
            "parent_input": {
                "name": art_name,
                "job": art_job,
                "personality": art_personality,
                "goal": art_goal,
                "extra_description": art_extra,
                "tone": art_tone,
            },
            "reference_image": selected_reference,
        }

        try:
            with st.spinner("Generating character art..."):
                result = call_create_art(payload)

            st.success("Character art generation completed.")

            left_result, right_result = st.columns([1, 1], gap="large")
            with left_result:
                st.subheader("Character Sheet")
                st.json(result.get("character_sheet", {}))
                st.subheader("Style Prompts")
                style_prompts = result.get("style_prompts", {})
                st.write("Active Style")
                st.code(style_prompts.get("active_style", ""))
                st.write("Soft Style")
                st.code(style_prompts.get("soft_style", ""))

            with right_result:
                st.subheader("Generated Images")
                generated_images = result.get("generated_images", {})
                render_generated_image(generated_images.get("active_style", ""), "Active Style")
                render_generated_image(generated_images.get("soft_style", ""), "Soft Style")

            if result.get("character_sheet"):
                st.session_state.story_character_sheet_json = json.dumps(
                    result["character_sheet"], ensure_ascii=False, indent=2
                )
            if result.get("story"):
                st.session_state.latest_story_package = result["story"]

            st.info("Story Generation 페이지에서 이어서 동화를 생성할 수 있습니다.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"백엔드 오류: {detail or exc}")
        except Exception as exc:
            st.error(f"실행 중 오류가 발생했습니다: {exc}")
