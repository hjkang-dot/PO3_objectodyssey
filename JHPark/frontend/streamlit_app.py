"""Main Streamlit page for prompt testing and image generation."""

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

from app.character import build_prompt_preview, category_options_for_gender, normalize_prompt_options

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
GENDER_LABELS = {"boy": "Boys' Preference", "girl": "Girls' Preference"}
BASE_STYLE_LABELS = {"active": "Active", "soft": "Soft"}
CATEGORY_LABELS = {
    "default": "None / Default",
    "adventure": "Adventure",
    "cozy": "Cozy",
    "magic": "Magic",
    "bright": "Bright",
    "fantasy": "Fantasy",
}


def _reference_image_path(filename: str) -> Path:
    """Build the preview path for a selectable reference image."""

    return ROOT_DIR / "nukki" / filename


def _load_reference_images() -> list[str]:
    """Load available reference image filenames from the backend."""

    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/reference-images", timeout=20)
        response.raise_for_status()
        data = response.json()
        return list(data.get("reference_images", []))
    except Exception as exc:
        st.error(f"Failed to load reference images: {exc}")
        return []


def _display_image_payload(value: str, caption: str) -> None:
    """Render a generated image from a path, URL, or base64 payload."""

    if not value:
        st.warning(f"{caption} image is empty.")
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


def _call_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the backend end-to-end pipeline."""

    response = requests.post(f"{FASTAPI_BASE_URL}/pipeline", json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _build_preview_character_sheet(
    original_object: str,
    name: str,
    job: str,
    traits_input: str,
    personality: str,
    goal: str,
    tone: str,
) -> dict[str, Any]:
    """Build a local preview-only character sheet from the form values."""

    traits = [item.strip() for item in traits_input.split(",") if item.strip()]
    fallback_traits = traits or [
        f"based on the appearance of {original_object}",
        f"wears visual hints of the job: {job}",
        "keeps a toy-like silhouette with readable facial features",
    ]
    return {
        "original_object": original_object.strip() or "reference object",
        "name": name.strip() or "Coco",
        "job": job.strip() or "storybook hero",
        "personality": personality.strip() or "warm and brave",
        "goal": goal.strip() or "wants to discover something new",
        "core_visual_traits": fallback_traits[:5],
        "tone": tone,
    }


def _build_prompt_summary(prompt_options: dict[str, str]) -> dict[str, str]:
    """Create a small display-friendly summary for the selected prompt controls."""

    return {
        "target_audience": GENDER_LABELS[prompt_options["gender"]],
        "base_style": BASE_STYLE_LABELS[prompt_options["base_style"]],
        "category": CATEGORY_LABELS[prompt_options["category"]],
    }


default_character_sheet = {
    "original_object": "",
    "name": "코코",
    "job": "탐험가",
    "personality": "다정하고 용감한",
    "goal": "새로운 유적을 발견하고 싶어함",
    "core_visual_traits": ["round toy body", "bright window eyes", "friendly hero silhouette", "anime style"],
    "tone": "warm",
}

if "story_character_sheet_json" not in st.session_state:
    st.session_state.story_character_sheet_json = json.dumps(default_character_sheet, ensure_ascii=False, indent=2)

st.set_page_config(page_title="PO3 Prompt Studio", page_icon="OO", layout="wide")

st.title("PO3 Prompt Studio")
st.caption("Choose a target audience preference, base prompt, and category, preview the final prompt, and optionally run image generation.")

reference_images = _load_reference_images()
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Prompt Controls")

    gender = st.selectbox(
        "Target Audience",
        options=["girl", "boy"],
        format_func=lambda value: GENDER_LABELS[value],
        index=0,
        help="This does not force the character to be male or female. It selects the visual/story presentation style that may appeal more to girls or boys.",
    )
    base_style = st.selectbox(
        "Base Prompt",
        options=["active", "soft"],
        format_func=lambda value: BASE_STYLE_LABELS[value],
        index=0,
    )

    category_options = category_options_for_gender(gender)
    category = st.selectbox(
        "Category",
        options=category_options,
        format_func=lambda value: CATEGORY_LABELS[value],
        index=0,
    )

    st.subheader("Character Inputs")
    original_object = st.text_input("Original Object", value="toy bus")
    name = st.text_input("Name", value="Coco")
    job = st.text_input("Job", value="space explorer")
    traits_input = st.text_area(
        "Traits",
        value="round toy-bus body, bright window eyes, friendly hero silhouette",
        height=90,
        help="Use comma-separated visual traits.",
    )

    with st.expander("More Character Fields", expanded=False):
        personality = st.text_input("Personality", value="warm and brave")
        goal = st.text_input("Goal", value="wants to discover a new star")
        tone = st.text_input("Story Tone", value="warm")

    st.subheader("Reference Image")
    if not reference_images:
        selected_reference = ""
        st.error("No usable reference images were found in the nukki folder.")
    else:
        selected_reference = st.selectbox("Reference Image", reference_images, index=0)
        preview_path = _reference_image_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"Reference image: {selected_reference}", width="stretch")

    prompt_options = normalize_prompt_options(
        {
            "gender": gender,
            "base_style": base_style,
            "category": category,
        }
    )
    preview_character_sheet = _build_preview_character_sheet(
        original_object=original_object,
        name=name,
        job=job,
        traits_input=traits_input,
        personality=personality,
        goal=goal,
        tone=tone,
    )
    prompt_preview = build_prompt_preview(preview_character_sheet, prompt_options)
    summary = _build_prompt_summary(prompt_options)

with right_col:
    st.subheader("Applied Template")
    st.info(prompt_preview["selected_template"])

    st.subheader("Selection Summary")
    st.json(summary)

    st.subheader("Final Prompt Preview")
    st.code(prompt_preview["selected_prompt"], language="text")

    st.subheader("Preview Character Sheet")
    st.json(preview_character_sheet)

st.divider()
st.subheader("Run Image Test")

run_clicked = st.button("Generate Character Art", type="primary", width="stretch")

if run_clicked:
    if not selected_reference:
        st.error("Select a reference image first.")
    else:
        payload = {
            "vision_result": {"objects": [original_object]},
            "parent_input": {
                "name": name,
                "job": job,
                "personality": personality,
                "goal": goal,
                "extra_description": traits_input,
                "original_object_hint": original_object,
                "traits_input": traits_input,
                "tone": tone,
            },
            "reference_image": selected_reference,
            "prompt_options": prompt_options,
        }

        try:
            with st.spinner("Generating character art with the selected prompt template..."):
                result = _call_pipeline(payload)

            st.session_state.story_character_sheet_json = json.dumps(
                result["character_sheet"], ensure_ascii=False, indent=2
            )
            st.session_state.latest_style_prompts = result.get("style_prompts", {})
            st.session_state.latest_generated_images = result.get("generated_images", {})
            st.session_state.latest_prompt_options = prompt_options
            st.session_state.latest_story_reference_image = (
                result.get("generated_images", {}).get("active_style")
                if tone == "모험적인"
                else result.get("generated_images", {}).get("soft_style")
                or result.get("generated_images", {}).get("active_style")
            )
            st.session_state.latest_story_package = result.get("story", {})
            st.session_state.current_story_page = 0

            st.success("Character art generation completed.")

            result_left, result_right = st.columns([1, 1], gap="large")
            with result_left:
                st.subheader("Returned Character Sheet")
                st.json(result["character_sheet"])
                st.subheader("Returned Style Prompts")
                st.write("Active Style")
                st.code(result["style_prompts"]["active_style"], language="text")
                st.write("Soft Style")
                st.code(result["style_prompts"]["soft_style"], language="text")

            with result_right:
                st.subheader("Generated Images")
                _display_image_payload(result["generated_images"]["active_style"], "Active Style")
                _display_image_payload(result["generated_images"]["soft_style"], "Soft Style")

            st.info(
                "The selected template above controls which prompt family was emphasized. "
                "The backend still returns both active_style and soft_style outputs for compatibility."
            )
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Backend error: {detail or exc}")
        except Exception as exc:
            st.error(f"Execution failed: {exc}")
