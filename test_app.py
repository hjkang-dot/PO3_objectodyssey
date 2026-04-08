import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
JHPARK_DIR = ROOT_DIR / "JHPark"
if str(JHPARK_DIR) not in sys.path:
    sys.path.insert(0, str(JHPARK_DIR))

from app.character import build_prompt_preview, category_options_for_gender, normalize_prompt_options

API_URL = "http://127.0.0.1:8000/extract"
BACKEND_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
REFERENCE_IMAGES_URL = f"{BACKEND_BASE_URL}/reference-images"
PIPELINE_URL = f"{BACKEND_BASE_URL}/pipeline"
STORY_URL = f"{BACKEND_BASE_URL}/generate-story"

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
STORY_TONE_OPTIONS = ["랜덤", "따뜻한", "모험적인", "교훈적인"]
TONE_TO_INDEX = {label: idx for idx, label in enumerate(STORY_TONE_OPTIONS[1:])}


def request_nukki_api(image_file: Any) -> dict[str, Any]:
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
        st.error(f"Failed to load reference images: {exc}")
        return []


def get_reference_preview_path(filename: str) -> Path:
    return ROOT_DIR / "nukki" / filename


def render_generated_image(value: str, caption: str) -> None:
    if not value:
        st.info(f"{caption} image is empty.")
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


def call_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(PIPELINE_URL, json=payload, timeout=1200)
    response.raise_for_status()
    return response.json()


def call_story_generation(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(STORY_URL, json=payload, timeout=1200)
    response.raise_for_status()
    return response.json()


def build_preview_character_sheet(
    original_object: str,
    name: str,
    job: str,
    traits_input: str,
    personality: str,
    goal: str,
    tone: str,
) -> dict[str, Any]:
    traits = [item.strip() for item in traits_input.split(",") if item.strip()]
    fallback_traits = traits or [
        f"based on the appearance of {original_object or 'reference object'}",
        f"wears visual hints of the job: {job or 'storybook hero'}",
        "keeps a toy-like silhouette with readable facial features",
    ]
    return {
        "original_object": original_object.strip() or "reference object",
        "name": name.strip() or "Coco",
        "job": job.strip() or "storybook hero",
        "personality": personality.strip() or "warm and brave",
        "goal": goal.strip() or "wants to discover something new",
        "core_visual_traits": fallback_traits[:5],
        "tone": tone or "따뜻한",
    }


def build_prompt_summary(prompt_options: dict[str, str]) -> dict[str, str]:
    return {
        "target_audience": GENDER_LABELS[prompt_options["gender"]],
        "base_style": BASE_STYLE_LABELS[prompt_options["base_style"]],
        "category": CATEGORY_LABELS[prompt_options["category"]],
    }


def get_story_default_character_sheet() -> dict[str, Any]:
    return {
        "original_object": "",
        "name": "코코",
        "job": "탐험가",
        "personality": "따뜻하고 공감이 많음",
        "goal": "새로운 세계를 발견하고 싶어함",
        "core_visual_traits": [
            "round toy body",
            "bright window eyes",
            "friendly hero silhouette",
            "anime style",
        ],
        "tone": "따뜻한",
    }


def reset_defaults_if_source_changed(default_story_character_sheet: dict[str, Any]) -> None:
    default_json = json.dumps(default_story_character_sheet, ensure_ascii=False, indent=2)
    if st.session_state.get("_default_story_character_sheet_json") != default_json:
        st.session_state.story_character_sheet_json = default_json
        st.session_state._default_story_character_sheet_json = default_json


def render_story_book(story_package: dict[str, Any]) -> None:
    pages = story_package.get("story_pages", [])
    if not pages:
        st.warning("표시할 동화 페이지가 없습니다.")
        return

    current_page = int(st.session_state.get("current_story_page", 0))
    current_page = max(0, min(current_page, len(pages) - 1))
    st.session_state.current_story_page = current_page
    page = pages[current_page]

    st.subheader(story_package.get("title", ""))
    st.caption(f"{current_page + 1} / {len(pages)} 페이지")

    page_left, page_right = st.columns([1.05, 0.95], gap="large")
    with page_left:
        render_generated_image(page.get("image_path") or "", f"Page {page.get('page_number', current_page + 1)}")
    with page_right:
        for sentence in page.get("sentences", []):
            st.write(sentence)
        with st.expander("이 페이지 이미지 프롬프트 보기", expanded=False):
            st.code(page.get("image_prompt", ""), language="text")

    nav_left, nav_mid, nav_right = st.columns([1, 1, 1])
    with nav_left:
        if st.button("이전 페이지", disabled=current_page == 0, width="stretch"):
            st.session_state.current_story_page = max(0, current_page - 1)
            st.rerun()
    with nav_mid:
        st.markdown(
            "<div style='text-align:center; padding-top:0.5rem;'>책을 넘기듯 페이지를 이동하세요.</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button("다음 페이지", disabled=current_page == len(pages) - 1, width="stretch"):
            st.session_state.current_story_page = min(len(pages) - 1, current_page + 1)
            st.rerun()

    if current_page == len(pages) - 1:
        cover_button_label = "표지 숨기기" if st.session_state.get("show_story_cover") else "표지 출력"
        if st.button(cover_button_label, key="toggle_story_cover", width="stretch"):
            st.session_state.show_story_cover = not st.session_state.get("show_story_cover", False)
            st.rerun()

    if st.session_state.get("show_story_cover"):
        st.subheader("동화책 표지")
        render_generated_image(story_package.get("cover_image_path") or "", "Story Cover")
        with st.expander("표지 이미지 프롬프트 보기", expanded=False):
            st.code(story_package.get("cover_prompt", ""), language="text")

    st.write("마지막 선택")
    choices = story_package.get("choices", [])
    if choices:
        choice_columns = st.columns(len(choices))
        for idx, choice in enumerate(choices):
            with choice_columns[idx]:
                if st.button(choice.get("text", ""), key=f"story_choice_{choice.get('id', idx)}", width="stretch"):
                    st.session_state.selected_story_choice = choice
        selected_choice = st.session_state.get("selected_story_choice")
        if selected_choice:
            st.info(f"선택한 이야기: {selected_choice['text']} ({selected_choice['id']})")


st.set_page_config(page_title="Object Odyssey Test App", page_icon="OO", layout="wide")

default_story_character_sheet = get_story_default_character_sheet()
reset_defaults_if_source_changed(default_story_character_sheet)

if "current_story_page" not in st.session_state:
    st.session_state.current_story_page = 0
if "show_story_cover" not in st.session_state:
    st.session_state.show_story_cover = False

default_original_object = str(default_story_character_sheet.get("original_object") or "")
default_name = str(default_story_character_sheet.get("name") or "")
default_job = str(default_story_character_sheet.get("job") or "")
default_traits_input = ", ".join(default_story_character_sheet.get("core_visual_traits") or [])
default_personality = str(default_story_character_sheet.get("personality") or "")
default_goal = str(default_story_character_sheet.get("goal") or "")
default_tone = str(default_story_character_sheet.get("tone") or STORY_TONE_OPTIONS[1])

st.title("Object Odyssey Test App")

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
st.title("Prompt-Driven Character Art Test")
st.caption("Generate a character first, then generate a 5-page illustrated storybook below.")

reference_images = load_reference_images()
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Prompt Controls")

    gender = st.selectbox(
        "Target Audience",
        options=["girl", "boy"],
        format_func=lambda value: GENDER_LABELS[value],
        index=0,
        help="This does not force the character to be male or female. It selects the presentation style that may appeal more to girls or boys.",
    )
    base_style = st.selectbox(
        "Base Prompt",
        options=["active", "soft"],
        format_func=lambda value: BASE_STYLE_LABELS[value],
        index=0,
    )
    category = st.selectbox(
        "Category",
        options=category_options_for_gender(gender),
        format_func=lambda value: CATEGORY_LABELS[value],
        index=0,
    )

    st.subheader("Character Inputs")
    original_object = st.text_input("Original Object", value=default_original_object)
    art_name = st.text_input("Name", value=default_name)
    art_job = st.text_input("Job", value=default_job)
    traits_input = st.text_area(
        "Traits",
        value=default_traits_input,
        height=90,
        help="Use comma-separated visual traits.",
    )

    with st.expander("More Character Fields", expanded=False):
        art_personality = st.text_input("Personality", value=default_personality)
        art_goal = st.text_input("Goal", value=default_goal)
        tone_options = STORY_TONE_OPTIONS[1:]
        tone_index = TONE_TO_INDEX.get(default_tone, 0)
        art_tone = st.selectbox("Story Tone", tone_options, index=tone_index)

    st.subheader("Reference Image")
    if not reference_images:
        st.error("No usable jpg, jpeg, or png images were found in the root nukki folder.")
        selected_reference = ""
    else:
        selected_reference = st.selectbox("Reference image", reference_images, index=0)
        preview_path = get_reference_preview_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"Reference image: {selected_reference}", width="stretch")

    prompt_options = normalize_prompt_options(
        {
            "gender": gender,
            "base_style": base_style,
            "category": category,
        }
    )
    preview_character_sheet = build_preview_character_sheet(
        original_object=original_object,
        name=art_name,
        job=art_job,
        traits_input=traits_input,
        personality=art_personality,
        goal=art_goal,
        tone=art_tone,
    )
    prompt_preview = build_prompt_preview(preview_character_sheet, prompt_options)
    summary = build_prompt_summary(prompt_options)

with right_col:
    st.subheader("Applied Template")
    st.info(prompt_preview["selected_template"])

    st.subheader("Selection Summary")
    st.json(summary)

    st.subheader("Final Prompt Preview")
    st.code(prompt_preview["selected_prompt"], language="text")

    st.subheader("Preview Character Sheet")
    st.json(preview_character_sheet)

run_create_art = st.button("Generate Character Art", type="primary", width="stretch")

if run_create_art:
    if not selected_reference:
        st.error("Select a reference image first.")
    else:
        payload = {
            "vision_result": {"objects": [original_object]},
            "parent_input": {
                "name": art_name,
                "job": art_job,
                "personality": art_personality,
                "goal": art_goal,
                "extra_description": traits_input,
                "original_object_hint": original_object,
                "traits_input": traits_input,
                "tone": art_tone,
            },
            "reference_image": selected_reference,
            "prompt_options": prompt_options,
        }

        try:
            with st.spinner("Generating character art and starter story package..."):
                result = call_pipeline(payload)

            st.success("Character art generation completed.")

            style_prompts = result.get("style_prompts", {})
            generated_images = result.get("generated_images", {})
            st.session_state.story_character_sheet_json = json.dumps(
                result.get("character_sheet", {}),
                ensure_ascii=False,
                indent=2,
            )
            st.session_state.latest_style_prompts = style_prompts
            st.session_state.latest_generated_images = generated_images
            st.session_state.latest_story_reference_image = (
                generated_images.get("active_style")
                if art_tone == "모험적인"
                else generated_images.get("soft_style") or generated_images.get("active_style")
            )
            st.session_state.latest_story_package = result.get("story", {})
            st.session_state.current_story_page = 0
            st.session_state.show_story_cover = False

            if result.get("story_error"):
                st.warning(f"Starter story generation skipped: {result['story_error']}")

            left_result, right_result = st.columns([1, 1], gap="large")
            with left_result:
                st.subheader("Character Sheet")
                st.json(result.get("character_sheet", {}))
                st.subheader("Style Prompts")
                st.write("Active Style")
                st.code(style_prompts.get("active_style", ""), language="text")
                st.write("Soft Style")
                st.code(style_prompts.get("soft_style", ""), language="text")

            with right_result:
                st.subheader("Generated Images")
                render_generated_image(generated_images.get("active_style", ""), "Active Style")
                render_generated_image(generated_images.get("soft_style", ""), "Soft Style")

        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Backend error: {detail or exc}")
        except Exception as exc:
            st.error(f"Execution failed: {exc}")

st.divider()
st.title("Illustrated Storybook Test")

story_extra_prompt = st.text_area(
    "Story Extra Prompt",
    value=(
        "15문장으로 구성하고, 5페이지가 인과관계로 이어지게 해줘. "
        "기승전결이 분명하고, 3페이지에서 갈등이 커지고, 4페이지에서 해결의 실마리가 나오고, "
        "5페이지는 감정적으로 만족스러운 결말이 되게 해줘. "
        "의성어와 의태어를 자연스럽게 넣고, 아이가 다음 장면을 궁금해하도록 만들어줘."
    ),
    height=110,
)
story_tone_option = st.selectbox("Story Style", STORY_TONE_OPTIONS, index=0)

run_story = st.button("Generate 5-Page Storybook", type="secondary", width="stretch")

if run_story:
    try:
        character_sheet = json.loads(st.session_state.story_character_sheet_json)
    except Exception as exc:
        st.error(f"character_sheet JSON 오류: {exc}")
    else:
        try:
            with st.spinner("Generating storybook pages and page images..."):
                story_package = call_story_generation(
                    {
                        "character_sheet": character_sheet,
                        "extra_prompt": story_extra_prompt,
                        "story_tone": None if story_tone_option == "랜덤" else story_tone_option,
                        "style_prompts": st.session_state.get("latest_style_prompts"),
                        "reference_image": st.session_state.get("latest_story_reference_image"),
                    }
                )
            st.session_state.latest_story_package = story_package
            st.session_state.current_story_page = 0
            st.session_state.show_story_cover = False
            st.success("5-page storybook generated.")
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Backend error: {detail or exc}")
        except Exception as exc:
            st.error(f"Execution failed: {exc}")

if st.session_state.get("latest_story_package"):
    render_story_book(st.session_state["latest_story_package"])
