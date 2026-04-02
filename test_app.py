import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# 1. 접속 정보 설정 (최상단)
API_URL = "http://127.0.0.1:8000/extract"

# 2. 통신 전용 함수 (비즈니스 로직 분리)
def request_nukki_api(image_file):
    """
    image_file: st.file_uploader로부터 받은 파일 객체
    """
    try:
        # 파일을 FastAPI가 원하는 형식(Multipart)으로 패킹
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        
        # FastAPI에 POST 요청 (AI 연산 시간을 고려해 timeout은 넉넉히 또는 None)
        response = requests.post(API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"서버 오류: {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "message": f"통신 실패: {str(e)}"}

# 3. Streamlit UI 배치
st.title("🎨 Object Odyssey: AI 에셋 생성")

uploaded_file = st.file_uploader("사진을 올려주세요", type=["jpg", "png"])

if uploaded_file:
    # 화면 왼쪽: 원본 표시
    st.image(uploaded_file, caption="업로드된 이미지", width=300)
    
    # 4. 실제 통신 실행 시점 (버튼 클릭 시)
    if st.button("🚀 AI 누끼 따기 실행"):
        with st.spinner("백엔드 GPU 서버에서 처리 중..."):
            
            # API 호출
            result = request_nukki_api(uploaded_file)
            
            # 결과 처리
            if result["status"] == "success":
                st.success(f"탐지 완료: {result['target']}")
                
                # [중요] FastAPI가 로컬 경로를 주면 직접 읽어서 표시
                # 서버가 다른 PC라면 이미지를 직접 리턴받는 로직이 추가로 필요합니다.
                if "output_path" in result:
                    st.image(result["output_path"], caption="결과 이미지")
            else:
                st.error(result["message"])


# ---------------------------------------------------------------------------
# 새 그림 생성 데모
# ---------------------------------------------------------------------------
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
REFERENCE_IMAGES_URL = f"{BACKEND_BASE_URL}/reference-images"
CREATE_ART_URL = f"{BACKEND_BASE_URL}/create-art"


# 기준 이미지 목록을 백엔드에서 불러오는 함수
def load_reference_images() -> list[str]:
    try:
        response = requests.get(REFERENCE_IMAGES_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("reference_images", []))
    except Exception as exc:
        st.error(f"기준 이미지 목록을 불러오지 못했습니다: {exc}")
        return []


# 기준 이미지 경로를 화면 미리보기용으로 만드는 함수
def get_reference_preview_path(filename: str) -> Path:
    return ROOT_DIR / "nukki" / filename


# 생성 결과가 경로 또는 base64일 때 화면에 표시하는 함수
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


# 새 그림 생성 API를 호출하는 함수
def call_create_art(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(CREATE_ART_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


# 새 그림 생성용 입력 화면
st.divider()
st.title("새 그림 생성 데모")
st.caption("루트 nukki 폴더의 기준 이미지를 읽고, 부모 프롬프트를 반영해 새로운 그림을 만듭니다.")

left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("비전 결과 입력")
    objects_text = st.text_input(
        "객체 이름",
        value="곰인형",
        help="쉼표로 여러 객체를 입력할 수 있습니다. 예: 곰인형, 컵, 책",
    )

    st.subheader("부모 프롬프트 입력")
    art_name = st.text_input("이름", value="코코")
    art_job = st.text_input("직업", value="우주 탐험가")
    art_personality = st.text_input("성격", value="용감하고 다정함")
    art_goal = st.text_input("하고 싶은 일", value="새로운 별을 찾고 싶어함")
    art_extra = st.text_area(
        "추가 설명",
        value="작은 별 모양 가방을 메고 다녔으면 좋겠어",
        height=100,
    )

with right_col:
    st.subheader("기준 이미지 선택")
    reference_images = load_reference_images()
    if not reference_images:
        st.error("루트 nukki 폴더에서 사용할 수 있는 jpg, jpeg, png 이미지가 없습니다.")
        selected_reference = ""
    else:
        selected_reference = st.selectbox("기준 이미지", reference_images, index=0)
        preview_path = get_reference_preview_path(selected_reference)
        if preview_path.exists():
            st.image(str(preview_path), caption=f"기준 이미지: {selected_reference}", use_container_width=True)

run_create_art = st.button("새 그림 생성 시작", type="primary", use_container_width=True)

if run_create_art:
    objects = [item.strip() for item in objects_text.split(",") if item.strip()]
    if not selected_reference:
        st.error("기준 이미지를 먼저 선택해 주세요.")
    else:
        payload = {
            "vision_result": {"objects": objects},
            "parent_input": {
                "name": art_name,
                "job": art_job,
                "personality": art_personality,
                "goal": art_goal,
                "extra_description": art_extra,
            },
            "reference_image": selected_reference,
        }

        try:
            with st.spinner("새 그림과 캐릭터 정보를 생성하는 중입니다..."):
                result = call_create_art(payload)

            st.success("그림 생성이 완료되었습니다.")

            left_result, right_result = st.columns([1, 1], gap="large")
            with left_result:
                st.subheader("캐릭터 시트")
                st.json(result.get("character_sheet", {}))

                st.subheader("이미지 프롬프트")
                style_prompts = result.get("style_prompts", {})
                st.write("활동적인 스타일")
                st.code(style_prompts.get("active_style", ""))
                st.write("부드러운 스타일")
                st.code(style_prompts.get("soft_style", ""))

            with right_result:
                st.subheader("생성된 이미지")
                generated_images = result.get("generated_images", {})
                render_generated_image(generated_images.get("active_style", ""), "활동적인 스타일")
                render_generated_image(generated_images.get("soft_style", ""), "부드러운 스타일")

            st.subheader("전체 결과")
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
