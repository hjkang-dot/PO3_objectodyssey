import sys
import json
import uuid
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Optional

# uvicorn unified_app.main:app --host 127.0.0.1 --port 8000 --reload

# 통합 앱 폴더를 루트로 설정 (현재 폴더)
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 환경변수 로드 (현재 폴더의 .env 우선)
env_path = ROOT_DIR / ".env"
load_dotenv(env_path, override=True)

# API 키 로드 확인 (디버깅용)
if not os.getenv("OPENAI_API_KEY"):
    print(f"[WARN] OPENAI_API_KEY가 {env_path}에서 로드되지 않았습니다.")
else:
    print(f"[INFO] OPENAI_API_KEY 로드 성공 (앞 4자리: {os.getenv('OPENAI_API_KEY')[:4]}...)")

# 비전 모듈 임포트
from unified_app.odyssey_vision import process_object_odyssey
from fastapi import Form
from unified_app.odyssey_audio import router as audio_router, generate_book_audios

# 저장용 폴더 경로 설정 (ROOT_DIR 기준)
SAVED_STORIES_DIR = ROOT_DIR / "saved_stories"
STATIC_OUTPUTS_DIR = ROOT_DIR / "static" / "outputs"
AUDIO_OUTPUTS_DIR = STATIC_OUTPUTS_DIR / "audios"
IMG_OUTPUTS_DIR = STATIC_OUTPUTS_DIR / "images"

# 폴더 자동 생성
SAVED_STORIES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# app 모듈 임포트
from unified_app.app.pipeline import build_character_sheet, build_style_prompts, generate_images
try:
    from unified_app.app.story_pipeline import generate_story_package as app_generate_story
except ImportError:
    # 혹시 JHPark/app 파일이 없다면 기존 방식 의존
    try:
        from unified_app.app.pipeline import generate_story as app_generate_story
    except ImportError:
        from unified_app.app.story import generate_story_chain as app_generate_story

from unified_app.app.utils import list_reference_images, resolve_reference_image_path

app = FastAPI(title="Unified PO3 App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 연결
app.include_router(audio_router)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(str(ROOT_DIR / "static" / "index.html"))

@app.post("/extract")
async def extract_object(file: UploadFile = File(...)) -> dict[str, str]:
    contents = await file.read()
    nparray = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # target_keywords 인자 제거 — BiRefNet 파이프라인은 키워드 없이 동작
    result = process_object_odyssey(img)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    return {
        "status"      : "success",
        # ※ "target" 필드 삭제 — YOLO 키워드 탐지 결과로 불필요
        "description" : result["description"],   # Florence-2 상세 묘사
        "output_path" : result["output_path"],   # 저장된 누끼 파일 경로
    }

def create_character_art(vision_result: dict, parent_input: dict, reference_image: str, image_style: str = "active_style") -> dict:
    character_sheet = build_character_sheet(vision_result, parent_input)
    style_prompts = build_style_prompts(character_sheet)
    generated_images = generate_images(style_prompts, reference_image, image_style)
    return {
        "character_sheet": character_sheet,
        "style_prompts": style_prompts,
        "generated_images": generated_images,
    }

class CreateArtPayload(BaseModel):
    vision_result: dict[str, Any]
    parent_input: dict[str, Any]
    reference_image: str
    image_style: str = "active_style"

@app.post("/create-art")
async def create_art(payload: CreateArtPayload) -> dict:
    try:
        result = create_character_art(payload.vision_result, payload.parent_input, payload.reference_image, payload.image_style)
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class GenerateStoryPayload(BaseModel):
    character_sheet: dict[str, Any]
    extra_prompt: Optional[str] = ""
    story_tone: Optional[str] = None
    # 만약에 이전 히스토리 등도 추가된다면 여기 포함

@app.post("/generate-story")
async def generate_story(payload: GenerateStoryPayload) -> dict:
    try:
        # JHPark/app/story_pipeline.py 에 구현된 최신 로직 활용
        result = app_generate_story(
            character_sheet=payload.character_sheet,
            extra_prompt=payload.extra_prompt,
            story_tone=payload.story_tone
        )
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# -------------------------------------------------------
# /generate-book: 스토리 생성 → 저장 → 책 UI용 데이터 반환
# 선택지(choices)는 사용하지 않으므로 응답에서 제외
# -------------------------------------------------------
class GenerateBookPayload(BaseModel):
    character_sheet: Optional[dict[str, Any]] = None
    story_tone: Optional[str] = None
    char_image_path: Optional[str] = None   # 주인공 캐릭터 이미지 경로 (저장용)
    character_name: Optional[str] = None    # 저장 파일명에 사용
    voice_id: Optional[str] = "default_women"
@app.post("/generate-book")
async def generate_book(payload: GenerateBookPayload) -> dict:
    """스토리 + TTS 스크립트를 한 번에 생성하고 saved_stories에 자동 저장한다."""
    try:
        print(f"[DEBUG] Generating book with tone: {payload.story_tone}, character: {payload.character_name}")
        story_package = app_generate_story(
            character_sheet=payload.character_sheet,
            extra_prompt="",
            story_tone=payload.story_tone,
        )
    except Exception as exc:
        print(f"[ERROR] 스토리 생성 실패: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"스토리 생성 실패: {exc}") from exc

    # TTS 음성 파일 일괄 생성 (부모 목소리 또는 기본값)
    try:
        paragraphs = story_package.get("story_paragraphs", [])
        page_audios = generate_book_audios(paragraphs, payload.voice_id)
    except Exception as exc:
        print(f"[WARN] TTS 생성 실패: {exc}")
        page_audios = [None] * len(paragraphs)

    # 자동 저장
    try:
        save_dir = ROOT_DIR / "saved_stories"
        save_dir.mkdir(exist_ok=True, parents=True)
        story_id = str(uuid.uuid4())[:8]
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        char_name = payload.character_name or story_package.get("title", "주인공")[:8]
        filename = f"story_{char_name}_{now_str}_{story_id}.json"
        
        save_payload = {
            "title": story_package.get("title", ""),
            "character_name": char_name,
            "image_path": payload.char_image_path,
            "story_paragraphs": paragraphs,
            "page_images": [payload.char_image_path] * len(paragraphs) if payload.char_image_path else None,
            "page_audios": page_audios,
            "tts_script": [
                {"line": item["line"], "tone": item["tone"]}
                for item in story_package.get("tts_script", [])
            ],
        }
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=2)
        saved_filename = filename
    except Exception as exc:
        print(f"[WARN] 자동 저장 실패: {exc}")
        saved_filename = None

    return {
        "status": "success",
        "title": story_package.get("title", ""),
        "story_paragraphs": paragraphs,
        "page_audios": page_audios,
        "saved_filename": saved_filename,
    }



# 페이지별 이미지 생성 엔드포인트 (B안: 각 동화 페이지마다 씬 이미지 생성)
class GeneratePageImagesPayload(BaseModel):
    character_sheet: dict[str, Any]
    story_paragraphs: List[str]
    reference_image: str                    # 누끼 이미지 경로 (캐릭터 레퍼런스)
    image_style: str = "active_style"


@app.post("/generate-page-images")
async def generate_page_images(payload: GeneratePageImagesPayload) -> dict:
    """스토리 문단별로 씬(장면) 일러스트를 생성한다."""
    from unified_app.app.image_flow import (
        build_reference_prompt_seed,
        compose_final_image_prompt,
    )
    from unified_app.app.services.gemini_service import GeminiService
    from unified_app.app.utils import resolve_reference_image_path

    gemini_service = GeminiService()
    reference_path = resolve_reference_image_path(payload.reference_image)

    # 레퍼런스 이미지에서 캐릭터 시각 정보 추출 (1회만 수행)
    try:
        ref_seed = build_reference_prompt_seed(reference_path, payload.image_style, gemini_service)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Reference image 분석 실패: {exc}") from exc

    char = payload.character_sheet
    char_summary = (
        f"캐릭터 이름: {char.get('name', '주인공')}, "
        f"직업: {char.get('job', '')}, "
        f"성격: {char.get('personality', '')}, "
        f"목표: {char.get('goal', '')}"
    )

    page_image_urls: List[Optional[str]] = []

    for idx, paragraph in enumerate(payload.story_paragraphs):
        # 각 페이지의 장면을 설명하는 스타일 프롬프트 생성
        scene_prompt = (
            f"{char_summary}. "
            f"Page {idx + 1} scene: {paragraph} "
            f"Children's picture-book illustration, "
            f"{'vibrant and energetic' if payload.image_style == 'active_style' else 'warm and gentle'} style, "
            f"ages 6-8, clear character silhouette."
        )
        final_prompt = compose_final_image_prompt(ref_seed, scene_prompt, payload.image_style)
        try:
            img_path = gemini_service.generate_image(
                final_prompt,
                str(reference_path),
                f"page_{idx + 1}_{payload.image_style}",
            )
            # 서버 상대 경로 → URL 경로로 변환
            if img_path and not img_path.startswith("/"):
                img_path = "/" + img_path.replace("\\", "/")
            page_image_urls.append(img_path)
        except Exception as exc:
            # 한 페이지 실패해도 나머지 계속 진행 (None으로 채움)
            print(f"[WARN] Page {idx + 1} image generation failed: {exc}")
            page_image_urls.append(None)

    return {"status": "success", "page_images": page_image_urls}


class SaveStoryPayload(BaseModel):
    title: str
    character_name: str
    image_path: Optional[str] = None
    story_paragraphs: list[str]
    page_images: Optional[List[str]] = None   # 각 페이지 이미지 URL 목록
    tts_script: Optional[List[dict]] = None   # TTS 스크립트 (저장용)

@app.post("/save-story")
async def save_story(payload: SaveStoryPayload) -> dict:
    try:
        save_dir = ROOT_DIR / "saved_stories"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        story_id = str(uuid.uuid4())[:8]
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"story_{payload.character_name}_{now_str}_{story_id}.json"
        
        file_path = save_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload.model_dump(), f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "filename": filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/get-stories")
async def get_stories() -> dict:
    """보관함 목록을 위한 요약 정보(메타데이터)만 반환한다."""
    try:
        save_dir = ROOT_DIR / "saved_stories"
        if not save_dir.exists():
            return {"status": "success", "stories": []}
            
        stories = []
        # 최신 파일이 먼저 오도록 정렬 (mtime 기준)
        json_files = sorted(save_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 요약 정보만 필터링 (메모리 및 네트워크 부하 감소)
                    summary = {
                        "title": data.get("title", "제목 없음"),
                        "character_name": data.get("character_name", "주인공"),
                        "image_path": data.get("image_path"),
                        "filename": file_path.name,
                        "created_at": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    }
                    stories.append(summary)
            except Exception as e:
                print(f"[WARN] 파일 읽기 실패 ({file_path.name}): {e}")
                
        return {"status": "success", "stories": stories}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/get-story/{filename}")
async def get_story_detail(filename: str) -> dict:
    """특정 동화의 전체 본문을 상세 조회한다."""
    try:
        file_path = ROOT_DIR / "saved_stories" / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Story not found")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["filename"] = filename
            return {"status": "success", "story": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.delete("/delete-story/{filename}")
async def delete_story(filename: str) -> dict:
    """보관함에서 스토리를 삭제한다."""
    try:
        file_path = ROOT_DIR / "saved_stories" / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Story not found")
        
        # 파일 내용에서 TTS URL 등을 확인하여 관련 파일도 삭제 시도 (Optional)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tts_url = data.get("tts_url")
                if tts_url:
                    # tts_url이 /static/outputs/audios/... 형태인 경우 처리
                    audio_rel_path = tts_url.lstrip("/")
                    audio_full_path = ROOT_DIR / audio_rel_path
                    if audio_full_path.exists():
                        audio_full_path.unlink()
        except: pass

        file_path.unlink()
        return {"status": "success", "message": f"{filename} 삭제 완료"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

class CleanupPayload(BaseModel):
    path: str

@app.post("/cleanup-temp")
async def cleanup_temp(payload: CleanupPayload) -> dict:
    """취소 시 생성된 임시 파일을 삭제한다."""
    try:
        # 보안상 static/outputs 내의 파일만 삭제 허용
        target_path = ROOT_DIR / payload.path.lstrip("/")
        if "static" in str(target_path) and target_path.exists():
            target_path.unlink()
            return {"status": "success", "message": "임시 파일 삭제 완료"}
        return {"status": "skipped", "message": "파일이 없거나 삭제 권한이 없습니다."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_app.main:app", host="127.0.0.1", port=8000, reload=True)
