import sys
import json
import uuid
import os
from pathlib import Path
from datetime import datetime
import shutil
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
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
from unified_app.odyssey_audio import router as audio_router, generate_book_audios, _generate_single_audio_internal
from unified_app.app.storybook_core import (
    generate_story_package, 
    generate_story_text_only, 
    generate_single_story_image
)

# 저장용 폴더 경로 설정 (ROOT_DIR 기준)
SAVED_STORIES_DIR = ROOT_DIR / "saved_stories"
STATIC_OUTPUTS_DIR = ROOT_DIR / "static" / "outputs"
AUDIO_OUTPUTS_DIR = STATIC_OUTPUTS_DIR / "audios"
IMG_OUTPUTS_DIR = STATIC_OUTPUTS_DIR / "images"

# 폴더 자동 생성
SAVED_STORIES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# app 모듈 임포트 (통합 패키지 경로 기반)
try:
    from unified_app.app.pipeline import build_character_sheet, build_style_prompts, generate_images
    from unified_app.app.story_pipeline import generate_story_package as app_generate_story
except ImportError as e:
    print(f"[WARN] 루트 패키지 임포트 실패, 로컬 경로 시도: {e}")
    from unified_app.app.pipeline import build_character_sheet, build_style_prompts, generate_images
    from unified_app.app.story_pipeline import generate_story_package as app_generate_story

from unified_app.app.models import PromptOptions
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
app.mount("/static", StaticFiles(directory=(ROOT_DIR / "static").resolve().as_posix()), name="static")

# [수정] 자산 전송 방식 변경: StaticFiles 마운트 대신 전용 핸들러 사용
# 이는 윈도우 환경에서 경로 인식 문제로 인한 404 에러를 가장 확실하게 해결하는 방법입니다.
from fastapi.responses import FileResponse
@app.get("/stories/{file_path:path}")
async def serve_story_assets(file_path: str):
    file_path = file_path.lstrip("/")
    full_path = SAVED_STORIES_DIR / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset not found: {file_path}")
    return FileResponse(full_path)

@app.get("/")
async def serve_frontend():
    return FileResponse(str(ROOT_DIR / "static" / "index.html"))

@app.post("/extract")
def extract_object(file: UploadFile = File(...)) -> dict[str, str]:
    # 동기 방식으로 파일 읽기 (Thread pool에서 실행됨)
    contents = file.file.read()
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

def create_character_art(vision_result: dict, parent_input: dict, reference_image: str, image_style: str = "active_style", prompt_options: Optional[PromptOptions] = None) -> dict:
    character_sheet = build_character_sheet(vision_result, parent_input)
    style_prompts = build_style_prompts(character_sheet, prompt_options)
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
    prompt_options: Optional[PromptOptions] = None

@app.post("/create-art")
async def create_art(payload: CreateArtPayload) -> dict:
    try:
        result = create_character_art(payload.vision_result, payload.parent_input, payload.reference_image, payload.image_style, payload.prompt_options)
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
    char_image_path: Optional[str] = None   # 주인공 캐릭터 이미지 경로 (레퍼런스)
    character_name: Optional[str] = None    # 저장 파일명에 사용
    image_style: str = "active_style"      # 그림 스타일 추가
    voice_id: Optional[str] = "default_women"
    prompt_options: Optional[PromptOptions] = None

@app.post("/generate-book")
def generate_book(payload: GenerateBookPayload) -> dict:
    """스토리 + TTS 스크립트 + 이미지/표지를 한 번에 생성하고 saved_stories에 자동 저장한다."""
    try:
        print(f"[DEBUG] Generating book with tone: {payload.story_tone}, style: {payload.image_style}, character: {payload.character_name}")
        
        # 1. 스토리 및 이미지/표지 생성
        try:
            # 캐릭터 스타일 프롬프트 생성 (PromptOptions 반영)
            style_prompts = build_style_prompts(payload.character_sheet, payload.prompt_options)
            
            print(f"[DEBUG] Calling app_generate_story with tone={payload.story_tone}")
            start_story_time = time.time()
            story_package = app_generate_story(
                character_sheet=payload.character_sheet,
                extra_prompt="",
                story_tone=payload.story_tone,
                style_prompts=style_prompts,
                reference_image=payload.char_image_path
            )
            story_duration = time.time() - start_story_time
            print(f"[DEBUG] Story & Image package generated in {story_duration:.2f}s. Title: {story_package.get('title')}")
        except Exception as exc:
            print(f"[ERROR] 스토리/이미지 생성 단계 실패: {exc}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"스토리/이미지 생성 실패: {exc}")

        paragraphs = story_package.get("story_paragraphs", [])
        if not paragraphs:
            print("[WARN] 생성된 스토리 문단이 없습니다. Fallback 확인 필요.")
        
        # 2. TTS 음성 파일 일괄 생성 (부모 목소리 또는 기본값)
        try:
            start_tts_time = time.time()
            page_audios = generate_book_audios(paragraphs, payload.voice_id)
            tts_duration = time.time() - start_tts_time
            print(f"[DEBUG] TTS audios generated in {tts_duration:.2f}s.")
        except Exception as exc:
            print(f"[WARN] TTS 생성 실패: {exc}")
            page_audios = [None] * (len(paragraphs) or 5)

        # 3. 이미지 경로 보정 (서버 상대 경로로 변환)
        def fix_path(p):
            if not p: return ""
            # URL 형식으로 변환 (앞에 / 추가 및 역슬래시 교체)
            try:
                p_str = str(p)
                return "/" + p_str.replace("\\", "/").lstrip("/")
            except:
                return ""

        # story_pages가 None이거나 비어있을 경우 대비
        story_pages = story_package.get("story_pages") or []
        page_images = []
        for img in story_pages:
            img_path = img.get("image_path") if isinstance(img, dict) else None
            page_images.append(fix_path(img_path))
            
        if not page_images:
            page_images = [""] * (len(paragraphs) or 5)

        return {
            "status": "success",
            "title": story_package.get("title", "마법의 동화책"),
            "story_paragraphs": paragraphs,
            "page_audios": page_audios,
            "page_images": page_images,
            "cover_image_path": fix_path(story_package.get("cover_image_path")),
            "tts_script": story_package.get("tts_script", []),
            "character_name": payload.character_name or "주인공",
            "char_image_path": payload.char_image_path
        }
    except HTTPException as hexc:
        # HTTPException은 FastAPI가 정의한 대로 응답하도록 전달
        raise hexc
    except Exception as exc:
        print(f"[CRITICAL] /generate-book 전역 에러 발생!")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Error Message: {str(exc)}")
        import traceback
        traceback.print_exc()
        
        # 503 에러 여부 확인 (외부 서비스 타임아웃 등)
        status_code = 500
        if "503" in str(exc) or "timeout" in str(exc).lower() or "unavailable" in str(exc).lower():
            status_code = 503
            
        raise HTTPException(status_code=status_code, detail=f"서버 내부 오류: {str(exc)}")

# --- 분할 요청 (Granular Request) 전용 엔드포인트 시작 ---

class PrepareStoryPayload(BaseModel):
    character_sheet: dict[str, Any]
    extra_prompt: str = ""
    story_tone: str | None = None

@app.post("/prepare-story")
def prepare_story(payload: PrepareStoryPayload) -> dict:
    """동화의 줄거리(텍스트)만 먼저 생성한다."""
    try:
        story_package = generate_story_text_only(
            payload.character_sheet,
            extra_prompt=payload.extra_prompt,
            story_tone=payload.story_tone
        )
        return {
            "status": "success",
            "story_package": story_package
        }
    except Exception as exc:
        print(f"[ERROR] /prepare-story 실패: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


class GeneratePageAssetPayload(BaseModel):
    item_type: str                  # "page" or "cover"
    page_data: Optional[dict] = None # page_number, page_text 등
    story_title: str
    character_sheet: dict[str, Any]
    style_prompts: dict[str, Any]
    reference_image: str
    story_tone: str
    voice_id: str = "default_women"

@app.post("/generate-page-asset")
async def generate_page_asset(payload: GeneratePageAssetPayload) -> dict:
    """단일 페이지의 이미지와 음성을 생성한다."""
    from unified_app.app.services.gemini_service import GeminiService
    gemini_service = GeminiService()

    result = {"status": "success", "image_url": None, "audio_url": None}

    # 1. 이미지 생성
    try:
        img_path = generate_single_story_image(
            item_type=payload.item_type,
            page_data=payload.page_data,
            story_title=payload.story_title,
            character_sheet=payload.character_sheet,
            style_prompts=payload.style_prompts,
            reference_image=payload.reference_image,
            story_tone=payload.story_tone,
            gemini_service=gemini_service
        )
        if img_path:
            # URL 형식으로 변환
            result["image_url"] = "/" + str(img_path).replace("\\", "/").lstrip("/")
    except Exception as exc:
        print(f"[WARN] 이미지 생성 실패 ({payload.item_type}): {exc}")
        # 이미지 실패해도 음성은 시도할 수 있도록 에러를 가두거나, 503 처리를 여기서 수행
        if "503" in str(exc) or "unavailable" in str(exc).lower():
            raise HTTPException(status_code=503, detail="이미지 생성 서버가 일시적으로 바쁩니다.")

    # 2. 음성 생성 (본문 페이지일 경우에만)
    if payload.item_type == "page" and payload.page_data:
        try:
            from unified_app.odyssey_audio import load_model_base, unload_model_base
            load_model_base() # 음성 생성 모델 준비
            try:
                audio_url = _generate_single_audio_internal(
                    payload.page_data.get("page_text", ""), 
                    payload.voice_id
                )
                result["audio_url"] = audio_url
            finally:
                unload_model_base() # VRAM 확보를 위해 즉시 언로드
        except Exception as exc:
            print(f"[WARN] 음성 생성 실패: {exc}")

    return result

# --- 분할 요청 엔드포인트 끝 ---

# 페이지별 이미지 생성 엔드포인트 (B안: 각 동화 페이지마다 씬 이미지 생성)
class GeneratePageImagesPayload(BaseModel):
    character_sheet: dict[str, Any]
    story_paragraphs: List[str]
    reference_image: str                    # 누끼 이미지 경로 (캐릭터 레퍼런스)
    image_style: str = "active_style"


@app.post("/generate-page-images")
def generate_page_images(payload: GeneratePageImagesPayload) -> dict:
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
    title: Optional[str] = "마법의 동화책"
    character_name: Optional[str] = "주인공"
    image_path: Optional[str] = None
    cover_image_path: Optional[str] = None
    story_paragraphs: Optional[List[str]] = Field(default_factory=list)
    page_images: Optional[List[str]] = None
    page_audios: Optional[List[str]] = None
    tts_script: Optional[List[dict]] = None
    
    model_config = ConfigDict(extra="allow") # 추가 필드 허용

@app.post("/save-story")
async def save_story(request: Request) -> dict:
    """
    [무적의 저장 로직] FastAPI의 유효성 검사를 우회하여 422 에러를 원천 차단합니다.
    모든 데이터는 raw JSON으로 받아 수동으로 처리합니다.
    """
    try:
        # 1. RAW 데이터 수신 (여기서 422 에러가 날 가능성 0%)
        try:
            data = await request.json()
        except Exception as e:
            print(f"[ERROR] JSON Parsing Failed: {e}")
            return {"status": "error", "detail": "올바른 JSON 형식이 아닙니다."}

        print(f"[DEBUG] Received story data for save: {list(data.keys())}")

        save_dir = ROOT_DIR / "saved_stories"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 2. 필드 수동 매핑 및 기본값 처리
        character_name = data.get("character_name") or "주인공"
        title = data.get("title") or "마법의 동화책"
        
        # [수정] 윈도우 환경의 한글 폴더명 URL 인코딩 문제를 방지하기 위해 
        # 폴더명에는 한글(주인공 이름)을 넣지 않고 날짜와 ID만 사용합니다.
        story_id = str(uuid.uuid4())[:8]
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"story_{now_str}_{story_id}"
        
        story_folder = save_dir / folder_name
        story_folder.mkdir(exist_ok=True, parents=True)
        assets_folder = story_folder / "assets"
        assets_folder.mkdir(exist_ok=True)

        to_cleanup = set()

        def localize_asset(url: Any, target_name: str) -> Any:
            """자산을 로컬 폴더(assets)로 복사하고 표준화된 짧은 이름을 부여합니다."""
            if not url or not isinstance(url, str) or url.startswith("data:"):
                return url
            if url.startswith("/stories/"): return url
            
            rel_path = url.lstrip("/")
            if rel_path.startswith("static/"): source_path = ROOT_DIR / rel_path
            elif rel_path.startswith("nukki/"): source_path = ROOT_DIR / rel_path
            else: return url
            
            if source_path.exists() and source_path.is_file():
                # 확장자 유지
                ext = source_path.suffix
                final_name = target_name if target_name.endswith(ext) else f"{target_name}{ext}"
                target_path = assets_folder / final_name
                
                shutil.copy2(source_path, target_path)
                if "static/outputs" in str(source_path.as_posix()):
                    to_cleanup.add(source_path)
                return f"/stories/{folder_name}/assets/{final_name}"
            return url

        # 3. 자산 로컬화 (사본 생성 + 표준 이름 부여)
        updated_data = dict(data) # 원본 복사
        
        if updated_data.get("image_path"):
            updated_data["image_path"] = localize_asset(updated_data["image_path"], "thumb")
        
        if updated_data.get("cover_image_path"):
            updated_data["cover_image_path"] = localize_asset(updated_data["cover_image_path"], "cover")

        if isinstance(updated_data.get("page_images"), list):
            new_imgs = []
            for i, img in enumerate(updated_data["page_images"]):
                new_imgs.append(localize_asset(img, f"page_{i+1}"))
            updated_data["page_images"] = new_imgs
            
        if isinstance(updated_data.get("page_audios"), list):
            new_auds = []
            for i, aud in enumerate(updated_data["page_audios"]):
                new_auds.append(localize_asset(aud, f"audio_{i+1}"))
            updated_data["page_audios"] = new_auds

        # 4. JSON 파일 저장
        file_path = story_folder / "story.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)

        # 5. 임시 파일 정리
        for path in to_cleanup:
            try:
                if path.exists(): path.unlink()
            except: pass
            
        print(f"[SUCCESS] Story saved as {folder_name}")
        return {"status": "success", "filename": folder_name}

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"status": "error", "detail": str(exc)}

@app.get("/get-stories")
def get_stories():
    """
    저장된 모든 동화 목록을 반환합니다. 
    성능 최적화: 디렉토리 정보를 효율적으로 읽어 I/O 부하를 줄입니다.
    """
    try:
        save_dir = ROOT_DIR / "saved_stories"
        if not save_dir.exists():
            return {"status": "success", "stories": []}
        
        stories = []
        # scandir로 디렉토리를 순회하며 story.json만 효율적으로 확인
        for entry in os.scandir(save_dir):
            if entry.is_dir():
                story_file = Path(entry.path) / "story.json"
                if story_file.exists():
                    try:
                        # 파일이 많아지면 메타데이터를 별도 index로 관리하는 것을 추천
                        with open(story_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        stories.append({
                            "title": data.get("title", "제목 없음"),
                            "character_name": data.get("character_name", "주인공"),
                            "image_path": data.get("image_path"),
                            "cover_image_path": data.get("cover_image_path"),
                            "filename": entry.name,
                            "mtime": entry.stat().st_mtime
                        })
                    except Exception as e:
                        print(f"[WARN] 스토리 읽기 실패 ({story_file}): {e}")
        
        # 최신 수정 시간 순으로 정렬
        stories.sort(key=lambda x: x["mtime"], reverse=True)
            
        return {"status": "success", "stories": stories}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/get-story/{filename}")
async def get_story_detail(filename: str) -> dict:
    """특정 동화의 전체 본문을 상세 조회한다."""
    try:
        # filename은 이제 폴더 이름임
        story_folder = ROOT_DIR / "saved_stories" / filename
        file_path = story_folder / "story.json"
        
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
        story_folder = ROOT_DIR / "saved_stories" / filename
        if not story_folder.exists():
            raise HTTPException(status_code=404, detail="Story not found")
        
        # 폴더 전체 삭제
        shutil.rmtree(story_folder)
        
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
    # 타임아웃 설정을 강화하여 이미지 생성 등 긴 작업 시간에도 안정적으로 연결 유지
    uvicorn.run(
        "unified_app.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        timeout_keep_alive=60
    )
