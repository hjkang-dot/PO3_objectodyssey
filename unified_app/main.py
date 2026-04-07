import sys
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

# 통합 앱 폴더를 루트로 설정 (현재 폴더)
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 환경변수 로드
load_dotenv(ROOT_DIR / ".env")

# 비전 모듈 임포트
from odyssey_vision import process_object_odyssey, target_keywords
from fastapi import Form
from odyssey_audio import router as audio_router

# app 모듈 임포트
from app.pipeline import build_character_sheet, build_style_prompts, generate_images
try:
    from app.story_pipeline import generate_story_package as app_generate_story
except ImportError:
    # 혹시 JHPark/app 파일이 없다면 기존 방식 의존
    try:
        from app.pipeline import generate_story as app_generate_story
    except ImportError:
        from app.story import generate_story_chain as app_generate_story

from app.utils import list_reference_images, resolve_reference_image_path

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

    result = process_object_odyssey(img, target_keywords)
    if result["status"] == "error":
        return {"status": "fail", "message": result["message"]}

    return {
        "status": "success",
        "target": result["target"],
        "output_path": result["output_path"],
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

import json
import uuid
from datetime import datetime

class SaveStoryPayload(BaseModel):
    title: str
    character_name: str
    image_path: Optional[str] = None
    story_paragraphs: list[str]

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
    try:
        save_dir = ROOT_DIR / "saved_stories"
        if not save_dir.exists():
            return {"status": "success", "stories": []}
            
        stories = []
        for file_path in save_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["filename"] = file_path.name
                stories.append(data)
                
        return {"status": "success", "stories": stories}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_app.main:app", host="127.0.0.1", port=8000, reload=True)
