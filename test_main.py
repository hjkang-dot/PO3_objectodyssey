from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import os
import requests
from odyssey_vision import process_object_odyssey, target_keywords, blacklist_keywords

app = FastAPI(title="Object Odyssey Vision API")

@app.post("/extract")
async def extract_object(file: UploadFile = File(...)):
    # 1. 업로드된 파일을 바이트로 읽기
    contents = await file.read()
    
    # 2. 바이트 데이터를 OpenCV 이미지(NumPy 배열)로 변환
    nparray = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

    # 3. 비전 파이프라인 실행
    result = process_object_odyssey(img, target_keywords, blacklist_keywords)
    
    if result["status"] == "error":
        return {"status": "fail", "message": result["message"]}
    
    return {
        "status": "success",
        "target": result["target"],
        "output_path": result["output_path"]
    }


# 박재혁 폴더 기능을 사용하기 위한 경로 추가
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
PO3_DIR = ROOT_DIR / "박재혁"
if str(PO3_DIR) not in sys.path:
    sys.path.insert(0, str(PO3_DIR))

from app.pipeline import build_character_sheet, build_style_prompts, generate_images
from app.utils import list_reference_images, resolve_reference_image_path


# 루트 nukki 폴더의 기준 이미지 목록을 가져오는 함수
def get_reference_images() -> list[str]:
    return list_reference_images()


# 기준 이미지 경로가 안전한지 확인하는 함수
def resolve_reference_image(reference_image: str) -> str:
    return str(resolve_reference_image_path(reference_image))


# 부모 프롬프트와 기준 이미지를 바탕으로 그림 생성 정보를 만드는 함수
def create_character_art(vision_result: dict, parent_input: dict, reference_image: str) -> dict:
    character_sheet = build_character_sheet(vision_result, parent_input)
    style_prompts = build_style_prompts(character_sheet)
    generated_images = generate_images(style_prompts, reference_image)

    return {
        "character_sheet": character_sheet,
        "style_prompts": style_prompts,
        "generated_images": generated_images,
    }


# 기준 이미지 목록을 확인하는 엔드포인트
@app.get("/reference-images")
def reference_images():
    images = get_reference_images()
    if not images:
        raise HTTPException(status_code=404, detail="루트 nukki 폴더에 사용할 이미지가 없습니다.")
    return {"reference_images": images}


# 부모 프롬프트와 기준 이미지로 새 그림을 만드는 엔드포인트
@app.post("/create-art")
async def create_art(payload: dict):
    vision_result = payload.get("vision_result", {})
    parent_input = payload.get("parent_input", {})
    reference_image = payload.get("reference_image")

    if not reference_image:
        raise HTTPException(status_code=400, detail="reference_image가 필요합니다.")

    try:
        result = create_character_art(vision_result, parent_input, reference_image)
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# 실행 방법: uvicorn main:app --reload