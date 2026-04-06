# from __future__ import annotations
# uvicorn test_main:app --reload --port 8501

import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile

from odyssey_vision import process_object_odyssey, target_keywords

app = FastAPI(title="Object Odyssey Vision API")


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


ROOT_DIR = Path(__file__).resolve().parent
PO3_DIR = next(
    (item for item in ROOT_DIR.iterdir() if item.is_dir() and (item / "app" / "main.py").exists()),
    ROOT_DIR,
)
if str(PO3_DIR) not in sys.path:
    sys.path.insert(0, str(PO3_DIR))

load_dotenv(PO3_DIR / ".env")

from app.pipeline import build_character_sheet, build_style_prompts, generate_images
from app.utils import list_reference_images, resolve_reference_image_path


def get_reference_images() -> list[str]:
    return list_reference_images()


def resolve_reference_image(reference_image: str) -> str:
    return str(resolve_reference_image_path(reference_image))


def create_character_art(vision_result: dict, parent_input: dict, reference_image: str) -> dict:
    character_sheet = build_character_sheet(vision_result, parent_input)
    style_prompts = build_style_prompts(character_sheet)
    generated_images = generate_images(style_prompts, reference_image)
    return {
        "character_sheet": character_sheet,
        "style_prompts": style_prompts,
        "generated_images": generated_images,
    }


@app.get("/reference-images")
def reference_images() -> dict[str, list[str]]:
    images = get_reference_images()
    if not images:
        raise HTTPException(status_code=404, detail="No usable images were found in the root nukki folder.")
    return {"reference_images": images}


@app.post("/create-art")
async def create_art(payload: dict) -> dict:
    vision_result = payload.get("vision_result", {})
    parent_input = payload.get("parent_input", {})
    reference_image = payload.get("reference_image")

    if not reference_image:
        raise HTTPException(status_code=400, detail="reference_image is required.")

    try:
        result = create_character_art(vision_result, parent_input, reference_image)
        return {"status": "success", **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
