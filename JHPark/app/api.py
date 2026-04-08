"""FastAPI route definitions for the PO3 prototype."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import (
    CharacterSheetRequest,
    GenerateImagesRequest,
    GenerateStoryRequest,
    PipelineRequest,
    PipelineResponse,
    ReferenceImagesResponse,
    StoryRequest,
    StoryPackageResponse,
    StylePromptsRequest,
)
from app.pipeline import (
    build_character_sheet,
    build_style_prompts,
    generate_images,
    generate_story,
    run_pipeline,
)
from app.utils import list_reference_images

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple health payload."""

    return {"status": "ok"}


@router.get("/reference-images", response_model=ReferenceImagesResponse)
def get_reference_images() -> dict[str, list[str]]:
    """Return available reference images from the root nukki folder."""

    images = list_reference_images()
    if not images:
        raise HTTPException(
            status_code=404,
            detail="No jpg, jpeg, or png reference images were found in the root nukki folder.",
        )
    return {"reference_images": images}


@router.post("/character-sheet")
def post_character_sheet(payload: CharacterSheetRequest) -> dict:
    """Generate a structured character sheet."""

    try:
        return build_character_sheet(payload.vision_result.model_dump(), payload.parent_input.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/style-prompts")
def post_style_prompts(payload: StylePromptsRequest) -> dict:
    """Generate two style-specific prompts."""

    try:
        return build_style_prompts(
            payload.character_sheet.model_dump(),
            payload.prompt_options.model_dump() if payload.prompt_options else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate-images")
def post_generate_images(payload: GenerateImagesRequest) -> dict:
    """Generate the active and soft style images."""

    try:
        return generate_images(payload.style_prompts.model_dump(), payload.reference_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/generate-story", response_model=StoryPackageResponse)
def post_generate_story(payload: StoryRequest) -> dict:
    """Generate the story package from a character sheet."""

    try:
        return generate_story(
            payload.character_sheet.model_dump(),
            extra_prompt=payload.extra_prompt,
            story_tone=payload.story_tone,
            style_prompts=payload.style_prompts.model_dump() if payload.style_prompts else None,
            reference_image=payload.reference_image,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pipeline", response_model=PipelineResponse)
def post_pipeline(payload: PipelineRequest) -> dict:
    """Run the full pipeline from inputs to the final demo result."""

    try:
        return run_pipeline(
            payload.vision_result.model_dump(),
            payload.parent_input.model_dump(),
            payload.reference_image,
            payload.prompt_options.model_dump() if payload.prompt_options else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
