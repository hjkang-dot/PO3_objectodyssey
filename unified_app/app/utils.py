"""Utility helpers for reference images, output paths, and parsing."""

from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUKKI_DIR = PROJECT_ROOT / "nukki"
OUTPUTS_DIR = PROJECT_ROOT / "static" / "outputs"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def ensure_outputs_dir() -> Path:
    """Create the outputs directory if it does not already exist."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def list_reference_images() -> list[str]:
    """Return the available image filenames in the root nukki folder."""

    if not NUKKI_DIR.exists():
        return []

    images = [
        item.name
        for item in NUKKI_DIR.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(images)


def _is_relative_to(path: Path, base: Path) -> bool:
    """Compatibility helper for checking path containment."""

    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_reference_image_path(reference_image: str) -> Path:
    """Resolve and validate a reference image path inside the root nukki folder."""

    if not reference_image:
        raise ValueError("reference_image is required.")

    raw_path = Path(reference_image)
    if raw_path.is_absolute():
        candidate = raw_path.resolve()
    else:
        candidate = (NUKKI_DIR / raw_path.name).resolve()

    nukki_root = NUKKI_DIR.resolve()
    if not _is_relative_to(candidate, nukki_root):
        raise ValueError("Reference image must stay inside the root nukki folder.")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    if candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Only jpg, jpeg, and png reference images are supported.")

    return candidate


def read_image_bytes(path: Path) -> bytes:
    """Read an image file as raw bytes."""

    return path.read_bytes()


def load_pil_image(path: Path) -> Image.Image:
    """Load an image file as a Pillow image."""

    return Image.open(path).convert("RGBA")


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""

    normalized = re.sub(r"[^a-zA-Z0-9가-힣]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "item"


def save_pil_image(image: Image.Image, prefix: str) -> Path:
    """Save a Pillow image into outputs and return the absolute path."""

    ensure_outputs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{slugify(prefix)}_{timestamp}_{os.getpid()}_{image.size[0]}x{image.size[1]}.png"
    output_path = OUTPUTS_DIR / filename
    image.save(output_path, format="PNG")
    return output_path


def project_relative_path(path: Path) -> str:
    """Return a project-root relative path string for API responses."""

    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON text defensively, including fenced or noisy model output."""

    if not text:
        raise ValueError("Empty JSON response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Failed to parse JSON from model response.")


def safe_json_dumps(data: Any) -> str:
    """Serialize data to JSON with pretty output for logs and examples."""

    return json.dumps(data, ensure_ascii=False, indent=2)


def normalize_story_list(story: Any) -> list[str]:
    """Normalize the story field into a list of 4 to 6 sentences."""

    if isinstance(story, list):
        sentences = [str(item).strip() for item in story if str(item).strip()]
        return sentences[:6]

    if isinstance(story, str):
        chunks = re.split(r"(?<=[.!?])\s+", story.strip())
        sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
        return sentences[:6]

    return []


def image_to_data_uri(path: Path) -> str:
    """Convert a local image path to a base64 data URI."""

    suffix = path.suffix.lower().lstrip(".")
    mime_type = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime_type};base64,{encoded}"
