"""Gemini service helpers for text and image generation."""

from __future__ import annotations

import os
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from app.prompts import reference_image_prompt
from app.utils import load_pil_image, project_relative_path, save_pil_image
from app.utils import safe_json_loads

try:  # pragma: no cover - optional dependency
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None

try:  # pragma: no cover - optional dependency
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai_types = None


@dataclass
class GeminiService:
    """Wrapper around Google Gemini text and image calls."""

    api_key: str | None = None
    text_model: str = "gemini-2.0-flash"
    image_model: str = "gemini-2.5-flash-image"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY") or None
        self.text_model = os.getenv("GEMINI_TEXT_MODEL", self.text_model)
        configured_image_model = os.getenv("GEMINI_IMAGE_MODEL", self.image_model)
        if configured_image_model.startswith("imagen-3"):
            configured_image_model = "gemini-2.5-flash-image"
        self.image_model = configured_image_model

    @property
    def is_configured(self) -> bool:
        """Return True when a Gemini API key and SDK are available."""

        return bool(self.api_key and genai is not None)

    def _client(self) -> Any:
        if not self.is_configured:
            raise RuntimeError("Gemini API is not configured.")
        return genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str) -> str:
        """Generate text using Gemini."""

        if not self.is_configured:
            raise RuntimeError("Gemini API is not configured.")

        client = self._client()
        response = client.models.generate_content(
            model=self.text_model,
            contents=prompt,
        )
        text = getattr(response, "text", None) or ""
        if not text and hasattr(response, "candidates"):
            text = str(response)
        if not text:
            raise RuntimeError("Gemini text generation returned an empty response.")
        return text.strip()

    def describe_reference_image(self, reference_image: str, style_label: str) -> dict[str, Any]:
        """Convert a reference image into a structured prompt seed."""

        reference_path = Path(reference_image).resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {reference_image}")

        if not self.is_configured:
            stem = reference_path.stem
            return {
                "prompt": f"An illustration inspired by {stem}.",
                "reference_description": f"An object-based character inspired by {stem}.",
                "key_visual_facts": [stem],
            }

        client = self._client()
        image_obj = load_pil_image(reference_path).convert("RGB")
        prompt = reference_image_prompt(reference_path.stem, style_label)
        request_kwargs: dict[str, Any] = {
            "model": self.text_model,
            "contents": [image_obj, prompt],
        }
        if genai_types is not None:
            request_kwargs["config"] = genai_types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(**request_kwargs)
        text = getattr(response, "text", None) or ""
        if not text:
            text = str(response)
        parsed = safe_json_loads(text)
        prompt_text = str(parsed.get("prompt") or "").strip()
        if not prompt_text:
            raise RuntimeError("Reference image prompt generation returned an empty response.")
        return {
            "prompt": prompt_text,
            "reference_description": str(parsed.get("reference_description") or "").strip(),
            "key_visual_facts": [str(item).strip() for item in parsed.get("key_visual_facts", []) if str(item).strip()],
        }

    def _extract_image_bytes(self, response: Any) -> bytes | None:
        """Extract inline image bytes from a Gemini response if present."""

        parts = getattr(response, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None) if inline_data else None
            if data:
                return data

        generated_images = getattr(response, "generated_images", None)
        if generated_images:
            first = generated_images[0]
            image_obj = getattr(first, "image", None)
            image_bytes = getattr(image_obj, "image_bytes", None)
            if image_bytes:
                return image_bytes

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None) if inline_data else None
                if data:
                    return data
        return None

    def _fallback_style_image(self, reference_image: Path, style_label: str, prompt: str) -> Path:
        """Create a local stylized derivative when the API is unavailable."""

        image = load_pil_image(reference_image).resize((1024, 1024))

        if style_label == "active_style":
            image = ImageEnhance.Color(image).enhance(1.45)
            image = ImageEnhance.Contrast(image).enhance(1.25)
            image = ImageEnhance.Sharpness(image).enhance(1.3)
            image = image.filter(ImageFilter.DETAIL)
            overlay = Image.new("RGBA", image.size, (255, 188, 64, 45))
            image = Image.alpha_composite(image, overlay)
        else:
            image = ImageEnhance.Color(image).enhance(0.9)
            image = ImageEnhance.Contrast(image).enhance(0.92)
            image = image.filter(ImageFilter.SMOOTH_MORE)
            overlay = Image.new("RGBA", image.size, (255, 235, 229, 55))
            image = Image.alpha_composite(image, overlay)

        framed = ImageOps.expand(image, border=18, fill=(255, 255, 255, 255))
        canvas = Image.new("RGBA", framed.size, (255, 255, 255, 255))
        canvas.alpha_composite(framed)

        draw = ImageDraw.Draw(canvas)
        label = "ACTIVE" if style_label == "active_style" else "SOFT"
        text = f"{label} STYLE"
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:  # pragma: no cover - font availability depends on OS
            font = ImageFont.load_default()
        draw.rounded_rectangle((24, 24, 232, 82), radius=18, fill=(0, 0, 0, 160))
        draw.text((42, 38), text, fill=(255, 255, 255, 255), font=font)

        return save_pil_image(canvas, f"{style_label}_{prompt[:24]}")

    def generate_image_from_prompt(self, prompt: str, style_label: str) -> str:
        """Generate a style-specific image from a text prompt and return a project-relative path."""

        if not self.is_configured:
            raise RuntimeError("Gemini API is not configured.")

        client = self._client()

        request_kwargs: dict[str, Any] = {
            "model": self.image_model,
            "contents": prompt,
        }
        if genai_types is not None:
            request_kwargs["config"] = genai_types.GenerateContentConfig(response_modalities=["Image"])

        response = client.models.generate_content(**request_kwargs)
        generated_bytes = self._extract_image_bytes(response)
        if not generated_bytes:
            raise RuntimeError(
                "Gemini image generation did not return image bytes. "
                "Check the model name, API key, and billing/quota settings."
            )

        image = Image.open(io.BytesIO(generated_bytes)).convert("RGBA")
        saved = save_pil_image(image, f"{style_label}_{prompt[:24]}")
        return project_relative_path(saved)

    def generate_image(self, prompt: str, reference_image: str, style_label: str) -> str:
        """Compatibility wrapper for older callers."""

        reference_path = Path(reference_image).resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image does not exist: {reference_image}")
        if not self.is_configured:
            return project_relative_path(self._fallback_style_image(reference_path, style_label, prompt))
        return self.generate_image_from_prompt(prompt, style_label)
