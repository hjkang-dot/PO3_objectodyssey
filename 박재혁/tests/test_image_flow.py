from pathlib import Path

from app.image_flow import compose_final_image_prompt, generate_images


class StubGeminiService:
    def __init__(self) -> None:
        self.is_configured = True
        self.calls: list[tuple[str, str, str]] = []

    def describe_reference_image(self, reference_image: str, style_label: str) -> dict[str, object]:
        stem = Path(reference_image).stem
        return {
            "prompt": f"{stem} character identity",
            "reference_description": f"{stem} reference description",
            "key_visual_facts": [f"{stem} silhouette", "round face", "small backpack"],
        }

    def generate_image(self, prompt: str, reference_image: str, style_label: str) -> str:
        self.calls.append((prompt, reference_image, style_label))
        return f"outputs/{style_label}.png"


def test_compose_final_image_prompt_requests_full_redraw() -> None:
    prompt = compose_final_image_prompt(
        {
            "prompt": "toy robot identity",
            "reference_description": "a squat toy robot with a red helmet",
            "key_visual_facts": ["boxy silhouette", "red helmet", "big eyes"],
        },
        "Energetic robot child hero in a dramatic 3/4 pose.",
        "active_style",
    )

    assert "not as a final canvas" in prompt
    assert "brand-new drawn image" in prompt
    assert "Do not apply a simple filter" in prompt


def test_generate_images_uses_reference_guided_generation(tmp_path: Path, monkeypatch) -> None:
    reference_dir = tmp_path / "nukki"
    reference_dir.mkdir()
    reference_path = reference_dir / "sample.png"
    reference_path.write_bytes(b"fake-image")

    monkeypatch.setattr("app.utils.NUKKI_DIR", reference_dir)

    gemini_service = StubGeminiService()
    outputs = generate_images(
        {
            "active_style": "Dynamic character redesign for a brave explorer.",
            "soft_style": "Gentle character redesign for a cozy bedtime scene.",
        },
        "sample.png",
        gemini_service,
    )

    assert outputs == {
        "active_style": "outputs/active_style.png",
        "soft_style": "outputs/soft_style.png",
    }
    assert [call[2] for call in gemini_service.calls] == ["active_style", "soft_style"]
    assert all(str(reference_path) == call[1] for call in gemini_service.calls)
    assert all("brand-new drawn image" in call[0] for call in gemini_service.calls)
