"""Example script for testing the full pipeline locally."""

from __future__ import annotations

from app.pipeline import run_pipeline
from app.utils import list_reference_images, safe_json_dumps


def main() -> None:
    """Run the prototype pipeline with sample demo inputs."""

    images = list_reference_images()
    if not images:
        raise SystemExit("No reference images found in 박재혁/nukki.")

    vision_result = {"objects": ["곰인형"]}
    parent_input = {
        "name": "코코",
        "job": "우주 탐험가",
        "personality": "용감하고 다정함",
        "goal": "새로운 별을 찾고 싶어함",
        "extra_description": "작은 별 모양 가방을 메고 다녔으면 좋겠어",
    }
    reference_image = images[0]

    result = run_pipeline(vision_result, parent_input, reference_image)
    print(safe_json_dumps(result))


if __name__ == "__main__":
    main()

