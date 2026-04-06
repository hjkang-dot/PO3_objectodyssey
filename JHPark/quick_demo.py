"""Short demo runner for the PO3 prototype."""

from __future__ import annotations

from app.pipeline import run_pipeline
from app.utils import list_reference_images, safe_json_dumps


def main() -> None:
    """Run the end-to-end pipeline with sample inputs."""

    images = list_reference_images()
    if not images:
        raise SystemExit("No reference images found in JHPark/nukki.")

    payload = run_pipeline(
        vision_result={"objects": ["고양이 인형"]},
        parent_input={
            "name": "코코",
            "job": "우주 탐험가",
            "personality": "용감하고 다정함",
            "goal": "새로운 별을 찾고 싶어함",
            "extra_description": "작은 별 모양 가방을 메고 있으면 좋겠음",
        },
        reference_image=images[0],
    )
    print(safe_json_dumps(payload))


if __name__ == "__main__":
    main()
