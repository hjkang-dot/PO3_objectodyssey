"""story generation만 단독으로 확인하는 예제."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.story_pipeline import generate_story_package


def main() -> None:
    """샘플 character_sheet로 story package를 생성해 출력한다."""

    load_dotenv(Path(__file__).resolve().parent / ".env")

    character_sheet = {
        "original_object": "곰인형",
        "name": "코코",
        "job": "우주 탐험가",
        "personality": "용감하고 다정함",
        "goal": "새로운 별을 찾고 싶어함",
        "core_visual_traits": ["작은 별가방", "반짝이는 우주복"],
        "tone": "모험적인",
    }

    story_package = generate_story_package(character_sheet)
    print(json.dumps(story_package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
