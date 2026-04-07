import json
from app.story_pipeline import generate_story_package
from pydantic import ValidationError

sheet = {
    "original_object": "sword",
    "name": "칼리",
    "job": "warrior",
    "personality": "brave",
    "goal": "defeat the dragon",
    "core_visual_traits": ["shiny"],
    "tone": "모험적인"
}

try:
    result = generate_story_package(sheet)
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()
