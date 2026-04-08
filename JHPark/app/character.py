"""Character sheet builders and deterministic image prompt templates."""

from __future__ import annotations

from typing import Any

from app.models import CharacterSheet
from app.prompts import character_sheet_prompt
from app.services.gemini_service import GeminiService
from app.utils import safe_json_loads

# 이 기본 지시문은 참고 이미지를 "그대로 보정하는 대상"이 아니라
# "정체성을 추출할 참고 재료"로 다루게 만들기 위한 블록이다.
# 즉 사진을 살짝 손보는 것이 아니라, 원본 오브젝트의 핵심 특징을 유지한
# 완전히 새로운 그림책용 오브젝트 캐릭터를 만들도록 유도한다.
BASE_INSTRUCTION = (
    "Use the reference image only to extract character identity cues. "
    "Reimagine the subject as an object-based storybook character for children ages 6 to 8 and paint a completely new illustration. "
    "Give the character a memorable toy-like silhouette, readable facial expression, and a strong hero-character presence. "
    "Do not apply a simple filter, recolor, texture overlay, or photo retouch. "
    "Preserve recognizable identity cues from the reference image such as silhouette, face placement, standout accessories, and memorable proportions. "
    "Keep the design clearly non-human and object-based rather than turning it into a person or a humanoid child. "
    "Do not add realistic human anatomy, adult body proportions, or a full human body plan. "
    "Arms and legs are optional; if they appear, keep them tiny, simple, and secondary to the original object form."
)

# 내부 호환을 위해 boy/girl 키를 유지하지만,
# 실제 의미는 캐릭터의 성별이 아니라 어떤 취향의 아이들에게 더 어필할 연출인지에 가깝다.
BASE_STYLE_OPTIONS = ("active", "soft")
DEFAULT_PROMPT_OPTIONS = {"gender": "girl", "base_style": "active", "category": "default"}
GENDER_CATEGORY_OPTIONS = {
    "boy": ("default", "adventure", "cozy", "magic"),
    "girl": ("default", "bright", "cozy", "fantasy"),
}


def fallback_character_sheet(vision_result: dict[str, Any], parent_input: dict[str, Any]) -> dict[str, Any]:
    """텍스트 모델이 실패해도 최소한의 캐릭터 시트를 안정적으로 만든다."""

    objects = vision_result.get("objects") or []
    original_object_hint = str(parent_input.get("original_object_hint") or "").strip()
    original_object = original_object_hint or (str(objects[0]) if objects else "unknown object")
    name = str(parent_input.get("name") or "Coco")
    job = str(parent_input.get("job") or "friend of the stars")
    personality = str(parent_input.get("personality") or "warm and brave")
    goal = str(parent_input.get("goal") or "wants to discover something new")
    extra = str(parent_input.get("extra_description") or "").strip()
    traits_input = str(parent_input.get("traits_input") or "").strip()
    tone = str(parent_input.get("tone") or "warm").strip()

    traits = [
        f"based on the appearance of {original_object}",
        "keeps the same friendly silhouette as the reference image",
        f"wears visual hints of the job: {job}",
    ]
    if traits_input:
        traits.extend(part.strip() for part in traits_input.split(",") if part.strip())
    if extra:
        traits.append(extra)

    return {
        "original_object": original_object,
        "name": name,
        "job": job,
        "personality": personality,
        "goal": goal,
        "core_visual_traits": traits[:5],
        "tone": tone,
    }


def validate_character_sheet(
    data: dict[str, Any],
    vision_result: dict[str, Any],
    parent_input: dict[str, Any],
) -> dict[str, Any]:
    """모델 출력과 fallback 값을 병합해 최종 스키마를 안정화한다."""

    fallback = fallback_character_sheet(vision_result, parent_input)
    merged = {**fallback, **data}

    core_traits = merged.get("core_visual_traits")
    if not isinstance(core_traits, list):
        core_traits = fallback["core_visual_traits"]

    cleaned = {
        "original_object": str(merged.get("original_object") or fallback["original_object"]),
        "name": str(merged.get("name") or fallback["name"]),
        "job": str(merged.get("job") or fallback["job"]),
        "personality": str(merged.get("personality") or fallback["personality"]),
        "goal": str(merged.get("goal") or fallback["goal"]),
        "core_visual_traits": [str(item).strip() for item in core_traits if str(item).strip()]
        or fallback["core_visual_traits"],
        "tone": str(merged.get("tone") or fallback["tone"]),
    }
    return CharacterSheet.model_validate(cleaned).model_dump()


def build_character_sheet(
    vision_result: dict[str, Any],
    parent_input: dict[str, Any],
    gemini_service: GeminiService,
) -> dict[str, Any]:
    """비전 결과와 사용자 입력을 바탕으로 최종 character sheet를 만든다."""

    prompt = character_sheet_prompt(vision_result, parent_input)
    try:
        raw_text = gemini_service.generate_text(prompt)
        parsed = safe_json_loads(raw_text)
    except Exception:
        parsed = {}
    return validate_character_sheet(parsed, vision_result, parent_input)


def normalize_prompt_options(prompt_options: dict[str, Any] | None = None) -> dict[str, str]:
    """프롬프트 선택값을 정리해 예전 입력 형식과도 호환되게 만든다."""

    raw = prompt_options or {}
    gender = str(raw.get("gender") or DEFAULT_PROMPT_OPTIONS["gender"]).strip().lower()
    if gender not in GENDER_CATEGORY_OPTIONS:
        gender = DEFAULT_PROMPT_OPTIONS["gender"]

    base_style = str(raw.get("base_style") or DEFAULT_PROMPT_OPTIONS["base_style"]).strip().lower()
    if base_style not in BASE_STYLE_OPTIONS:
        base_style = DEFAULT_PROMPT_OPTIONS["base_style"]

    category = str(raw.get("category") or DEFAULT_PROMPT_OPTIONS["category"]).strip().lower()
    if category in {"", "none", "basic", "base"}:
        category = "default"
    if category not in GENDER_CATEGORY_OPTIONS[gender]:
        category = "default"

    return {"gender": gender, "base_style": base_style, "category": category}


def category_options_for_gender(gender: str) -> list[str]:
    """선택된 취향 키에 맞는 카테고리 목록을 반환한다."""

    normalized_gender = normalize_prompt_options({"gender": gender})["gender"]
    return list(GENDER_CATEGORY_OPTIONS[normalized_gender])


def prompt_template_name(prompt_options: dict[str, Any] | None = None) -> str:
    """현재 선택값이 어떤 템플릿 키로 해석되는지 반환한다."""

    normalized = normalize_prompt_options(prompt_options)
    if normalized["category"] != "default":
        return f"{normalized['gender']}_{normalized['category']}"
    return f"{normalized['gender']}_{normalized['base_style']}"


def _joined_traits(character_sheet: dict[str, Any]) -> str:
    """핵심 시각 특징 배열을 사람이 읽기 쉬운 한 줄 설명으로 합친다."""

    traits = [str(item).strip() for item in character_sheet.get("core_visual_traits", []) if str(item).strip()]
    return ", ".join(traits) if traits else "friendly toy-like silhouette, clear outfit cues"


def _prompt_subject(character_sheet: dict[str, Any]) -> str:
    """모든 이미지 프롬프트가 공유하는 캐릭터 핵심 설명 블록을 만든다."""

    name = str(character_sheet.get("name") or "This character")
    job = str(character_sheet.get("job") or "storybook hero")
    original_object = str(character_sheet.get("original_object") or "reference object")
    traits = _joined_traits(character_sheet)
    return (
        f"{name}, a {job}, should preserve the same silhouette, face placement, body shape, and distinctive details "
        f"of the {original_object} while becoming a clearly illustrated object-based character. "
        f"Keep these defining traits visible: {traits}. "
        "The result should read as a stylized character version of the object itself, not as a human wearing or holding the object."
    )


# 아래 프롬프트 블록들은 그림체와 비율을 프롬프트 내부에 직접 고정하기 위한 구간이다.
# 동시에 사람형 캐릭터로 흐르지 않도록 "오브젝트 기반 캐릭터" 조건을 반복해서 강조한다.
def build_boy_active_prompt(character_sheet: dict[str, Any]) -> str:
    """남자아이 취향의 기본 active 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object hero designed to strongly appeal to boys who enjoy adventurous main characters, "
        "using a dynamic adventurous pose with strong action readability, clear motion, bright lighting, bold simple background shapes, "
        "and a fun hero-story composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human child, superhero kid, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_boy_soft_prompt(character_sheet: dict[str, Any]) -> str:
    """남자아이 취향의 기본 soft 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to appeal to boys who enjoy warm and playful everyday stories, "
        "using a relaxed everyday pose, cozy lighting, simple playful background elements, a calm readable composition, "
        "and a soft daily-life story atmosphere. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human child, school kid, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_girl_active_prompt(character_sheet: dict[str, Any]) -> str:
    """여자아이 취향의 기본 active 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to strongly appeal to girls who enjoy bright and confident main-character energy, "
        "using a lively heroic pose with clear motion, bright lighting, colorful simple background shapes, and a fun main-character composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human girl, doll-like child, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_girl_soft_prompt(character_sheet: dict[str, Any]) -> str:
    """여자아이 취향의 기본 soft 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to appeal to girls who enjoy warm, gentle, and lovable story moments, "
        "using a warm gentle pose with cozy lighting, simple dreamy background elements, a clear readable composition, "
        "and a tender everyday story atmosphere. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human girl, doll-like child, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


# 카테고리 프롬프트는 같은 그림체를 유지한 채 장면 연출만 바꾸기 위한 블록이다.
# 선화, 채색, 얼굴 구조, 비율은 고정하고 포즈와 조명, 배경 모양만 카테고리별로 달라지게 한다.
def build_boy_adventure_prompt(character_sheet: dict[str, Any]) -> str:
    """남자아이 취향의 adventure 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object hero designed to strongly appeal to boys who enjoy lively adventure stories, "
        "using a dynamic adventurous pose with strong action readability, bright lighting, bold simple background shapes, "
        "and a fun hero-story composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human child, superhero kid, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_boy_cozy_prompt(character_sheet: dict[str, Any]) -> str:
    """남자아이 취향의 cozy 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to appeal to boys who enjoy warm and playful daily-life stories, "
        "using a relaxed everyday pose with cozy lighting, simple playful background elements, a calm readable composition, "
        "and a soft daily-life story atmosphere. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human child, school kid, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_boy_magic_prompt(character_sheet: dict[str, Any]) -> str:
    """남자아이 취향의 magic 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object hero designed to appeal to boys who enjoy curious and mysterious magical adventures, "
        "using a magical pose with dreamy glowing accents, whimsical readable background shapes, soft fantasy lighting, "
        "and a wonder-filled story composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human child, wizard kid, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_girl_bright_prompt(character_sheet: dict[str, Any]) -> str:
    """여자아이 취향의 bright 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to strongly appeal to girls who enjoy bright, confident, and cheerful main characters, "
        "using a lively heroic pose with clear motion, bright lighting, colorful simple background shapes, and a fun main-character composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human girl, fashion doll child, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_girl_cozy_prompt(character_sheet: dict[str, Any]) -> str:
    """여자아이 취향의 cozy 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to appeal to girls who enjoy soft, lovable, and comforting story moments, "
        "using a warm gentle pose with cozy lighting, simple dreamy background elements, a clear readable composition, "
        "and a tender everyday story atmosphere. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human girl, doll-like child, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


def build_girl_fantasy_prompt(character_sheet: dict[str, Any]) -> str:
    """여자아이 취향의 fantasy 카테고리 프롬프트를 만든다."""

    subject = _prompt_subject(character_sheet)
    return (
        f"{BASE_INSTRUCTION} "
        f"{subject} "
        "Render the character as a toy-like object protagonist designed to appeal to girls who enjoy fairytale fantasy and sparkling storybook magic, "
        "using a magical princess-like pose with glowing accents, whimsical but simple fantasy background shapes, soft sparkling lighting, "
        "and a clear fairytale composition. "
        "Use clean soft linework, gentle cel shading, rounded simplified shapes, slightly oversized head proportions, "
        "large expressive eyes or eye-like facial features, child-friendly facial construction, bright cheerful but balanced colors, "
        "and if small limbs are present keep them short, simple, and non-human, and a polished digital picture-book finish. "
        "Keep the exact same line style, shading style, proportions, and facial construction. "
        "Do not turn the character into a human girl, princess child, or humanoid mascot. "
        "Do not switch to watercolor, rough sketch, painterly textures, photorealism, semi-realism, dark fantasy, horror, or mixed art styles."
    )


BASE_TEMPLATE_BUILDERS = {
    "boy_active": build_boy_active_prompt,
    "boy_soft": build_boy_soft_prompt,
    "girl_active": build_girl_active_prompt,
    "girl_soft": build_girl_soft_prompt,
}

CATEGORY_TEMPLATE_BUILDERS = {
    "boy_adventure": build_boy_adventure_prompt,
    "boy_cozy": build_boy_cozy_prompt,
    "boy_magic": build_boy_magic_prompt,
    "girl_bright": build_girl_bright_prompt,
    "girl_cozy": build_girl_cozy_prompt,
    "girl_fantasy": build_girl_fantasy_prompt,
}


def build_selected_prompt(character_sheet: dict[str, Any], prompt_options: dict[str, Any] | None = None) -> tuple[str, str]:
    """현재 선택값으로 어떤 템플릿이 쓰였는지와 최종 프롬프트를 함께 반환한다."""

    normalized = normalize_prompt_options(prompt_options)
    template_name = prompt_template_name(normalized)
    builder = CATEGORY_TEMPLATE_BUILDERS.get(template_name) or BASE_TEMPLATE_BUILDERS[template_name]
    return template_name, builder(character_sheet)


def build_prompt_preview(character_sheet: dict[str, Any], prompt_options: dict[str, Any] | None = None) -> dict[str, str]:
    """Streamlit 테스트 페이지에서 보여줄 프롬프트 미리보기 정보를 만든다."""

    normalized = normalize_prompt_options(prompt_options)
    template_name, selected_prompt = build_selected_prompt(character_sheet, normalized)
    return {
        "gender": normalized["gender"],
        "base_style": normalized["base_style"],
        "category": normalized["category"],
        "selected_template": template_name,
        "selected_prompt": selected_prompt,
    }


def fallback_style_prompts(character_sheet: dict[str, Any], prompt_options: dict[str, Any] | None = None) -> dict[str, str]:
    """LLM 없이도 일정한 품질로 style prompt를 만들기 위한 로컬 템플릿 함수."""

    normalized = normalize_prompt_options(prompt_options)
    gender = normalized["gender"]

    active_prompt = BASE_TEMPLATE_BUILDERS[f"{gender}_active"](character_sheet)
    soft_prompt = BASE_TEMPLATE_BUILDERS[f"{gender}_soft"](character_sheet)

    if normalized["category"] != "default":
        category_key = f"{gender}_{normalized['category']}"
        category_prompt = CATEGORY_TEMPLATE_BUILDERS[category_key](character_sheet)
        if normalized["base_style"] == "soft":
            soft_prompt = category_prompt
        else:
            active_prompt = category_prompt

    return {"active_style": active_prompt, "soft_style": soft_prompt}


def validate_style_prompts(
    data: dict[str, Any],
    character_sheet: dict[str, Any],
    prompt_options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """외부 응답이 있더라도 최종적으로 API 형식에 맞는 style prompt 두 개를 보장한다."""

    fallback = fallback_style_prompts(character_sheet, prompt_options)
    merged = {**fallback, **(data or {})}
    return {
        "active_style": str(merged.get("active_style") or fallback["active_style"]).strip(),
        "soft_style": str(merged.get("soft_style") or fallback["soft_style"]).strip(),
    }


def build_style_prompts(
    character_sheet: dict[str, Any],
    _gemini_service: GeminiService,
    prompt_options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """최종 active/soft 이미지 프롬프트를 deterministic 템플릿으로 만든다."""

    return validate_style_prompts({}, character_sheet, prompt_options)
