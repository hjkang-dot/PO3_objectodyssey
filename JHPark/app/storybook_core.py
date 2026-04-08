"""동화책 생성 관련 함수를 한곳에 모아 둔 핵심 모듈.

이 파일은 다음 역할을 한 번에 담당한다.

1. 동화 생성용 프롬프트 조립
2. 생성 결과(JSON) 정규화와 검증
3. 모델 출력이 불안정할 때 사용할 fallback 동화 생성
4. 페이지 이미지와 표지 이미지 프롬프트 조립
5. 최종적으로 5페이지 동화 패키지 생성

앞으로 동화 기능을 수정할 때는 가능하면 이 파일만 먼저 보면 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models import ALLOWED_STORY_TONES, CharacterSheet, StoryPackageResponse
from app.services.openai_service import OpenAIService
from app.utils import PROJECT_ROOT, safe_json_loads

if TYPE_CHECKING:
    from app.services.gemini_service import GeminiService

DEFAULT_STORY_TONE = "따뜻한"
FALLBACK_TTS_TONES = [
    "상냥하게 기대하며",
    "두근거리며",
    "조금 긴장하며",
    "안도하며",
    "밝고 따뜻하게",
]


# 아래 프롬프트 함수들은 동화의 구조와 톤을 사람이 읽기 쉽게 분리해 둔 영역이다.
# 실제 모델에게는 최종적으로 story_generation_prompt()에서 하나의 문자열로 합쳐서 전달한다.


def build_common_story_rules() -> str:
    """모든 동화 스타일에 공통으로 적용할 규칙을 반환한다."""

    return """
[공통 규칙]
- 반드시 JSON만 반환한다. 마크다운, 코드블록, 설명문은 금지한다.
- JSON 키는 정확히 title, story_pages, tts_script, choices 만 사용한다.
- 주인공은 character_sheet의 기존 캐릭터와 동일한 존재로 유지한다.
- name, original_object, job, personality, goal 정보를 이야기 전개에 자연스럽게 반영한다.
- 대상 독자는 6세에서 8세 아동이다.
- 전체 이야기는 정확히 15문장이다.
- story_pages는 정확히 5개이며, 각 페이지는 정확히 3문장이다.
- page_number는 1, 2, 3, 4, 5 순서로 고정한다.
- 이야기 전체는 반드시 기승전결 구조를 가져야 한다.
- 1페이지는 주인공 소개와 흥미로운 사건의 시작으로 독자의 집중을 끌어야 한다.
- 2페이지는 목표를 향한 첫 행동과 기대감을 선명하게 보여야 한다.
- 3페이지는 가장 큰 문제, 실수, 갈등, 장애물 중 하나를 분명하게 제시해야 한다.
- 4페이지는 주인공이 스스로 문제를 해결하거나 중요한 도움을 받아 전환점을 만들어야 한다.
- 5페이지는 감정적으로 만족스러운 결말과 여운, 그리고 다음 상상을 부르는 마무리여야 한다.
- 모든 페이지는 앞뒤 인과관계가 이어져야 하며, 갑작스러운 장면 점프를 피한다.
- 문장은 짧고 또렷하게 쓰되 너무 단조롭지 않게 리듬을 살린다.
- 아이가 듣는 동안 집중할 수 있도록 의성어와 의태어를 적절히 사용한다.
- 너무 추상적이거나 교훈만 설명하는 문장은 피하고, 행동과 장면으로 보여준다.
- title은 짧고 기억하기 쉬워야 한다.
- tts_script는 문장 단위 배열이며, 각 항목은 line 과 tone 만 가진다.
- tts_script는 15문장 전체를 처음부터 끝까지 빠짐없이 덮어야 한다.
- tone은 낭독 방식 설명이어야 한다. 예: 상냥하게, 조용히 기대하며, 신나게, 살짝 긴장감 있게
- choices는 정확히 2개이며, 각 항목은 id 와 text 만 가진다.
- choice id는 snake_case 영어여야 한다.
- choice text는 아이가 이야기 다음 행동을 상상하며 고를 수 있게 짧고 분명해야 한다.
- 각 page의 image_prompt는 한 문단 영어 프롬프트로 작성한다.
- image_prompt는 같은 캐릭터, 같은 그림체, 같은 얼굴 구조, 같은 색감, 같은 세계관을 유지해야 한다.
- image_prompt에는 글자, 자막, 말풍선, 간판 문구, 페이지 번호를 넣지 않는다.
- image_prompt에는 no text, no letters, no words, no typography, no speech bubbles, no captions 를 명시한다.
""".strip()


def normalize_extra_prompt(extra_prompt: str) -> str:
    """사용자 추가 요청이 비어 있을 때도 프롬프트 문장이 무너지지 않게 정리한다."""

    return extra_prompt.strip() or "없음"


def format_character_sheet(character_sheet: dict[str, Any]) -> str:
    """캐릭터 시트를 프롬프트 본문에 넣기 쉬운 형태로 정리한다."""

    return f"""
[캐릭터 정보]
{character_sheet}
""".strip()


def build_warm_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """따뜻한 톤 동화의 감정선과 장면 방향을 만든다."""

    return f"""
[스타일]
- 전체 분위기는 다정하고 포근한 "따뜻한" 톤으로 만든다.
- 갈등이 있더라도 무섭지 않게 만들고, 안심할 수 있는 감정선으로 회복시킨다.
- 배경은 집, 골목, 작은 공원, 놀이터, 밤하늘 같은 친숙한 공간을 활용한다.
- 놀람, 두근거림, 안도감이 자연스럽게 이어지도록 장면 전환을 만든다.
- 마지막 장면은 마음이 편안해지고 미소가 남는 결말로 마무리한다.
- choices도 아이가 상상 놀이를 이어가기 쉬운 방향으로 만든다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_adventure_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """모험적인 톤 동화의 속도감과 해결 흐름을 만든다."""

    return f"""
[스타일]
- 전체 분위기는 신나고 앞으로 나아가는 "모험적인" 톤으로 만든다.
- 출발, 탐색, 위기, 해결, 귀환의 흐름이 자연스럽게 이어지게 한다.
- 배경은 숲길, 하늘길, 바다 가장자리, 별빛 공간처럼 상상력이 살아나는 장소를 사용한다.
- 위험은 과하게 무섭지 않게, 하지만 아이가 손에 땀을 쥘 만큼은 분명하게 보여준다.
- 주인공이 용기, 재치, 관찰력 중 하나로 문제를 푸는 장면을 꼭 넣는다.
- 마지막은 성취감과 다음 모험의 기대를 함께 남긴다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_lesson_story_style(character_sheet: dict[str, Any], extra_prompt: str = "") -> str:
    """교훈적인 톤 동화의 메시지를 설교조가 아니라 사건 중심으로 유지한다."""

    return f"""
[스타일]
- 전체 분위기는 또렷하고 이해하기 쉬운 "교훈적인" 톤으로 만든다.
- 교훈을 직접 설명하기보다 사건과 선택의 결과로 깨닫게 한다.
- 약속, 배려, 용기, 책임, 정직 같은 가치를 행동으로 드러낸다.
- 주인공이 한 번쯤 망설이거나 실수하지만, 스스로 더 좋은 선택을 하게 만든다.
- 마지막은 뿌듯함과 배움이 남도록 부드럽게 정리한다.
- choices는 배운 점을 다음 행동으로 이어 볼 수 있게 만든다.

[추가 요청]
{normalize_extra_prompt(extra_prompt)}

{format_character_sheet(character_sheet)}
""".strip()


def build_story_style_guide(character_sheet: dict[str, Any], story_tone: str, extra_prompt: str = "") -> str:
    """선택한 tone에 맞는 세부 스타일 가이드를 반환한다."""

    style_map = {
        "따뜻한": build_warm_story_style,
        "모험적인": build_adventure_story_style,
        "교훈적인": build_lesson_story_style,
    }
    guide_builder = style_map.get(story_tone, build_warm_story_style)
    return guide_builder(character_sheet, extra_prompt=extra_prompt)


def story_generation_prompt(
    character_sheet: dict[str, Any],
    extra_prompt: str = "",
    story_tone: str | None = None,
) -> str:
    """동화 생성 모델에 전달할 최종 프롬프트를 조립한다."""

    selected_style = story_tone or character_sheet.get("tone") or ALLOWED_STORY_TONES[0]
    style_guide = build_story_style_guide(character_sheet, selected_style, extra_prompt=extra_prompt)

    return f"""
너는 6세에서 8세 아이를 위한 참여형 동화를 만드는 작가야.

[사용 가능한 스타일]
- {", ".join(ALLOWED_STORY_TONES)}

{build_common_story_rules()}

{style_guide}

[출력 품질 강화]
- 각 페이지의 3문장은 한 장면 안에서 자연스럽게 이어져야 한다.
- 페이지가 바뀔 때마다 사건이 실제로 한 단계씩 진전되어야 한다.
- 1페이지는 호기심, 2페이지는 기대감, 3페이지는 긴장감, 4페이지는 안도와 돌파, 5페이지는 기쁨과 여운이 느껴지게 한다.
- 우연히 해결되는 전개를 피하고, 주인공의 성격과 목표가 해결 과정에 기여하게 한다.
- 모든 page의 image_prompt는 그 페이지 문장을 그림으로 옮긴 장면 설명이어야 한다.
- image_prompt는 영어 한 문단으로 쓰고, 아이용 그림책 장면처럼 명확한 행동과 배경을 포함해야 한다.
- image_prompt 안에는 절대 글자 요소를 넣지 않는다.
- 표지용으로도 활용될 수 있도록 title은 시각적으로 기억하기 좋은 짧은 제목으로 만든다.
""".strip()


def story_prompt(character_sheet: dict[str, Any], extra_prompt: str = "", story_tone: str | None = None) -> str:
    """기존 호출부와의 호환을 위한 별칭 함수."""

    return story_generation_prompt(character_sheet, extra_prompt=extra_prompt, story_tone=story_tone)


# 아래부터는 실제 생성 결과를 다루는 파이프라인 함수들이다.
# 이 블록은 모델 출력이 불안정해도 앱 전체가 죽지 않도록 정규화, 검증, fallback을 담당한다.


def _normalize_story_character_sheet(character_sheet: dict[str, Any]) -> dict[str, Any]:
    """스토리 생성 전에 캐릭터 시트를 검증하고 tone 값을 안정화한다."""

    validated = CharacterSheet.model_validate(character_sheet).model_dump()
    if validated["tone"] not in ALLOWED_STORY_TONES:
        validated["tone"] = DEFAULT_STORY_TONE
    return validated


def _resolve_story_tone(character_sheet: dict[str, Any], story_tone: str | None) -> str:
    """최종적으로 어떤 동화 tone을 쓸지 결정한다."""

    if story_tone in ALLOWED_STORY_TONES:
        return story_tone
    if story_tone in (None, "", "랜덤"):
        return character_sheet.get("tone") or DEFAULT_STORY_TONE
    return character_sheet["tone"]


def build_story_prompt(character_sheet: dict[str, Any], extra_prompt: str = "", story_tone: str | None = None) -> str:
    """외부에서 바로 쓸 수 있는 공개용 스토리 프롬프트 조립 함수."""

    normalized = _normalize_story_character_sheet(character_sheet)
    resolved_tone = _resolve_story_tone(normalized, story_tone)
    return story_generation_prompt(normalized, extra_prompt=extra_prompt, story_tone=resolved_tone)


def _split_sentences_from_text(page_text: str) -> list[str]:
    """문단 문자열이 들어왔을 때 문장 배열로 최대한 복구한다."""

    normalized = (
        page_text.replace("!", ".")
        .replace("?", ".")
        .replace("。", ".")
        .replace("\n", " ")
        .strip()
    )
    parts = [part.strip() for part in normalized.split(".") if part.strip()]
    return [f"{part}." for part in parts]


def _normalize_story_package_structure(data: dict[str, Any]) -> dict[str, Any]:
    """모델이 약간 다른 구조를 주더라도 앱이 이해할 수 있는 형태로 맞춘다."""

    pages = data.get("story_pages") or []
    normalized_pages: list[dict[str, Any]] = []

    for idx, page in enumerate(pages, start=1):
        raw_sentences = page.get("sentences", []) if isinstance(page, dict) else []
        sentences = [str(item).strip() for item in raw_sentences if str(item).strip()]
        page_text = str(page.get("page_text") or " ".join(sentences)).strip() if isinstance(page, dict) else ""
        if not sentences and page_text:
            sentences = _split_sentences_from_text(page_text)

        normalized_pages.append(
            {
                "page_number": int(page.get("page_number") or idx) if isinstance(page, dict) else idx,
                "sentences": sentences,
                "page_text": page_text or " ".join(sentences),
                "image_prompt": str(page.get("image_prompt") or "").strip() if isinstance(page, dict) else "",
                "image_path": page.get("image_path") if isinstance(page, dict) else None,
            }
        )

    normalized = dict(data)
    normalized["story_pages"] = normalized_pages
    normalized["story_paragraphs"] = [page["page_text"] for page in normalized_pages]
    return normalized


def _validate_story_package(data: dict[str, Any]) -> dict[str, Any]:
    """정규화된 story package가 앱 규칙을 만족하는지 최종 검증한다."""

    normalized_data = _normalize_story_package_structure(data)
    story_package = StoryPackageResponse.model_validate(normalized_data).model_dump()

    combined_story = " ".join(story_package["story_paragraphs"])
    combined_tts = " ".join(item["line"] for item in story_package["tts_script"])

    if len(story_package["story_pages"]) != 5:
        raise ValueError("Story package must contain exactly 5 pages.")

    sentence_count = sum(len(page["sentences"]) for page in story_package["story_pages"])
    if sentence_count != 15:
        raise ValueError("Story package must contain exactly 15 sentences across 5 pages.")

    if len(story_package["tts_script"]) < 15:
        raise ValueError("TTS script must cover all 15 story sentences.")

    if len(combined_tts) < max(40, len(combined_story) // 2):
        raise ValueError("TTS script is too short to cover the full story.")

    return story_package


def _build_fallback_sentences(character_sheet: dict[str, Any], story_tone: str) -> list[list[str]]:
    """모델 응답이 비어 있거나 망가졌을 때 사용할 5페이지 fallback 문장을 만든다."""

    name = str(character_sheet.get("name") or "코코").strip()
    original_object = str(character_sheet.get("original_object") or "작은 물건").strip()
    job = str(character_sheet.get("job") or "탐험가").strip()
    personality = str(character_sheet.get("personality") or "상냥하고 공감이 많은").strip()
    goal = str(character_sheet.get("goal") or "새로운 곳을 발견하는 것").strip()

    if story_tone == "모험적인":
        return [
            [
                f"{name}는 {original_object}에서 태어난 {job}답게 반짝반짝 새벽빛 속에서 오늘의 탐험을 준비했어요.",
                f'"와, 오늘은 꼭 {goal}!" 하고 {name}가 씩 웃자 마음은 두근두근 흔들렸어요.',
                f"그때 멀리서 길을 가리키는 작은 별지도가 팔랑 나타나며 첫 모험의 문을 열었어요.",
            ],
            [
                f"{name}는 {personality} 마음을 다잡고 별지도를 따라 씽씽 앞으로 나아갔어요.",
                f"풀잎 사이에서는 사각사각 바람 소리가 나고, 앞쪽 언덕에서는 반짝이는 빛이 손짓했어요.",
                f"{name}는 저 빛 너머에 정말 특별한 비밀이 있을지 궁금해서 한 걸음 더 나아갔어요.",
            ],
            [
                f"하지만 언덕 아래 다다르자 커다란 안개가 훅 내려와 길이 한순간에 모두 사라지고 말았어요.",
                f'{name}는 "이대로라면 길을 잃겠어!" 하고 멈칫했지만 쿵쾅거리는 심장 소리 속에서도 주변을 살폈어요.',
                f"바로 그때 안개 속에서 작은 울음소리가 들려와 누군가 더 큰 곤란에 빠졌다는 걸 알게 되었어요.",
            ],
            [
                f"{name}는 겁을 꾹 누르고 빛나는 눈으로 주위를 비추며 울음소리를 따라 조심조심 움직였어요.",
                f"안개 뒤에는 길을 잃은 별새 한 마리가 있었고, {name}는 별지도의 표시와 자신의 재치를 떠올렸어요.",
                f"마침내 {name}가 별새와 함께 올바른 길을 찾아내자 안개는 스르르 걷히고 별빛 다리가 펼쳐졌어요.",
            ],
            [
                f"{name}는 별새와 나란히 다리를 건너며 드디어 {goal}에 한 발짝 가까워졌다는 걸 느꼈어요.",
                f"건너편에는 새로운 세계의 입구가 눈부시게 열려 있었고 바람도 잘했어 하고 등을 밀어 주는 것 같았어요.",
                f"그날 밤 {name}는 다음 모험을 기대하며 내일은 또 어떤 반짝이는 일이 기다릴지 상상했어요.",
            ],
        ]

    if story_tone == "교훈적인":
        return [
            [
                f"{name}는 {original_object}에서 태어난 {job}로 오늘도 {goal}을 꿈꾸며 아침 햇살을 맞았어요.",
                f'{name}는 "천천히 잘 살피면 분명 멋진 일이 생길 거야." 하고 말했지만 마음은 벌써 콩닥콩닥 바빴어요.',
                f"마침 마을 끝에서 도움을 기다리는 작은 반짝 종이배 하나가 흔들흔들 손을 들었어요.",
            ],
            [
                f"{name}는 {personality} 성격답게 그냥 지나치지 않고 종이배에게 무슨 일이 있었는지 조용히 물어보았어요.",
                f"종이배는 소중한 별씨앗을 잃어버려 집으로 돌아가지 못한다고 했고, {name}는 함께 찾아보기로 했어요.",
                f"둘은 차근차근 발자국을 따라가며 어디서부터 일이 꼬였는지 살펴보기 시작했어요.",
            ],
            [
                f"그러다 {name}는 더 빨리 찾고 싶은 마음에 주변을 꼼꼼히 보지 않고 앞으로만 달려가 버렸어요.",
                f'바로 뒤에서 종이배가 "앗!" 하고 외쳤지만, 그만 작은 웅덩이에 별씨앗 하나가 풍덩 빠지고 말았어요.',
                f"{name}는 얼굴이 화끈해졌고 서두르면 오히려 더 소중한 것을 놓칠 수 있다는 걸 깨달았어요.",
            ],
            [
                f"{name}는 심호흡을 후 하고 내쉰 뒤 종이배에게 미안하다고 또박또박 말했어요.",
                f"그리고 이번에는 급하지 않게 주변을 하나씩 살피며 반짝 흔적을 따라 별씨앗들을 다시 모으기 시작했어요.",
                f"조금 느려 보여도 서로 힘을 합쳐 움직이자 마지막 별씨앗까지 되찾을 수 있었어요.",
            ],
            [
                f"종이배는 환하게 웃으며 {name}에게 고맙다고 말했고, {name}의 마음속도 뭉게뭉게 따뜻해졌어요.",
                f"{name}는 {goal}만큼 중요한 건 함께하는 마음과 차분한 선택이라는 걸 오래오래 기억하기로 했어요.",
                f"집으로 돌아가는 길, 둘은 다음에도 서두르지 않고 서로를 도와주자고 약속했어요.",
            ],
        ]

    return [
        [
            f"{name}는 {original_object}에서 태어난 {job}답게 포근한 아침빛 속에서 오늘의 작은 계획을 세웠어요.",
            f'{name}는 "언젠가 꼭 {goal}!" 하고 속삭였고, 마음은 몽글몽글 설렘으로 부풀었어요.',
            f"그때 창가에서 팔랑팔랑 흔들리는 지도 조각 하나가 {name}를 조용히 불러 세웠어요.",
        ],
        [
            f"{name}는 {personality} 성격답게 지도 조각을 소중히 들고 천천히 길을 따라 나아갔어요.",
            f"골목 끝 작은 공원에서는 사뿐사뿐 낙엽이 춤추고, 분수 옆에서는 반짝 물방울이 웃고 있었어요.",
            f"{name}는 오늘이 평소와는 조금 다른 특별한 날이 될 것 같아 살짝 두근거렸어요.",
        ],
        [
            f"그런데 분수 옆 바람이 갑자기 휙 불면서 지도 조각이 하늘로 붕 떠올라 버렸어요.",
            f'{name}는 "저걸 놓치면 오늘의 길도 잃어버릴지 몰라!" 하고 깜짝 놀랐어요.',
            f"높은 나뭇가지에 걸린 지도 조각은 금방이라도 멀리 날아갈 것처럼 바르르 떨리고 있었어요.",
        ],
        [
            f"{name}는 서두르지 말자고 스스로 다독이며 주변을 찬찬히 살폈어요.",
            f"곧 벤치 옆에 놓인 긴 리본 끈을 발견한 {name}는 살금살금 끈을 던져 지도 조각을 끌어내렸어요.",
            f"지도 조각이 폭신하게 품 안으로 돌아오자 바람도 스르르 잦아들고 공원은 다시 환하게 빛났어요.",
        ],
        [
            f"{name}는 조각을 펼쳐 보며 오늘의 작은 용기 덕분에 {goal}에 조금 더 가까워졌다고 느꼈어요.",
            f"돌아가는 길에는 노을빛이 사르르 내려앉았고 {name}의 발걸음도 한층 가볍고 환해졌어요.",
            f"그날 저녁 {name}는 내일 또 어떤 반짝이는 일이 기다릴지 상상하며 포근히 잠이 들었어요.",
        ],
    ]


def _build_fallback_image_prompt(
    character_sheet: dict[str, Any],
    page_number: int,
    page_text: str,
    story_tone: str,
) -> str:
    """fallback 동화용 페이지 이미지 프롬프트를 영어로 만든다."""

    name = str(character_sheet.get("name") or "the main character").strip()
    original_object = str(character_sheet.get("original_object") or "object").strip()
    job = str(character_sheet.get("job") or "storybook hero").strip()
    traits = ", ".join(character_sheet.get("core_visual_traits") or [])
    mood_map = {
        1: "gentle curiosity and introduction",
        2: "forward movement and growing expectation",
        3: "clear conflict and rising tension",
        4: "problem solving and emotional turning point",
        5: "warm resolution and hopeful afterglow",
    }
    tone_hint = {
        "따뜻한": "soft warm storybook lighting",
        "모험적인": "bright adventurous lighting",
        "교훈적인": "clear friendly picture-book lighting",
    }.get(story_tone, "soft warm storybook lighting")
    return (
        f"Illustrate page {page_number} of a children's picture book. "
        f"Show {name}, an object-based character inspired by {original_object}, working as a {job}. "
        f"Preserve the same design and visual traits: {traits}. "
        f"Create a scene with {mood_map[page_number]}, readable action, child-friendly staging, and {tone_hint}. "
        f"Depict this exact story moment: {page_text} "
        "No text, no letters, no words, no typography, no speech bubbles, no captions, no page numbers, no logos."
    ).strip()


def _build_fallback_story_package(
    character_sheet: dict[str, Any],
    story_tone: str,
    extra_prompt: str = "",
) -> dict[str, Any]:
    """모델 출력이 불안정할 때도 앱을 계속 쓸 수 있도록 완성형 동화 패키지를 만든다."""

    name = str(character_sheet.get("name") or "코코").strip()
    goal = str(character_sheet.get("goal") or "반짝이는 새로운 세계를 발견하는 것").strip()
    title_suffix = {
        "따뜻한": "와 포근한 반짝 지도",
        "모험적인": "와 별빛 다리의 비밀",
        "교훈적인": "와 차분한 용기의 약속",
    }.get(story_tone, "와 반짝이는 하루")
    title = f"{name}{title_suffix}"

    sentence_pages = _build_fallback_sentences(character_sheet, story_tone)
    story_pages: list[dict[str, Any]] = []
    tts_script: list[dict[str, str]] = []

    for idx, sentences in enumerate(sentence_pages, start=1):
        page_text = " ".join(sentences)
        story_pages.append(
            {
                "page_number": idx,
                "sentences": sentences,
                "page_text": page_text,
                "image_prompt": _build_fallback_image_prompt(character_sheet, idx, page_text, story_tone),
                "image_path": None,
            }
        )
        tone = FALLBACK_TTS_TONES[idx - 1]
        for sentence in sentences:
            tts_script.append({"line": sentence, "tone": tone})

    if story_tone == "교훈적인":
        choices = [
            {"id": "help_a_friend", "text": "다음에는 누군가를 더 도와볼까?"},
            {"id": "try_again_carefully", "text": "다음에는 더 천천히 도전해볼까?"},
        ]
    elif story_tone == "모험적인":
        choices = [
            {"id": "follow_new_light", "text": "다음 빛을 따라가 볼까?"},
            {"id": "rest_before_next_trip", "text": "잠깐 쉬고 다시 떠날까?"},
        ]
    else:
        choices = [
            {"id": "visit_the_park_again", "text": "내일 다시 공원에 가볼까?"},
            {"id": "share_the_map", "text": "친구와 지도를 함께 볼까?"},
        ]

    return {
        "title": title,
        "story_pages": story_pages,
        "story_paragraphs": [page["page_text"] for page in story_pages],
        "tts_script": tts_script,
        "choices": choices,
        "fallback_used": True,
        "fallback_reason": extra_prompt.strip() or f"Recovered from invalid model output while aiming for {goal}.",
    }


def _resolve_reference_path(reference_image: str) -> Path:
    """상대경로나 절대경로로 들어온 reference image 경로를 실제 파일 경로로 맞춘다."""

    candidate = Path(reference_image)
    resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Story illustration reference image not found: {reference_image}")
    return resolved


def _select_story_style_key(story_tone: str, style_prompts: dict[str, Any]) -> str:
    """스토리 tone에 더 잘 맞는 캐릭터 스타일 키를 고른다."""

    if story_tone == "모험적인" and style_prompts.get("active_style"):
        return "active_style"
    if style_prompts.get("soft_style"):
        return "soft_style"
    return "active_style"


def _build_story_scene_prompt(base_style_prompt: str, page: dict[str, Any], title: str) -> str:
    """페이지 한 장의 상황 이미지를 만들기 위한 최종 scene prompt를 조립한다."""

    page_text = str(page.get("page_text") or "").strip()
    image_prompt = str(page.get("image_prompt") or "").strip()
    page_number = int(page.get("page_number") or 1)
    return (
        f"{base_style_prompt} "
        f"Illustrate page {page_number} of the story titled '{title}'. "
        "Keep the exact same recurring character, same object-based anatomy, same facial construction, same costume details, "
        "same line style, same shading style, and same picture-book rendering style as the reference character image. "
        "Show one clear story moment for this page, with readable staging, visual cause-and-effect, and child-friendly storytelling clarity. "
        "Do not draw any text, letters, words, captions, speech bubbles, page numbers, logos, or typography anywhere in the image. "
        f"Story page text: {page_text} "
        f"Scene direction: {image_prompt}"
    ).strip()


def _build_story_cover_prompt(base_style_prompt: str, character_sheet: dict[str, Any], title: str) -> str:
    """내지와 같은 그림체를 유지하는 표지용 prompt를 조립한다."""

    name = str(character_sheet.get("name") or "the main character").strip()
    job = str(character_sheet.get("job") or "storybook hero").strip()
    goal = str(character_sheet.get("goal") or "a meaningful adventure").strip()
    original_object = str(character_sheet.get("original_object") or "object friend").strip()
    traits = ", ".join(character_sheet.get("core_visual_traits") or [])
    return (
        f"{base_style_prompt} "
        f"Create a children's storybook cover illustration for the title '{title}'. "
        f"Show {name}, an object-based character inspired by {original_object}, as the clear main focus. "
        f"Highlight the role of {job} and the emotional promise of {goal}. "
        f"Preserve these key visual traits: {traits}. "
        "Use a clean, iconic cover composition with one strong focal pose, a simple magical or adventurous environment, "
        "clear foreground-background separation, and a polished picture-book finish. "
        "Leave visual breathing room for future title placement, but do not render any text yourself. "
        "Do not draw any letters, words, captions, speech bubbles, logos, numbers, or typography anywhere in the image."
    ).strip()


def _generate_story_page_images(
    story_package: dict[str, Any],
    character_sheet: dict[str, Any],
    style_prompts: dict[str, Any] | None,
    reference_image: str | None,
    story_tone: str,
    gemini_service: "GeminiService",
) -> dict[str, Any]:
    """페이지 이미지 5장과 표지 1장을 만들어 동화 패키지에 채워 넣는다."""

    if not style_prompts or not reference_image or not gemini_service.is_configured:
        return story_package

    reference_path = _resolve_reference_path(reference_image)
    style_key = _select_story_style_key(story_tone, style_prompts)
    base_style_prompt = str(style_prompts.get(style_key) or "").strip()
    if not base_style_prompt:
        return story_package

    from app.image_flow import build_reference_prompt_seed, compose_final_image_prompt

    reference_seed = build_reference_prompt_seed(reference_path, style_key, gemini_service)
    enriched_pages: list[dict[str, Any]] = []

    for page in story_package["story_pages"]:
        scene_prompt = _build_story_scene_prompt(base_style_prompt, page, story_package["title"])
        final_prompt = compose_final_image_prompt(reference_seed, scene_prompt, style_key)
        image_path = gemini_service.generate_image(
            final_prompt,
            str(reference_path),
            f"story_page_{page['page_number']}",
        )
        enriched_page = dict(page)
        enriched_page["image_path"] = image_path
        enriched_pages.append(enriched_page)

    enriched_story = dict(story_package)
    enriched_story["story_pages"] = enriched_pages
    enriched_story["story_paragraphs"] = [page["page_text"] for page in enriched_pages]
    enriched_story["story_image_style"] = style_key

    cover_scene_prompt = _build_story_cover_prompt(base_style_prompt, character_sheet, story_package["title"])
    cover_final_prompt = compose_final_image_prompt(reference_seed, cover_scene_prompt, style_key)
    enriched_story["cover_prompt"] = cover_final_prompt
    enriched_story["cover_image_path"] = gemini_service.generate_image(
        cover_final_prompt,
        str(reference_path),
        "story_cover",
    )
    return enriched_story


def generate_story_package(
    character_sheet: dict[str, Any],
    extra_prompt: str = "",
    story_tone: str | None = None,
    style_prompts: dict[str, Any] | None = None,
    reference_image: str | None = None,
    gemini_service: "GeminiService" | None = None,
) -> dict[str, Any]:
    """스토리 생성 전체 흐름을 실행하는 공개 함수.

    순서는 다음과 같다.
    1. 캐릭터 시트와 tone 정리
    2. LLM용 스토리 프롬프트 조립
    3. OpenAI 응답 파싱
    4. 구조 검증
    5. 실패 시 fallback 동화 생성
    6. 가능하면 페이지 이미지와 표지 이미지까지 생성
    """

    normalized = _normalize_story_character_sheet(character_sheet)
    resolved_tone = _resolve_story_tone(normalized, story_tone)
    normalized["tone"] = resolved_tone

    prompt = build_story_prompt(normalized, extra_prompt=extra_prompt, story_tone=resolved_tone)
    service = OpenAIService()

    parsed: dict[str, Any] | None = None
    if service.is_configured:
        try:
            raw_text = service.generate_story_json(prompt)
            parsed = safe_json_loads(raw_text)
        except Exception:
            parsed = None

    if parsed is None:
        story_package = _build_fallback_story_package(normalized, resolved_tone, extra_prompt=extra_prompt)
    else:
        try:
            story_package = _validate_story_package(parsed)
        except Exception:
            story_package = _build_fallback_story_package(normalized, resolved_tone, extra_prompt=extra_prompt)

    if gemini_service is None:
        from app.services.gemini_service import GeminiService

        illustration_service = GeminiService()
    else:
        illustration_service = gemini_service
    try:
        return _generate_story_page_images(
            story_package,
            character_sheet=normalized,
            style_prompts=style_prompts,
            reference_image=reference_image,
            story_tone=resolved_tone,
            gemini_service=illustration_service,
        )
    except Exception as exc:
        raise RuntimeError(f"Story image generation failed: {exc}") from exc
