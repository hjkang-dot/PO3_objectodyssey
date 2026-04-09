# 422 에러 해결 및 UI 버그 수정 계획

사용자가 보고한 `/save-story` 422 에러와 '속 표지'에 `**COVER_PAGE**` 문구가 노출되는 UI 버그를 해결하기 위한 계획입니다. 또한, '똘이' 이야기 저장을 다시 시도할 수 있도록 로직을 강화합니다.

## User Review Required

> [!IMPORTANT]
> - '똘이' 이야기가 현재 브라우저 화면에 여전히 떠 있는지 확인이 필요합니다. 만약 화면에 있다면, 아래 수정 사항을 적용한 후 새로고침 없이 바로 저장을 다시 시도할 수 있도록 조치하겠습니다.
> - `**COVER_PAGE**` 문구는 내부적으로 표지 위치를 잡기 위한 임시 플레이스홀더였으나, 렌더링 로직의 인덱스 계산 착오로 노출된 것으로 보입니다.

## Proposed Changes

### 1. 전용 플랜 폴더 및 플로우 관리 [workrule.md 준수]
- `PO3_objectodyssey_planlist` 폴더를 생성하여 모든 플랜 파일을 관리합니다.
- 전체 서비스 플로우를 명확히 정리하여 문서화합니다.

---

### 2. Backend (`main.py`)
- `/save-story` 엔드포인트에 상세 에러 로깅 추가 (Pydantic 에러 발생 시 원인 파악용).
- `SaveStoryPayload` 모델의 필드 타입 및 필수 여부 재점검.

#### [MODIFY] [main.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/main.py)
- `save_story` 함수 내부에 `try-except` 블록을 강화하여 에러 발생 시 상세 응답을 반환하도록 합니다.

---

### 3. Frontend (`index.html`)
- `saveCurrentStory` 함수에서 데이터 누락 여부를 엄격히 체크합니다.
- `renderBook` 함수에서 `**COVER_PAGE**` 로직을 개선하여 데이터 배열에 직접 침투하지 않도록 분리합니다.
- '똘이' 이야기를 유실하지 않도록 현재 `lastGeneratedStory` 상태를 유지하면서 저장을 다시 시도하는 버튼 로직을 점검합니다.

#### [MODIFY] [index.html](file:///c:/project_3rd/PO3_objectodyssey/unified_app/static/index.html)
- `renderBook`에서 표지와 내지 텍스트를 그리는 인덱스 로직 수정.
- `saveCurrentStory` 시 `character_name`이 정확히 전달되도록 수정.

## Open Questions

> [!QUESTION]
> 1. '똘이' 이야기가 현재 브라우저 화면에 보이고 있나요? (보이고 있다면 코드 수정 후 바로 다시 저장하기를 누를 수 있습니다.)
> 2. 422 에러 발생 시 브라우저 콘솔이나 화면에 다른 안내 문구가 있었나요?

## Verification Plan

### Automated Tests
- 브라우저 도구를 사용하여 `/save-story`에 고의로 잘못된 데이터를 보내 422 에러가 로깅되는지 확인.
- 수정된 `renderBook` 로직으로 표지(Spread 0)와 내지(Spread 1)가 정상적으로 출력되는지 확인.

### Manual Verification
- '똘이' 이야기를 임의로 생성한 후 '저장하기' 버튼을 눌러 성공적으로 `saved_stories` 폴더에 생성되는지 확인.
- 저장된 이야기를 다시 불러왔을 때 `**COVER_PAGE**` 문구가 나오지 않는지 확인.
