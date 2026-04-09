# JHPark 최신 인자값(Prompt Options) 통합 계획

JHPark 폴더의 `character.py` 및 프론트엔드 구성을 바탕으로 `unified_app`에도 상세 캐릭터/스타일 제어 인자(`gender`, `base_style`, `category`)를 통합합니다.

## Proposed Changes

### 1. 백엔드 파이프라인 수정

#### [MODIFY] [pipeline.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/app/pipeline.py)
- `build_style_prompts` 함수가 `prompt_options` 인자를 받도록 수정합니다.
- `run_pipeline` 함수가 `prompt_options`를 받아 스타일 프롬프트 생성 시 사용하도록 수정합니다.

### 2. API 엔드포인트 수정

#### [MODIFY] [main.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/main.py)
- `CreateArtPayload` 및 `GenerateBookPayload` 모델에 `prompt_options: Optional[PromptOptions] = None` 필드를 추가합니다.
- `/create-art`에서 전달받은 `prompt_options`를 파이프라인으로 전달합니다.
- `/generate-book`에서도 `prompt_options`를 사용하여 스타일 프롬프트를 생성한 뒤 동화책 생성(`app_generate_story`)에 활용합니다.

### 3. 프론트엔드 UI/UX 수정

#### [MODIFY] [index.html](file:///c:/project_3rd/PO3_objectodyssey/unified_app/static/index.html)
- **UI 요소 추가**:
    - **타겟 취향**: 남자아이(Boy), 여자아이(Girl) 선택 드롭다운
    - **기본 스타일**: Active, Soft 선택 드롭다운
    - **카테고리**: 성별 취향에 따른 하위 카테고리 드롭다운 (Adventure, Cozy, Magic, Bright, Fantasy 등)
- **JavaScript 로직 수정**:
    - `startGenerateArt` 및 `startStoryGen` 함수에서 새로 추가된 UI 값을 수집하여 서버로 전송합니다.
    - 성별 선택 시 해당 성별에 맞는 카테고리 목록이 나타나도록 동적 처리합니다.

## Verification Plan

### Automated Tests
- `main.py`를 실행하고 브라우저에서 새로운 인자값들이 포함된 페이로드로 캐릭터 아트 및 동화책이 정상적으로 생성되는지 확인합니다.
- 서버 로그에서 `prompt_options`가 올바르게 전달되는지 디버그 메시지를 확인합니다.

### Manual Verification
- 프론트엔드 UI에서 '취향'이나 '카테고리'를 바꿨을 때 생성되는 이미지의 분위기나 프롬프트가 달라지는지 확인합니다.
