# 보관함 기능 오류 해결 및 코드 정리 계획

사용자의 "보관함이 안 열려요" 및 기타 잠재적 버그를 해결하기 위한 계획입니다.

## 1. 개요
현재 보관함(Library) 기능에서 몇 가지 문제점이 발견되었습니다.
- 보관함에서 동화책을 선택했을 때 화면이 동화책 보기 화면(`screen-story`)으로 전환되지 않는 문제.
- `index.html`에 `saveCurrentStory` 함수가 중복 정의되어 있어 로직 혼선 및 잠재적 오류 발생 가능성.
- 백엔드(`/get-stories`)에서 동화책의 표지 이미지 경로를 반환하지 않아 목록에서 이미지가 안 보일 수 있는 문제.

## 2. 전체 서비스 플로우
1. **성별 선택 (`screen-gender`)**: 앱 시작 시 아이의 성별을 선택하여 테마 설정.
2. **홈 화면 (`screen-home`)**: 모험 시작, 목소리 관리, 보관함 가기 버튼 제공.
3. **캐릭터 생성 (`screen-input` -> `screen-loading`)**: 장난감 사진 촬영/업로드 및 캐릭터 정보 입력.
4. **캐릭터 결과 (`screen-result`)**: 생성된 AI 캐릭터 확인 및 읽어줄 목소리 선택.
5. **동화 생성 및 감상 (`screen-loading` -> `screen-story`)**: AI가 이야기를 지어내고 페이지별 이미지/음성 생성.
6. **보관함 (`screen-library`)**: 이전에 저장된 동화 목록 확인 및 다시 읽기.
7. **목소리 클론 (`screen-voice-clone`)**: 아이나 부모의 목소리를 녹음하여 저장 및 삭제.

## 3. 상세 수정 계획

### [Component] Backend (main.py)
#### [MODIFY] [main.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/main.py)
- `/get-stories` 엔드포인트 수정: 응답 데이터에 `cover_image_path` 추가. (목록에서 썸네일을 정확히 보여주기 위함)

### [Component] Frontend (index.html)
#### [MODIFY] [index.html](file:///c:/project_3rd/PO3_objectodyssey/unified_app/static/index.html)
- **중복 함수 제거**: 1616라인 부근의 구형 `saveCurrentStory`를 제거하고, 1770라인의 최신 버전을 개선하여 통합.
- **보관함 전환 로직 수정**: `renderSavedStory` 함수 마지막에 `showScreen('screen-story')`를 추가하여 동화책 선택 시 화면이 즉시 전환되도록 수정.
- **코드 정리**: 불필요한 콘솔 로그 정리 및 오류 메시지 한글화 점검.

## 4. 검증 계획
### 자동화 테스트
- `python -m py_compile main.py`를 통해 백엔드 문법 체크. (UV 환경 활용)

### 수동 테스트
1. 홈 화면에서 '내 보관함 가기' 버튼을 눌러 보관함 화면이 뜨는지 확인.
2. 보관함 목록 중 하나를 클릭했을 때 동화책 화면으로 정상 전환되는지 확인.
3. 새로운 동화를 생성한 후 보관함에 정상 저장되고 목록에 업데이트되는지 확인.

## 5. 수정 완료 (Implementation Result) - 2026-04-09
- **백엔드**: `main.py`의 `/get-stories` 응답에 `cover_image_path`를 추가하여 보관함 썸네일 표시 준비 완료.
- **프론트엔드**: `index.html`에서 중복된 `saveCurrentStory` 함수를 제거하여 코드 최적화 및 로직 일원화.
- **버그 수정**: `renderSavedStory` 함수 마지막에 `showScreen('screen-story')`를 추가하여 보관함에서 동화책 선택 시 화면이 즉시 전환되도록 해결.
- **안정성**: `python -m py_compile`을 통한 서버 문법 검증 완료.
