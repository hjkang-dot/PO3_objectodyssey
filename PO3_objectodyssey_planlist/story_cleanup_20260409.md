# 동화 저장 후 임시 파일 자동 삭제 구현 계획 (2026-04-09)

동화 생성이 완료되어 `saved_stories` 폴더에 최종 저장이 완료되면, 생성을 위해 사용되었던 임시 폴더(outputs/images, outputs/audios 등)의 파일들을 자동으로 삭제하여 시스템 저장 공간을 효율적으로 관리하도록 수정합니다.

## 수정 대상 및 이유

### 1. [unified_app/main.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/main.py)
- **수정 위치**: `/save-story` 엔드포인트 내의 `localize_asset` 함수 및 저장 완료 로직.
- **수정 이유**: 현재는 `shutil.copy2`를 사용하여 파일을 복사만 하고 원본을 남겨둡니다. 저장 성공 후 원본을 삭제하여 데이터 중복을 방지해야 합니다.
- **수정 내용**: 
    - `localize_asset` 시 원본 파일 경로를 리스트에 수집.
    - `story.json` 저장이 성공하면 수집된 경로들을 삭제.
    - 보관함(saved_stories)의 경로는 이미 assets 폴더 내를 가리키므로 삭제 후에도 조회에 문제가 없음.

## 삭제 폴더 범위

1. **static/outputs/images/**: 생성된 배경/장면 이미지들.
2. **static/outputs/audios/**: 생성된 TTS 음성 파일들.
3. **nukki/** (선택 사항): 캐릭터 추출 이미지. (사용자 확인 후 결정)

## 보관함 무결성 검증

- `localize_asset` 기능이 소스 파일을 `saved_stories/{folder}/assets/`로 안전하게 복사했는지 확인.
- `story.json` 파일의 `image_path`, `page_images`, `page_audios` 값이 새로운 `stories/` 경로로 잘 업데이트 되었는지 확인.

---
*수정 전 이 계획에 대한 승인을 부탁드립니다.*
