# 서비스 플로우 및 시스템 아키텍처 (Service Flow)

본 문서는 `PO3_objectodyssey` 프로젝트의 전체적인 동작 흐름과 데이터 관리 규칙을 정의합니다.

## 1. 개요 (Overview)
본 서비스는 사용자가 사물 사진을 업로드하고 대화를 통해 나만의 동화를 생성하고 보관하는 플랫폼입니다.

## 2. 레이어 관점 플로우 (Layered Flow)

### A. 사용자 계층 (Frontend/User)
- **메인화면**: 캐릭터와 대화하여 사물 사진 업로드 유도.
- **스토리 생성**: GPT-4 및 DALL-E를 활용한 실시간 동화 구성 및 이미지 생성.
- **이미지/음성 렌더링**: 서버에서 제공하는 자산 주소를 기반으로 동화책 큐레이션.
- **보관함 (Library)**: 생성된 모든 동화를 리스트 형태로 조회하고 재생.

### B. 서버 계층 (Backend/FastAPI)
- **채팅 API**: LangChain 기반 챗봇 대화 로직 처리.
- **스토리 API**: 
    - 동화 텍스트 생성 (LLM)
    - 이미지 프롬프트 생성 및 이미지 생성 (DALL-E)
    - 음성 파일 생성 (OpenAI TTS)
- **자산 서빙**: 윈도우 환경 대응을 위해 `FileResponse`를 통한 수동 자산 전송 모듈 운영.

### C. 데이터 계층 (Data/Saved Stories)
- **구조**: `unified_app/saved_stories/{story_id}/`
- **파일명 규칙 (Standardized)**:
    - `story.json`: 모든 텍스트 및 자산 주소 메타데이터.
    - `assets/thumb.png`: 보관함용 썸네일.
    - `assets/cover.png`: 동화 표지 이미지.
    - `assets/page_N.png`: 페이지별 이미지 (N은 1부터 시작).
    - `assets/audio_N.wav`: 페이지별 음성 (N은 1부터 시작).

## 3. 핵심 규칙 (Standard Rules)
1. **경로 표준화**: 모든 자산 경로는 `/stories/{folder}/assets/{filename}` 형식을 따르며, 백엔드에서 절대 경로로 매핑함.
2. **파일명 최소화**: 윈도우 경로 길이 제한(260자)을 방지하기 위해 10자 이내의 영문 파일명을 강제함.
3. **폴더명 ASCII**: 한글 폴더명은 허용되지 않으며, `story_YYYYMMDD_HHMMSS_UUID` 형식만 사용함.

---
*마지막 업데이트: 2026-04-09*
