# JSON 파싱 에러 해결 및 백엔드 안정성 강화 계획 (2026-04-09)

사용자가 `/generate-book` 과정에서 `unexpected token '<', "IDOCTYPE" ... is not valid JSON` 에러를 겪고 있습니다. 이는 백엔드에서 에러나 타임아웃으로 인해 JSON이 아닌 HTML(500 에러 페이지 등)을 반환했기 때문입니다.

## 수정 계획 요약

1. **백엔드 (main.py) 수정**: `/generate-book` 엔드포인트 전체를 `try...except`로 보호하여 항상 JSON을 반환하게 합니다.
2. **프론트엔드 (index.html) 수정**: `safeFetch` 유틸리티 함수를 도입하여 비-JSON 응답(HTML 에러 페이지 등)을 우아하게 처리합니다.

---

## 상세 변경 사항

### 1. 백엔드 (unified_app/main.py)
- `/generate-book` 내에서 발생하는 모든 예외를 잡아 `HTTPException` 또는 직접 `{"status": "error", "detail": "..."}` JSON을 반환하도록 합니다.
- 데이터 정규화 과정(`fix_path`, `page_images` 생성 등)에서 데이터가 없을 경우에 대한 방어 로직을 추가합니다.

### 2. 프론트엔드 (unified_app/static/index.html)
- `safeFetch(url, options)` 함수 구현:
  ```javascript
  async function safeFetch(url, options) {
      const response = await fetch(url, options);
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
          return await response.json();
      }
      throw new Error(`서버 응답이 올바르지 않습니다 (Status: ${response.status})`);
  }
  ```
- 기존 `fetch().then(res => res.json())` 패턴을 `safeFetch`로 교체합니다.

---

## 전체 서비스 플로우 (v1.2) - 2026-04-09 업데이트

1. **객체 촬영/업로드**: 사용자가 사물 사진을 찍거나 업로드합니다.
2. **캐릭터 추출 (`/extract`)**: 누끼 제거를 통해 사물만 추출합니다.
3. **주인공 변신 (`/create-art`)**: 추출된 사물을 바탕으로 캐릭터 시트와 선택된 스타일의 이미지를 생성합니다.
4. **동화 생성 (`/generate-book`)**: 
    - 5페이지 동화 텍스트 생성
    - 장면 일러스트 및 표지 생성
    - 페이지별 TTS 음성 생성 (부모 목소리 반영 가능)
5. **동화 감상**: 생성된 동화를 읽고 음성을 듣습니다.
6. **보관함 (`/save-story`, `/get-stories`)**: 생성된 동화를 저장하고 나중에 다시 봅니다.
7. **목소리 학습 (`/save-reference-audio`)**: 부모의 목소리를 녹음하여 동화 낭독에 사용합니다.

---

## 검증 계획

1. **실패 시나리오 테스트**: 백엔드에서 강제로 에러를 발생시키고, 프론트엔드가 "Unexpected token <" 대신 에러 창을 띄우는지 확인합니다.
2. **성공 시나리오 테스트**: 정상적으로 동화가 생성되고 TTS가 재생되는지 확인합니다.
