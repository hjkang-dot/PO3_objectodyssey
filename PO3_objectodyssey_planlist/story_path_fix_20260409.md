# [Error Fix] FileNotFoundError in storybook_core.py (Windows Path Resolution) - 2026-04-09

## 개요
Windows 환경에서 프론트엔드가 `/static/outputs/img.png`와 같이 시작하는 경로를 보낼 때, `pathlib.Path`가 이를 드라이브 루트(C:\) 기준의 절대 경로로 오인하여 발생하는 `FileNotFoundError`를 해결합니다.

## 수정 세부 사항

### [MODIFY] [storybook_core.py](file:///c:/project_3rd/PO3_objectodyssey/unified_app/app/storybook_core.py)

`_resolve_reference_path` 함수를 다음과 같이 개선합니다:
- 입력 경로의 맨 앞 `/`를 제거(`lstrip("/")`)합니다.
- 프로젝트 루트(`PROJECT_ROOT`) 기준의 상대 경로로 먼저 확인합니다.
- 진짜 절대 경로인 경우의 예외 처리도 포함합니다.

```python
def _resolve_reference_path(reference_image: str) -> Path:
    # URL 경로처럼 앞에 /가 붙어서 오는 경우를 위해 lstrip("/") 처리
    clean_path = reference_image.lstrip("/")
    candidate = Path(clean_path)
    
    # 1. 프로젝트 루트 기준 상대 경로로 먼저 시도
    resolved = (PROJECT_ROOT / candidate).resolve()
    if resolved.exists():
        return resolved
        
    # 2. 만약 입력값이 진짜 절대 경로(Windows 드라이브 문자 포함 등)라면 그대로 시도
    abs_candidate = Path(reference_image)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return abs_candidate.resolve()
        
    raise FileNotFoundError(f"Story illustration reference image not found: {reference_image}")
```

## 검증 계획

### 자동 테스트
- `generate-book` API를 `/static/`으로 시작하는 이미지 경로와 함께 호출하여 오류 없이 동화가 생성되는지 확인합니다.

### 수동 검증
- 콘솔 로그의 `[DEBUG]` 메시지를 확인하고 프로세스가 정상적으로 끝나는지 체크합니다.
