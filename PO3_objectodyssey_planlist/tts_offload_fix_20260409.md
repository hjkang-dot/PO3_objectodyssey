# [Fix] TTS 모델 오프로딩 오류 수정 (AttributeError) - 2026-04-09

## 1. 개요
사용자가 제공한 로그에서 `[Warning] Failed to offload model: 'Qwen3TTSModel' object has no attribute 'to'` 메시지가 발견되었습니다. 이는 `Qwen3TTSModel` 클래스가 하위 모델을 감싸는 래퍼 클래스이어서 직접적인 `.to()` 메서드를 지원하지 않기 때문입니다. 이를 실제 모델 인스턴스(`.model`)에 접근하도록 수정하여 GPU 가용량을 확보하고 오류를 해결합니다.

## 2. 수정 계획

### 파일: `unified_app/odyssey_audio.py`

- **`load_model_base()` (L100 부근)**:
  - `model.to('cpu')` -> `model.model.to('cpu')`
- **`generate_audio_v2()` (L200 부근)**:
  - `model.to(device)` -> `model.model.to(device)`
- **`generate_audio_v2()` finally 블록 (L245 부근)**:
  - `model.to('cpu')` -> `model.model.to('cpu')`

또한 안전을 위해 `hasattr(model, 'model')` 체크를 추가하여 견고한 코드를 작성합니다.

## 3. 검증 계획
- 동화 생성 프로세스를 실행하여 로그에 더 이상 `AttributeError` 관련 워닝이 뜨지 않는지 확인합니다.
- GPU 메모리 로그(`log_gpu_memory`)를 통해 모델이 정상적으로 CPU로 내려가고 GPU로 올라가는지 모니터링합니다.
