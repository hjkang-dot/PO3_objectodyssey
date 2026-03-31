1. uv venv
2. .venv\Scripts\activate
3. uv sync


#이슈 노트
- transformers 버전에 따른 qwen-tts 의존성 충돌
    .방안 1 : 의존성 무시하고 강제 설치 pip install qwen-tts --no-deps => 실패
    .방안 2 : 모델 변경


