#qwen3-tts 사용하기 위해 별도 가상환경 구축함

#audio_module 개요
- 음성 복제 api


#오디오 백엔드 서버 실행 순서
1. cd audio_module
2. audio_module 가상환경 활성화
3. uv run odyssey_audio.py (포트 8001)

#api 사용법
POST /generate_audio

Request body
{
  "text": "생성할 문구",
  "voice_id": "참조파일명"
}

Response 
오디오 파일