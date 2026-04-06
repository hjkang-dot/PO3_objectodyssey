import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse # 파일 전송을 위해 추가
from pydantic import BaseModel
from typing import List
import torch
import soundfile as sf
import os
import uuid # 유니크한 파일명을 위해 추가
from qwen_tts import Qwen3TTSModel 
from IPython.display import Audio, display
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 요청 데이터 규격 정의 --------------------------
# voice clone 모델 입력 폼
class AudioRequest(BaseModel):
    text: str   #생성할 동화 문구
    voice_id: str = 'default_women' #복제할 오디오 파일명

# voice conversion 모델 입력 폼
class VCContent(BaseModel):
    text: str   #생성할 동화 문구
    instruct : str   #감정 시지문

class VCRequest(BaseModel):
    voice_id : str = 'default_women' #복제할 오디오 파일명
    data : List[VCContent]

# 2. 모델 로드 및 설정 ------------------------------
print("--- Loading Qwen-TTS Model ---")

#qwen3-tts-base 모델
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

#qwen3-tts-voiceDesign 모델 
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",  # VoiceDesign 모델
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

print(f"--- Model Loaded on {device} ---")

# 3. 함수 설정 -------------------------------------
#볼륨 정규화
def normalize_volume(audio, target_db=-20.0):
    """RMS 기준으로 목표 dB에 맞게 볼륨 정규화"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    # 클리핑 방지 (-1 ~ 1 범위 초과 시 제한)
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


@app.get("/")
async def hello():
    return {"message": "서버 동작 중"}


#부모 목소리 복제 TTS
@app.post("/generate_audio")
async def generate_audio(request: AudioRequest):
    ref_path = f"./audios/{request.voice_id}.wav"

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="해당 목소리 샘플을 찾을 수 없습니다.")

    try:
   
        with torch.no_grad(): 
            wavs, sr = model.generate_voice_clone(
                text=request.text,      
                language="Korean",
                ref_audio=ref_path,
                x_vector_only_mode=True

            )
        
        output_filename = f"output_{uuid.uuid4()}.wav"
        output_path = os.path.join("./audios", output_filename)
        sf.write(output_path, wavs[0], sr)

        # 경로만 보낼 경우
        # return {"status": "success", "output_path": output_path}
        
        # 파일을 직접 다운로드/재생하게 할 경우 (추천)
        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#감정 표현 TTS
@app.post("/generate_audio_v2")
async def generate_audio(request: VCRequest):
    ref_path = f"./audios/{request.voice_id}.wav"

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="해당 목소리 샘플을 찾을 수 없습니다.")

    try:
        fairy_tale_script = request.data
        print(fairy_tale_script)
        all_audio_segments = []
        final_sr = 24000 # 기본 샘플 레이트 예비 설정

        PAUSE_DURATION = 0.8  # 문장 사이 무음 길이 (초) — 취향껏 조절
        TARGET_DB = -20.0       #목표 볼륨 (dB)

        results = {}
        for i, line in enumerate(fairy_tale_script):
            results[i] = model.generate_voice_design(
                text=line.text,
                language="korean",
                instruct=line.instruct
            )

        # 합치기
        final_sr = results[0][1]
        all_audio_segments = []

        for i in range(len(results)):
            wavs, sr = results[i]
            audio = wavs[0]
            

            normalized = normalize_volume(audio, target_db=TARGET_DB)
            all_audio_segments.append(normalized)

            if i < len(results) - 1:
                silence = np.zeros(int(final_sr * PAUSE_DURATION))
                all_audio_segments.append(silence)
        combined = np.concatenate(all_audio_segments)
        output_filename = f"output_{uuid.uuid4()}.wav"
        output_path = os.path.join("./audios", output_filename)
        sf.write(output_path, combined,final_sr)

        # 경로만 보낼 경우
        # return {"status": "success", "output_path": output_path}
        
        # 파일을 직접 다운로드/재생하게 할 경우 (추천)
        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    os.makedirs("./audios", exist_ok=True)
    uvicorn.run(app, host="localhost", port=8001) 


#STT 기능 
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    # 1. 파일 확장자 검증 (선택 사항)
    allowed_extensions = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
    extension = file.filename.split(".")[-1].lower()
    
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {extension}")

    try:
        # 2. 업로드된 파일을 메모리에서 읽어 OpenAI API로 전달
        # file.file은 바이너리 스트림이므로, name을 포함한 튜플 형태로 전달해야 API가 인식합니다.
        result = client.audio.transcriptions.create(
            model="whisper-1", # gpt-4o-mini는 전사 모델명이 whisper-1으로 통용됩니다.
            file=(file.filename, await file.read(), file.content_type),
            language="ko"
        )
        
        # 3. 결과 텍스트 반환
        return {"text": result.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))