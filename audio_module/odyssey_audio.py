import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse # 파일 전송을 위해 추가
from pydantic import BaseModel
import torch
import soundfile as sf
import os
import uuid # 유니크한 파일명을 위해 추가
from qwen_tts import Qwen3TTSModel 

app = FastAPI()

#디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 요청 데이터 규격 정의
class AudioRequest(BaseModel):
    text: str   #생성할 동화 문구
    voice_id: str = 'default_women' #복제할 오디오 파일명

# 2. 모델 로드 및 설정
print("--- Loading Qwen-TTS Model (X-Vector Mode) ---")

#qwen3-tts
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

print(f"--- Model Loaded on {device} ---")

@app.get("/")
async def hello():
    return {"message": "서버 동작 중"}


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


@app.post("/generate_audio_v2")
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
    
if __name__ == "__main__":
    os.makedirs("./audios", exist_ok=True)
    uvicorn.run(app, host="localhost", port=8001) 