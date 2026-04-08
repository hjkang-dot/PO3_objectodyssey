from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import torch
import soundfile as sf
import os
import uuid
from qwen_tts import Qwen3TTSModel
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import gc
import traceback 

load_dotenv()

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 요청 데이터 규격 정의 --------------------------
class AudioRequest(BaseModel):
    text: str
    voice_id: str = 'default_women'

class VCContent(BaseModel):
    text: str
    instruct: str

class VCRequest(BaseModel):
    voice_id: str = 'default_women'
    data: List[VCContent]

# 2. 모델 로드 및 설정 ------------------------------
print("--- Loading Qwen-TTS CustomVoice Model (상시 로드) ---")

# ✅ 메인 모델 (1.7B VoiceDesign) — 상시 로드
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# ✅ model_base — 온디맨드 (평소엔 None)
model_base = None

print(f"--- Model Loaded on {device} ---")

# 3. 함수 설정 -------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
AUDIO_BASE_PATH = ROOT_DIR / "static" / "outputs" / "audios"
os.makedirs(AUDIO_BASE_PATH, exist_ok=True)


def log_gpu_memory(tag=""):
    """VRAM 사용량 로그"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        print(f"[GPU {tag}] allocated: {allocated:.1f}MB / reserved: {reserved:.1f}MB")


def normalize_volume(audio, target_db=-20.0):
    """RMS 기준으로 목표 dB에 맞게 볼륨 정규화"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


def load_model_base():
    """model_base 온디맨드 로드 — model을 먼저 CPU로 오프로드"""
    global model_base, model

    # ✅ 상시 모델을 CPU로 내려서 VRAM 확보
    if model is not None:
        print("--- model을 CPU로 오프로드 ---")
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("model CPU 오프로드 후")

    if model_base is None:
        print("--- Loading model_base (0.6B Base) ---")
        model_base = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        log_gpu_memory("model_base 로드 후")


def unload_model_base():
    """model_base 해제 후 model을 다시 GPU로 복귀"""
    global model_base, model

    if model_base is not None:
        print("--- model_base 해제 ---")
        del model_base
        model_base = None
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("model_base 해제 후")

    # ✅ 상시 모델 다시 GPU로 복귀
    if model is not None:
        print("--- model을 GPU로 복귀 ---")
        model.to(device)
        log_gpu_memory("model GPU 복귀 후")


# 4. 엔드포인트 -------------------------------------

# 부모 목소리 복제 TTS (온디맨드)
@router.post("/generate_audio")
async def generate_audio(request: AudioRequest):
    ref_path = AUDIO_BASE_PATH / f"{request.voice_id}.wav"

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="해당 목소리 샘플을 찾을 수 없습니다.")

    wavs = None
    audio_data = None

    try:
        log_gpu_memory("generate_audio 요청 시작")
        load_model_base()  # ✅ model_base 로드, model은 CPU로

        with torch.no_grad():
            wavs, sr = model_base.generate_voice_clone(
                text=request.text,
                language="Korean",
                ref_audio=str(ref_path),
                x_vector_only_mode=True
            )

        log_gpu_memory("generate_audio 생성 직후")

        raw = wavs[0]
        if hasattr(raw, 'cpu'):
            audio_data = raw.cpu().float().numpy()
        else:
            audio_data = np.array(raw, dtype=np.float32)

        del wavs
        wavs = None
        torch.cuda.empty_cache()

        output_filename = f"output_{uuid.uuid4()}.wav"
        output_path = AUDIO_BASE_PATH / output_filename
        sf.write(str(output_path), audio_data, sr)

        return {"status": "success", "url": f"/static/outputs/audios/{output_filename}"}

    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for var in [wavs, audio_data]:
            try:
                del var
            except:
                pass
        unload_model_base()  # ✅ model_base 해제, model GPU 복귀
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("generate_audio 완료")


# 감정 표현 TTS (상시 로드 모델 사용)
@router.post("/generate_audio_v2")
async def generate_audio_v2(request: VCRequest):
    ref_path = AUDIO_BASE_PATH / f"{request.voice_id}.wav"

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="해당 목소리 샘플을 찾을 수 없습니다.")

    # finally 블록에서 접근할 수 있도록 변수 초기화
    all_audio_segments = []
    combined = None

    try:
        fairy_tale_script = request.data
        print(fairy_tale_script)

        final_sr = 24000 # 기본 샘플 레이트 예비 설정
        PAUSE_DURATION = 0.8  # 문장 사이 무음 길이 (초) — 취향껏 조절
        TARGET_DB = -20.0       #목표 볼륨 (dB)

        ref_audio_path = str(ref_path)

        # 1. 루프 최적화: results 딕셔너리를 쓰지 않고 즉시 처리하여 GPU 부담 감소
        for i, line in enumerate(fairy_tale_script):
            with torch.no_grad():
                # 음성 생성
                wavs, sr = model.generate_custom_voice(
                    text=line.text,
                    language="korean",
                    speaker=['Sohee'],
                    instruct=line.instruct
                )
            final_sr = sr

            # [중요] GPU 텐서를 즉시 CPU Numpy 배열로 변환하여 GPU 메모리 해제
            audio_cpu = wavs[0]

            # 메모리 절약을 위해 원본 wavs 삭제
            del wavs

            normalized = normalize_volume(audio_cpu, target_db=TARGET_DB)
            all_audio_segments.append(normalized)

            if i < len(fairy_tale_script) - 1:
                silence = np.zeros(int(final_sr * PAUSE_DURATION))
                all_audio_segments.append(silence)
        
        # 매 문장 생성 후 캐시를 비워주면 더 안정적입니다 (속도는 약간 저하 가능)
        #torch.cuda.empty_cache()

        # 2. 오디오 합치기
        if all_audio_segments:
            combined = np.concatenate(all_audio_segments)
            
            output_filename = f"output_{uuid.uuid4()}.wav"
            output_path = AUDIO_BASE_PATH / output_filename
            sf.write(str(output_path), combined, final_sr)

            return {"status": "success", "url": f"/static/outputs/audios/{output_filename}"}
        else:
            raise ValueError("생성된 오디오 데이터가 없습니다.")

    except Exception as e:
        print(f"Error detail: {traceback.format_exc()}") # 에러 위치 로그 출력
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 3. 최종 메모리 정리
        # 리스트 내의 큰 넘파이 배열들 명시적 삭제
        if 'all_audio_segments' in locals():
            del all_audio_segments
        if 'combined' in locals():
            del combined
            
        gc.collect()           # Python 가비지 컬렉션 강제 실행
        torch.cuda.empty_cache() # GPU 캐시 비우기
        log_gpu_memory("generate_audio_v2 완료 후 정리")


# 목소리 저장 기능
@router.post("/save-reference-audio")
async def save_reference_audio(voice_id: str = Form("default_women"), file: UploadFile = File(...)):
    """프론트엔드에서 녹음한 사용자 목소리를 voice_id 이름으로 저장"""
    try:
        content = await file.read()
        save_path = AUDIO_BASE_PATH / f"{voice_id}.wav"
        with open(save_path, "wb") as f:
            f.write(content)
        return {"status": "success", "message": f"Saved as {voice_id}.wav"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))