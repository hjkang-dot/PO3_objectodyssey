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

# 모델 변수 초기화 (지연 로딩 용)
model = None
model_base = None

def get_model():
    """메인 모델(1.7B) 지연 로드"""
    global model
    if model is None:
        print("--- Loading Qwen-TTS CustomVoice Model (1.7B) to CPU ---")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cpu",
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True
        )
    return model

def get_model_base():
    """베이스 모델(0.6B) 지연 로드"""
    global model_base
    if model_base is None:
        print("--- Loading Qwen-TTS Base Model (0.6B) to GPU ---")
        model_base = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True
        )
    return model_base

print(f"--- Audio Service Initialized (Lazy Loading Mode) ---")

# 3. 함수 설정 -------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
AUDIO_BASE_PATH = ROOT_DIR / "static" / "outputs" / "audios"
os.makedirs(AUDIO_BASE_PATH, exist_ok=True)


def log_gpu_memory(tag=""):
    """VRAM 사용량 로그"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        print(f"[Audio GPU {tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")


def normalize_volume(audio, target_db=-20.0, max_gain=5.0, min_rms=0.005):
    """
    RMS 기준으로 목표 dB에 맞게 볼륨 정규화.
    - max_gain: 배경 노이즈가 과도하게 커지는 것을 방지하기 위한 최대 증폭 배수
    - min_rms: 이 값보다 작은 소리는 노이즈로 간주하여 정규화하지 않음
    """
    rms = np.sqrt(np.mean(audio ** 2))
    
    # 너무 작은 소리(노이즈)는 정규화 건너뜀
    if rms < min_rms:
        return audio
        
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    
    # 과도한 증폭 제한 (부스럭거리는 소리 방지)
    actual_gain = min(gain, max_gain)
    
    normalized = audio * actual_gain
    return np.clip(normalized, -1.0, 1.0)



def load_model_base():
    """model_base(0.6B) 온디맨드 로드. GPU 공간 확보 선행."""
    global model_base, model

    curr_model = get_model()
    if curr_model is not None:
        try:
            print("--- Offloading main model to CPU to make room for base model ---")
            if hasattr(curr_model, 'model'):
                curr_model.model.to('cpu')
            else:
                curr_model.to('cpu')
        except Exception as e:
            print(f"[Warning] Failed to offload model: {e}")
    
    gc.collect()
    torch.cuda.empty_cache()

    get_model_base()
    log_gpu_memory("model_base Load")


def unload_model_base():
    """model_base 사용 완료 후 캐시 정리. (주 모델 복귀는 추론 시에만 수행)"""
    global model_base
    if model_base is not None:
        print("--- Unloading model_base ---")
        del model_base
        model_base = None
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("model_base Unload")


# 4. 엔드포인트 -------------------------------------


def _generate_single_audio_internal(text: str, voice_id: str):
    """내부용: 단일 텍스트 TTS 생성 로직"""
    ref_path = AUDIO_BASE_PATH / f"{voice_id}.wav"
    if not os.path.exists(ref_path):
        ref_path = AUDIO_BASE_PATH / "default_parents.wav"
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Voice sample {voice_id} not found.")

    curr_model_base = get_model_base()
    with torch.no_grad():
        wavs, sr = curr_model_base.generate_voice_clone(
            text=text,
            language="Korean",
            ref_audio=str(ref_path),
            x_vector_only_mode=True
        )

    raw = wavs[0]
    audio_data = raw.cpu().float().numpy() if hasattr(raw, 'cpu') else np.array(raw, dtype=np.float32)

    output_filename = f"output_parents_{uuid.uuid4()}.wav"
    output_path = AUDIO_BASE_PATH / output_filename
    sf.write(str(output_path), audio_data, sr)
    return f"/static/outputs/audios/{output_filename}"


def generate_book_audios(paragraphs: List[str], voice_id: str = "default_women") -> List[str]:
    """부모 목소리 일괄 생성 호출"""
    urls = []
    try:
        load_model_base()
        for p in paragraphs:
            url = _generate_single_audio_internal(p, voice_id)
            urls.append(url)
        return urls
    finally:
        unload_model_base()


@router.post("/generate_audio")
def generate_audio(request: AudioRequest):
    try:
        load_model_base()
        url = _generate_single_audio_internal(request.text, request.voice_id)
        return {"status": "success", "url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        unload_model_base()


@router.post("/generate_audio_v2")
def generate_audio_v2(request: VCRequest):
    global model
    ref_path = AUDIO_BASE_PATH / f"{request.voice_id}.wav"
    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="해당 목소리 샘플을 찾을 수 없습니다.")

    all_audio_segments = []
    combined = None

    try:
        # ✅ 추론 전에 다른 모델(model_base)이 있다면 내리고 GPU 공간 확보
        unload_model_base()

        print("--- Moving main model to GPU ---")
        curr_model = get_model()
        try:
            if hasattr(curr_model, 'model'):
                curr_model.model.to(device)
            else:
                curr_model.to(device)
        except Exception as e:
            print(f"[Warning] Failed to move model to GPU: {e}")

        log_gpu_memory("Before generate_audio_v2 Loop")

        fairy_tale_script = request.data
        final_sr = 24000
        PAUSE_DURATION = 0.8  
        TARGET_DB = -20.0       

        for i, line in enumerate(fairy_tale_script):
            with torch.no_grad():
                wavs, sr = curr_model.generate_custom_voice(
                    text="어 " + line.text,
                    language="korean",
                    speaker=['Sohee'],
                    instruct="Speak slowly " + line.instruct
                )
            final_sr = sr
            audio_cpu = wavs[0]
            del wavs
            normalized = normalize_volume(audio_cpu, target_db=TARGET_DB)
            all_audio_segments.append(normalized)

            if i < len(fairy_tale_script) - 1:
                silence = np.zeros(int(final_sr * PAUSE_DURATION))
                all_audio_segments.append(silence)
        
        if all_audio_segments:
            combined = np.concatenate(all_audio_segments)
            output_filename = f"output_{uuid.uuid4()}.wav"
            output_path = AUDIO_BASE_PATH / output_filename
            sf.write(str(output_path), combined, final_sr)
            return {"status": "success", "url": f"/static/outputs/audios/{output_filename}"}
        else:
            raise ValueError("생성된 오디오 데이터가 없습니다.")

    except Exception as e:
        print(f"Error detail: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ✅ 작업 끝난 후 메인 모델 즉시 CPU로 반환
        print("--- Offloading main model to CPU ---")
        curr_model = get_model()
        if hasattr(curr_model, 'model'):
            curr_model.model.to('cpu')
        else:
            curr_model.to('cpu')
        
        if 'all_audio_segments' in locals(): del all_audio_segments
        if 'combined' in locals(): del combined
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("After generate_audio_v2 Cleanup")


@router.get("/list-voices")
async def list_voices():
    try:
        voices = []
        if (AUDIO_BASE_PATH / "default_parents.wav").exists():
            voices.append({"id": "default_parents", "name": "기본 목소리 (부모)"})
        elif (AUDIO_BASE_PATH / "default_women.wav").exists():
            voices.append({"id": "default_women", "name": "기본 목소리 (여성)"})
        
        files = sorted(AUDIO_BASE_PATH.glob("user_*.wav"), key=lambda x: os.path.getmtime(x), reverse=True)
        for file in files:
            voice_name = file.stem.replace("user_", "")
            voices.append({"id": file.stem, "name": f"{voice_name}의 목소리"})
        return {"status": "success", "voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-reference-audio")
def save_reference_audio(voice_id: str = Form(...), file: UploadFile = File(...)):
    try:
        clean_id = voice_id.replace("user_", "")
        final_id = f"user_{clean_id}"
        content = file.file.read()
        save_path = AUDIO_BASE_PATH / f"{final_id}.wav"
        with open(save_path, "wb") as f:
            f.write(content)
        return {"status": "success", "message": f"Saved as {final_id}.wav", "voice_id": final_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """OpenAI Whisper를 사용하여 음성을 텍스트로 변환합니다."""
    try:
        # 파일 확장자 추출 (기본값 webm)
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'webm'
        temp_filename = f"stt_{uuid.uuid4()}.{ext}"
        temp_path = AUDIO_BASE_PATH / temp_filename
        
        # 파일 임시 저장
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        # Whisper API 호출
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {"status": "success", "text": transcript.text}
    except Exception as e:
        print(f"[ERROR] STT 에러 발생: {e}")
        # 임시 파일이 남아있다면 삭제 시도
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-voice/{voice_id}")
async def delete_voice(voice_id: str):
    try:
        if ".." in voice_id or "/" in voice_id or "\\" in voice_id:
            raise HTTPException(status_code=400, detail="Invalid voice ID")
        file_path = AUDIO_BASE_PATH / f"{voice_id}.wav"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Voice file not found")
        if voice_id in ["default_parents", "default_women"]:
            raise HTTPException(status_code=403, detail="Cannot delete default voices")
        os.remove(file_path)
        return {"status": "success", "message": f"{voice_id} 삭제 완료"}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))