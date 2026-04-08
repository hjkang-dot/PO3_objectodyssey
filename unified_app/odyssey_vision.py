import os
import cv2
import torch
import types
import numpy as np
import datetime
import gc  # 가비지 컬렉션용 추가
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageSegmentation,
)

# ---------------------------------------------------------
# 1. 모델 로드 (초기에는 CPU에 로드 유지)
# ---------------------------------------------------------
def load_all_models():
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"--- Loading Vision Models to CPU (Offload Mode) ---")

    # ── Florence-2 ──────────────────────────────────────────
    f_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True # 네트워크 체크 스킵
    ).to("cpu") 
    f_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", 
        trust_remote_code=True,
        local_files_only=True
    )

    _lang_model = f_model.language_model

    def patched_prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = 0
            try:
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                    first_layer = past_key_values[0]
                    if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0 and first_layer[0] is not None:
                        past_length = first_layer[0].shape[2]
            except (AttributeError, TypeError, IndexError):
                past_length = 0

            if past_length > 0:
                if decoder_input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    remove_prefix_length = decoder_input_ids.shape[1] - 1
                decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,
            "encoder_outputs": kwargs.get("encoder_outputs"),
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "decoder_attention_mask": kwargs.get("decoder_attention_mask"),
            "use_cache": kwargs.get("use_cache"),
        }

    _lang_model.prepare_inputs_for_generation = types.MethodType(
        patched_prepare_inputs_for_generation, _lang_model
    )

    # ── BiRefNet ────────────────────
    biref_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True,
        local_files_only=True # 네트워크 체크 스킵
    ).to("cpu") 
    biref_model.eval()

    return f_model, f_processor, biref_model, target_device, torch_dtype


# 전역 설정
f_model, f_processor, biref_model, device, torch_dtype = load_all_models()

def log_vram(tag=""):
    """VRAM 상태 로그 출력용"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[Vision VRAM {tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

# ---------------------------------------------------------
# 2. BiRefNet 내부 추론 헬퍼
# ---------------------------------------------------------
def _run_birefnet(pil_img: Image.Image, model, device) -> np.ndarray:
    from torchvision import transforms
    orig_w, orig_h = pil_img.size
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)

    pred = preds[-1].squeeze().cpu().numpy()
    pred = 1 / (1 + np.exp(-pred))
    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return pred_resized


# ---------------------------------------------------------
# 3. 핵심 처리 함수 (동적 오프로딩 적용)
# ---------------------------------------------------------
def process_object_odyssey(input_image_array: np.ndarray) -> dict:
    global biref_model, f_model
    h, w = input_image_array.shape[:2]
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    img_rgb = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ── [Step 1] BiRefNet 누끼 마스크 추출 (VRAM 로드/언로드) ──
    try:
        log_vram("Before BiRefNet")
        biref_model.to(device) # GPU로 이동
        mask_float = _run_birefnet(pil_img, biref_model, device)
        biref_model.to("cpu")  # 전역 모델을 다시 CPU로 오프로드
        torch.cuda.empty_cache()
        gc.collect()
        log_vram("After BiRefNet")
    except Exception as e:
        biref_model.to("cpu")
        return {"status": "error", "message": f"BiRefNet 마스크 추출 실패: {e}"}

    mask_uint8 = (mask_float * 255).clip(0, 255).astype(np.uint8)

    # ── [Step 2] 파일 저장 ──
    img_rgba_pil = Image.fromarray(img_rgb).convert("RGBA")
    r, g, b, _ = img_rgba_pil.split()
    alpha_ch = Image.fromarray(mask_uint8, mode="L")
    img_rgba_pil = Image.merge("RGBA", (r, g, b, alpha_ch))
    img_rgba = np.array(img_rgba_pil)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    nukki_dir = os.path.join(base_dir, "nukki")
    os.makedirs(nukki_dir, exist_ok=True)
    filename = f"object_{now}.png"
    abs_output_path = os.path.join(nukki_dir, filename)
    img_rgba_pil.save(abs_output_path)
    output_path = f"nukki/{filename}"

    # ── [Step 3] Florence-2 배경 합성 ──
    mask_3ch = np.stack([mask_uint8] * 3, axis=-1).astype(np.float32) / 255.0
    white_bg = np.ones((h, w, 3), dtype=np.float32)
    img_rgb_f32 = img_rgb.astype(np.float32) / 255.0
    composited = (img_rgb_f32 * mask_3ch + white_bg * (1 - mask_3ch))
    composited_uint8 = (composited * 255).clip(0, 255).astype(np.uint8)
    pil_white_bg = Image.fromarray(composited_uint8)

    # ── [Step 4] Florence-2 분석 (VRAM 로드/언로드) ───────────
    try:
        log_vram("Before Florence-2")
        f_model.to(device) # GPU로 이동
        task_prompt = "<DETAILED_CAPTION>"
        inputs = f_processor(
            text=task_prompt,
            images=pil_white_bg,
            return_tensors="pt"
        ).to(device, torch_dtype)

        generated_ids = f_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            use_cache=False,
        )
        generated_text = f_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = f_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(w, h)
        )
        florence_text = parsed_answer[task_prompt]
        
        f_model.to("cpu") # 다시 CPU로 오프로드
        torch.cuda.empty_cache()
        gc.collect()
        log_vram("After Florence-2")
        print(f"[Florence-2 DETAILED_CAPTION] {florence_text}")
    except Exception as e:
        f_model.to("cpu")
        return {"status": "error", "message": f"Florence-2 분석 실패: {e}"}

    return {
        "status"      : "success",
        "description" : florence_text,
        "output_path" : output_path,
        "rgba_image"  : img_rgba,
    }
