import os
import cv2
import torch
import types
import numpy as np
import datetime
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageSegmentation,  # BiRefNet 로드용
)

# ---------------------------------------------------------
# ※ 삭제된 레거시 항목
# - target_keywords 리스트 (YOLO-World 텍스트 탐지용)
# - ultralytics YOLOWorld, SAM 임포트 및 로드 로직
# - process_object_odyssey() 의 target_keywords 파라미터
# - YOLO 탐지 / SAM 분할 처리 로직 전체
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1. 모델 로드 (FastAPI Startup / Streamlit Cache용)
# ---------------------------------------------------------
def load_all_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ── Florence-2 ──────────────────────────────────────────
    # transformers 5.x 호환: eager 어텐션 사용, dtype 파라미터명 변경
    f_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    f_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    # Florence-2 런타임 패치: past_key_values 내부 원소가 None일 수 있는 경우 처리
    # generate() -> language_model.generate() -> language_model.prepare_inputs_for_generation() 체인이므로
    # 패치를 f_model.language_model에 적용해야 함
    _lang_model = f_model.language_model

    def patched_prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            past_length = 0
            try:
                # transformers 4.50+의 Cache 객체인 경우
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                # 기존 튜플/리스트 구조인 경우
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

    # language_model 인스턴스에 메서드 바인딩
    _lang_model.prepare_inputs_for_generation = types.MethodType(
        patched_prepare_inputs_for_generation, _lang_model
    )

    # ── BiRefNet (배경 제거 / 누끼 추출) ────────────────────
    biref_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    ).to(device)
    biref_model.eval()

    return f_model, f_processor, biref_model, device, torch_dtype


# 모델 초기화 (전역)
f_model, f_processor, biref_model, device, torch_dtype = load_all_models()

# ---------------------------------------------------------
# 2. BiRefNet 내부 추론 헬퍼
# ---------------------------------------------------------
def _run_birefnet(pil_img: Image.Image, model, device) -> np.ndarray:
    """
    PIL RGB 이미지를 받아 BiRefNet으로 전경 마스크(0~1 float32 numpy)를 반환한다.
    마스크는 원본 이미지와 동일한 크기로 리사이즈된다.
    """
    from torchvision import transforms

    orig_w, orig_h = pil_img.size

    # BiRefNet 권장 입력 크기: 1024×1024
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)

    # BiRefNet 출력: 리스트의 마지막 원소가 최종 예측
    pred = preds[-1].squeeze().cpu().numpy()  # shape: (H, W)

    # sigmoid 정규화 후 원본 크기로 리사이즈
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return pred_resized  # float32, 값 범위 0.0 ~ 1.0


# ---------------------------------------------------------
# 3. 핵심 처리 함수 (FastAPI / Streamlit 공용)
# ---------------------------------------------------------
def process_object_odyssey(input_image_array: np.ndarray) -> dict:
    """
    input_image_array : OpenCV 형태(BGR) 또는 NumPy 배열 이미지

    반환값
    ------
    성공 시:
        {
            "status"      : "success",
            "description" : str,        # Florence-2 상세 묘사 텍스트
            "output_path" : str,        # 저장된 누끼 PNG 상대 경로
            "rgba_image"  : np.ndarray  # Streamlit 등에서 바로 표시용 RGBA 배열
        }
    실패 시:
        {
            "status"  : "error",
            "message" : str
        }

    ※ 삭제된 반환 필드
    - "target"  : YOLO-World 기반 키워드 탐지 결과 (레거시 제거로 불필요)
    - "message" : 성공 시 항상 "처리 완료" 고정 문자열이었으므로 제거
    """
    h, w = input_image_array.shape[:2]
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # BGR → RGB PIL 변환 (이후 모든 PIL 처리는 RGB 기준)
    img_rgb = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # ── [Step 1] BiRefNet 누끼 마스크 추출 ──────────────────
    try:
        mask_float = _run_birefnet(pil_img, biref_model, device)  # float32, 0.0~1.0
    except Exception as e:
        return {"status": "error", "message": f"BiRefNet 마스크 추출 실패: {e}"}

    # 이진 마스크 (uint8, 0 또는 255)
    mask_uint8 = (mask_float * 255).clip(0, 255).astype(np.uint8)

    # ── [Step 2] 파일 저장 (배경 투명 RGBA PNG) ─────────────
    img_rgba_pil = Image.fromarray(img_rgb).convert("RGBA")
    # 알파 채널을 마스크로 교체
    r, g, b, _ = img_rgba_pil.split()
    alpha_ch = Image.fromarray(mask_uint8, mode="L")
    img_rgba_pil = Image.merge("RGBA", (r, g, b, alpha_ch))

    # 결과를 numpy로도 보관 (Streamlit 표시용)
    img_rgba = np.array(img_rgba_pil)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    nukki_dir = os.path.join(base_dir, "nukki")
    os.makedirs(nukki_dir, exist_ok=True)

    filename = f"object_{now}.png"
    abs_output_path = os.path.join(nukki_dir, filename)
    img_rgba_pil.save(abs_output_path)
    output_path = f"nukki/{filename}"

    # ── [Step 3] Florence-2 분석용 흰색 배경 이미지 준비 ────
    # 마스크: 객체 영역은 그대로, 배경만 흰색(255,255,255)으로 채움
    # → Florence-2가 배경 노이즈 없이 객체에만 집중하도록 유도
    mask_3ch = np.stack([mask_uint8] * 3, axis=-1).astype(np.float32) / 255.0
    white_bg = np.ones((h, w, 3), dtype=np.float32)  # 흰색 배경
    img_rgb_f32 = img_rgb.astype(np.float32) / 255.0

    # 합성: 객체 픽셀 * mask + 흰색 배경 * (1 - mask)
    composited = (img_rgb_f32 * mask_3ch + white_bg * (1 - mask_3ch))
    composited_uint8 = (composited * 255).clip(0, 255).astype(np.uint8)
    pil_white_bg = Image.fromarray(composited_uint8)

    # ── [Step 4] Florence-2 DETAILED_CAPTION 실행 ───────────
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
        use_cache=False,  # transformers 버전 호환성 문제 방지 (past_key_values 내 None 에러)
    )
    generated_text = f_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = f_processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(w, h)
    )
    florence_text = parsed_answer[task_prompt]
    print(f"[Florence-2 DETAILED_CAPTION] {florence_text}")

    # ── 결과 반환 ────────────────────────────────────────────
    return {
        "status"      : "success",
        "description" : florence_text,   # Florence-2 상세 묘사
        "output_path" : output_path,     # 저장된 누끼 파일 상대 경로
        "rgba_image"  : img_rgba,        # RGBA numpy 배열 (Streamlit 표시용)
    }
