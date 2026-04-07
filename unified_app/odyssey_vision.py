import re
import cv2
import torch
import numpy as np
import datetime
from PIL import Image
from ultralytics import YOLOWorld, SAM
from transformers import AutoProcessor, AutoModelForCausalLM

target_keywords = [
    # 1. 피규어, 동물, 인형 및 역할놀이 소품 (범용 무기 및 판타지 아이템 포함)
    "figure", "figurine", "miniature", "character figure", "anime figure", # 피규어 기본형 추가
    "teddy bear", "bear", "plush toy", "plushie", "stuffed animal", "action figure",
    "toy robot", "robot", "toy dinosaur", "dinosaur", "toy dragon", "dragon",
    "toy monster", "monster", "barbie doll", "baby doll", "fairy doll", "doll",
    "wizard figure", "superhero toy", "alien toy", "astronaut toy", "lego set",
    "lego", "building block", "block", "puzzle piece", "puzzle", "toy",
    "toy sword", "toy gun", "water gun", "foam sword", "toy weapon", "toy blaster",
    "sword", "gun", "pistol", "shield", "bow", "arrow", "wand", "magic wand", 
    "staff", "crown", "tiara", "armor", "cape", "cloak",

    # 2. 오감 놀이 및 보드게임 (신규 추가)
    "slime", "play dough", "dough", "clay", "kinetic sand", "sand bucket", "toy shovel",
    "board game", "playing card", "card", "dice", "domino", "rubiks cube", "cube",
    "rubber duck", "bath toy", "water balloon",

    # 3. 직업/역할놀이 세트 및 장난감 악기 (신규 추가)
    "doctor kit", "toy stethoscope", "syringe toy", "toy kitchen", "cash register", 
    "toy phone", "toy camera", "toy microphone",
    "toy piano", "keyboard toy", "toy guitar", "ukulele", "xylophone", "tambourine", 
    "castanets", "toy drum", "recorder",

    # 4. 식기 및 조리 도구
    "thermos flask", "flask", "water bottle", "bottle", "drinking tumbler",
    "tumbler", "coffee mug", "mug", "tea cup", "cup", "drinking glass", "glass",
    "glass pitcher", "pitcher", "water jug", "jug", "tea pot", "electric kettle",
    "kettle", "serving tray", "tray", "dinner plate", "plate", "soup bowl", "bowl",
    "silver spoon", "spoon", "metal fork", "fork", "table knife", "wooden chopsticks",
    "chopsticks", "lunch box", "box", "picnic basket", "basket",

    # 5. 탈것 및 이동수단 (RC카 등 확장)
    "toy car", "car", "toy truck", "truck", "toy train", "train", "toy bus", "bus",
    "fire engine toy", "fire engine", "police car toy", "police car", "ambulance toy",
    "ambulance", "toy airplane", "airplane", "plane", "toy helicopter", "helicopter",
    "toy rocket", "rocket", "toy spaceship", "spaceship", "toy boat", "boat",
    "toy sailboat", "sailboat", "toy submarine", "submarine", "kick scooter",
    "scooter", "skateboard", "tricycle", "bicycle", "bike", "rc car", "remote control car",

    # 6. 공 및 스포츠 용품
    "spinning top", "top", "soccer ball", "basketball", "baseball", "tennis ball",
    "golf ball", "beach ball", "ball", "rubber balloon", "balloon", "flying kite",
    "kite", "yo-yo toy", "yoyo", "marble stone", "marble", "frisbee disc", "frisbee",
    "hula hoop", "jumping rope", "bubble wand", "safety helmet", "helmet",

    # 7. 학용품 및 문구 (미술 소품 확장)
    "pencil case", "sketch book", "note book", "book", "story book", "picture book",
    "writing pen", "pen", "color pencil", "pencil", "crayon stick", "crayon",
    "felt-tip marker", "marker", "oil pastel", "paint brush", "brush", "color palette",
    "palette", "painting canvas", "canvas", "rubber eraser", "eraser", "pencil sharpener",
    "plastic ruler", "ruler", "glue stick", "glue", "scotch tape", "tape",
    "safety scissors", "paper sticker", "sticker", "origami", "color paper", "beads",

    # 8. 가방, 모자 및 액세서리
    "back pack", "backpack", "shoulder bag", "tote bag", "bag", "coin purse", "purse",
    "leather wallet", "wallet", "wrist watch", "sun glasses", "eye glasses", "glasses",
    "winter hat", "baseball cap", "hat", "cap", "sun visor", "rain umbrella", "umbrella",
    "key chain", "house key", "key", "jewelry ring", "ring", "necklace", "bracelet",
    "earring", "hair clip", "hair band", "hair brush",

    # 9. 신발 및 의류 액세서리
    "sneakers", "sandals", "shoes", "winter boots", "boots", "rain boots", "cotton socks",
    "socks", "winter gloves", "gloves", "woolen scarf", "scarf", "neck tie", "tie",
    "hand mirror", "mirror", "pocket comb", "comb", "hand fan", "fan",

    # 10. 자연물 및 식품
    "red apple", "apple", "yellow banana", "banana", "orange fruit", "orange",
    "purple grapes", "grape", "strawberry", "cherry fruit", "cherry", "watermelon slice",
    "watermelon", "sweet corn", "corn", "orange carrot", "carrot", "red tomato", "tomato",
    "brown potato", "potato", "sliced bread", "bread", "chocolate cookie", "cookie",
    "cream cake", "cake", "fruit candy", "candy", "sweet lollipop", "lollipop",
    "ice cream cone", "ice cream", "small flower", "flower", "green leaf", "leaf",
    "round stone", "stone", "tree stick", "stick", "pine cone", "sea shell", "shell"
]

# ---------------------------------------------------------
# 1. 모델 영역 로드 (FastAPI Startup/Streamlit Cache용)
# ---------------------------------------------------------
def load_all_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Florence-2 (transformers 5.x 호환: eager 어텐션 사용, dtype 파라미터명 변경)
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
    import types
    _lang_model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, _lang_model)

    # YOLO-World
    y_model = YOLOWorld("models/yolov8m-worldv2.pt")

    # SAM 2.1
    s_model = SAM("models/sam2.1_b.pt")

    return f_model, f_processor, y_model, s_model, device, torch_dtype

# 모델 초기화 (전역)
f_model, f_processor, y_model, s_model, device, torch_dtype = load_all_models()

# ---------------------------------------------------------
# 2. 핵심 처리 함수 (FastAPI/Streamlit 공용)
# ---------------------------------------------------------
def process_object_odyssey(input_image_array, target_keywords):
    """
    input_image_array: OpenCV형태(BGR) 또는 NumPy 배열 이미지
    """
    h, w, _ = input_image_array.shape
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # [1] Florence-2 캡션
    # OpenCV(BGR) -> PIL(RGB) 변환 후 사용
    img_rgb = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    inputs = f_processor(text="<CAPTION>", images=pil_img, return_tensors="pt").to(device, torch_dtype)
    generated_ids = f_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
        use_cache=False,  # transformers 버전 호환성 문제 방지 (past_key_values 내 None 에러)
    )
    generated_text = f_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = f_processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(w, h))
    florence_text = parsed_answer["<CAPTION>"].lower()
    print(florence_text)

    # [2] 타깃 키워드 정렬 (최적화)
    target_keywords.sort(key=len, reverse=True)

    # [4] 타깃 키워드 추출
    detected_target = None
    for t_kw in target_keywords:
        if re.search(rf'\b{re.escape(t_kw)}(?:s|es)?\b', florence_text):
            detected_target = f"{florence_text},{t_kw}"
            break

    if not detected_target:
        return {"status": "error", "message": "대상을 찾지 못했습니다.", "target": None}

    # [5] YOLO-World 탐지 (중복 로직 통합 및 새 토크나이저 방식)
    print(detected_target)
    y_model.to('cpu')
    y_model.set_classes([detected_target])
    y_model.to('cuda')
    
    # 텍스트 임베딩 dtype를 모델에서 가져와 맞춤(오류 방지)
    m_dtype = next(y_model.model.parameters()).dtype
    y_model.model.txt_feats = y_model.model.txt_feats.to(device='cuda', dtype=m_dtype)

    yolo_results = y_model(input_image_array, conf=0.1)

    if len(yolo_results[0].boxes) == 0:
        return {"status": "error", "message": f"'{detected_target}' 위치 탐지 실패", "target": detected_target}

    # [6] SAM 2.1 마스크 추출
    box_coords = yolo_results[0].boxes.xyxy[0].cpu().numpy().tolist()
    sam_results = s_model.predict(input_image_array, bboxes=[box_coords], device='cuda')

    if sam_results[0].masks is not None:
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        img_rgba = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = (mask_resized * 255).astype(np.uint8)
        
        # 파일 저장 및 결과 반환
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        nukki_dir = os.path.join(base_dir, "nukki")
        os.makedirs(nukki_dir, exist_ok=True)
        
        filename = f"{detected_target}_{now}.png"
        abs_output_path = os.path.join(nukki_dir, filename)
        cv2.imwrite(abs_output_path, img_rgba)
        
        output_path = f"nukki/{filename}"
        
        return {
            "status": "success", 
            "message": "처리 완료", 
            "target": detected_target, 
            "output_path": output_path,
            "rgba_image": img_rgba # Streamlit에서 바로 보여주기 위함
        }
    
    return {"status": "error", "message": "SAM 분할 실패", "target": detected_target}
