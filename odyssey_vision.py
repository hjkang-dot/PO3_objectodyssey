import re
import cv2
import torch
import numpy as np
import datetime
from PIL import Image
from ultralytics import YOLOWorld, SAM
from transformers import AutoProcessor, AutoModelForCausalLM

target_keywords = [
    # 1. 인형 및 장난감
    "teddy bear", "bear", "plush toy", "plushie", "stuffed animal", "action figure",
    "toy robot", "robot", "toy dinosaur", "dinosaur", "toy dragon", "dragon",
    "toy monster", "monster", "barbie doll", "baby doll", "fairy doll", "doll",
    "wizard figure", "superhero toy", "alien toy", "astronaut toy", "lego set",
    "lego", "building block", "block", "puzzle piece", "puzzle", "toy",

    # 2. 식기 및 음료 용기
    "thermos flask", "flask", "water bottle", "bottle", "drinking tumbler",
    "tumbler", "coffee mug", "mug", "tea cup", "cup", "drinking glass", "glass",
    "glass pitcher", "pitcher", "water jug", "jug", "tea pot", "electric kettle",
    "kettle", "serving tray", "tray", "dinner plate", "plate", "soup bowl", "bowl",
    "silver spoon", "spoon", "metal fork", "fork", "table knife", "wooden chopsticks",
    "chopsticks", "lunch box", "box", "picnic basket", "basket",

    # 3. 이동수단 장난감
    "toy car", "car", "toy truck", "truck", "toy train", "train", "toy bus", "bus",
    "fire engine toy", "fire engine", "police car toy", "police car", "ambulance toy",
    "ambulance", "toy airplane", "airplane", "plane", "toy helicopter", "helicopter",
    "toy rocket", "rocket", "toy spaceship", "spaceship", "toy boat", "boat",
    "toy sailboat", "sailboat", "toy submarine", "submarine", "kick scooter",
    "scooter", "skateboard", "tricycle", "bicycle", "bike",

    # 4. 놀이 및 스포츠 용품
    "spinning top", "top", "soccer ball", "basketball", "baseball", "tennis ball",
    "golf ball", "beach ball", "ball", "rubber balloon", "balloon", "flying kite",
    "kite", "yo-yo toy", "yoyo", "marble stone", "marble", "frisbee disc", "frisbee",
    "hula hoop", "jumping rope", "bubble wand", "safety helmet", "helmet",

    # 5. 학용품 및 도구
    "pencil case", "sketch book", "note book", "book", "story book", "picture book",
    "writing pen", "pen", "color pencil", "pencil", "crayon stick", "crayon",
    "felt-tip marker", "marker", "oil pastel", "paint brush", "brush", "color palette",
    "palette", "painting canvas", "canvas", "rubber eraser", "eraser", "pencil sharpener",
    "plastic ruler", "ruler", "glue stick", "glue", "scotch tape", "tape",
    "safety scissors", "paper sticker", "sticker",

    # 6. 의류, 가방 및 소품
    "back pack", "backpack", "shoulder bag", "tote bag", "bag", "coin purse", "purse",
    "leather wallet", "wallet", "wrist watch", "sun glasses", "eye glasses", "glasses",
    "winter hat", "baseball cap", "hat", "cap", "sun visor", "rain umbrella", "umbrella",
    "key chain", "house key", "key", "jewelry ring", "ring", "necklace", "bracelet",
    "earring", "hair clip", "hair band", "hair brush",

    # 7. 신발 및 기타 잡화
    "sneakers", "sandals", "shoes", "winter boots", "boots", "rain boots", "cotton socks",
    "socks", "winter gloves", "gloves", "woolen scarf", "scarf", "neck tie", "tie",
    "hand mirror", "mirror", "pocket comb", "comb", "hand fan", "fan",

    # 8. 자연물 및 음식
    "red apple", "apple", "yellow banana", "banana", "orange fruit", "orange",
    "purple grapes", "grape", "strawberry", "cherry fruit", "cherry", "watermelon slice",
    "watermelon", "sweet corn", "corn", "orange carrot", "carrot", "red tomato", "tomato",
    "brown potato", "potato", "sliced bread", "bread", "chocolate cookie", "cookie",
    "cream cake", "cake", "fruit candy", "candy", "sweet lollipop", "lollipop",
    "ice cream cone", "ice cream", "small flower", "flower", "green leaf", "leaf",
    "round stone", "stone", "tree stick", "stick", "pine cone", "sea shell", "shell"
]

blacklist_keywords = [
    # 1. 날카롭거나 위험한 물건
    "knife", "razor", "blade", "cutter", "saw", "hammer", "screwdriver", 
    "drill", "needle", "pin", "sword", "gun", "weapon",
    
    # 2. 화기 및 화학/약품
    "lighter", "matches", "candle", "firecracker", "stove", "poison", 
    "chemical", "medicine", "pill", "drug",
    
    # 3. 고가 전자제품 (동화 시나리오 부적합 및 파손 방지)
    "smartphone", "tablet", "laptop", "monitor", "tv", "camera", 
    "computer", "smartwatch",
    
    # 4. 유해 매체 및 오염물
    "cigarette", "alcohol", "wine", "beer", "vape", "trash", "garbage"
]

# ---------------------------------------------------------
# 1. 모델 전역 로드 (FastAPI Startup/Streamlit Cache용)
# ---------------------------------------------------------
def load_all_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Florence-2
    f_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    f_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

    # YOLO-World
    y_model = YOLOWorld("models/yolov8m-worldv2.pt")

    # SAM 2.1
    s_model = SAM("models/sam2.1_t.pt")

    return f_model, f_processor, y_model, s_model, device, torch_dtype

# 모델 초기화 (전역)
f_model, f_processor, y_model, s_model, device, torch_dtype = load_all_models()

# ---------------------------------------------------------
# 2. 핵심 처리 함수 (FastAPI/Streamlit 연동용)
# ---------------------------------------------------------
def process_object_odyssey(input_image_array, target_keywords, blacklist_keywords):
    """
    input_image_array: OpenCV형태(BGR) 또는 NumPy 배열 이미지
    """
    h, w, _ = input_image_array.shape
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # [1] Florence-2 캡셔닝
    # OpenCV(BGR) -> PIL(RGB) 변환 필수
    img_rgb = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    inputs = f_processor(text="<CAPTION>", images=pil_img, return_tensors="pt").to(device, torch_dtype)
    generated_ids = f_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )
    generated_text = f_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = f_processor.post_process_generation(generated_text, task="<CAPTION>", image_size=(w, h))
    florence_text = parsed_answer["<CAPTION>"].lower()

    # [2] 블랙리스트 및 타겟 키워드 정렬 (최적화)
    target_keywords.sort(key=len, reverse=True)
    blacklist_keywords.sort(key=len, reverse=True)

    # [3] 블랙리스트 검사 (가장 먼저 수행)
    for b_kw in blacklist_keywords:
        if re.search(rf'\b{re.escape(b_kw)}(?:s|es)?\b', florence_text):
            return {"status": "error", "message": f"위험 물건({b_kw}) 감지됨", "target": None}

    # [4] 타겟 키워드 추출
    detected_target = None
    for t_kw in target_keywords:
        if re.search(rf'\b{re.escape(t_kw)}(?:s|es)?\b', florence_text):
            detected_target = t_kw
            break

    if not detected_target:
        return {"status": "error", "message": "타겟을 찾지 못했습니다.", "target": None}

    # [5] YOLO-World 탐지 (중복 로직 통합 및 동기화)
    y_model.to('cpu')
    y_model.set_classes([detected_target])
    y_model.to('cuda')
    
    # 데이터 타입 및 텐서 동기화 (에러 방지 핵심)
    m_dtype = next(y_model.model.parameters()).dtype
    y_model.model.txt_feats = y_model.model.txt_feats.to(device='cuda', dtype=m_dtype)

    yolo_results = y_model(input_image_array, conf=0.1)

    if len(yolo_results[0].boxes) == 0:
        return {"status": "error", "message": f"'{detected_target}' 위치 탐지 실패", "target": detected_target}

    # [6] SAM 2.1 누끼 추출
    box_coords = yolo_results[0].boxes.xyxy[0].cpu().numpy().tolist()
    sam_results = s_model.predict(input_image_array, bboxes=[box_coords], device='cuda')

    if sam_results[0].masks is not None:
        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        img_rgba = cv2.cvtColor(input_image_array, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = (mask_resized * 255).astype(np.uint8)
        
        # 파일 저장 및 결과 반환
        output_path = f"nukki/{detected_target}_{now}.png"
        cv2.imwrite(output_path, img_rgba)
        
        return {
            "status": "success", 
            "message": "처리 완료", 
            "target": detected_target, 
            "output_path": output_path,
            "rgba_image": img_rgba # Streamlit에서 바로 보여주기 위함
        }
    
    return {"status": "error", "message": "SAM 분할 실패", "target": detected_target}