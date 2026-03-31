from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from odyssey_vision import process_object_odyssey, target_keywords, blacklist_keywords

app = FastAPI(title="Object Odyssey Vision API")

@app.post("/extract")
async def extract_object(file: UploadFile = File(...)):
    # 1. 업로드된 파일을 바이트로 읽기
    contents = await file.read()
    
    # 2. 바이트 데이터를 OpenCV 이미지(NumPy 배열)로 변환
    nparray = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

    # 3. 비전 파이프라인 실행
    result = process_object_odyssey(img, target_keywords, blacklist_keywords)
    
    if result["status"] == "error":
        return {"status": "fail", "message": result["message"]}
    
    return {
        "status": "success",
        "target": result["target"],
        "output_path": result["output_path"]
    }

# 실행 방법: uvicorn main:app --reload