import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

# 1. 접속 정보 설정 (최상단)
API_URL = "http://127.0.0.1:8000/extract"

# 2. 통신 전용 함수 (비즈니스 로직 분리)
def request_nukki_api(image_file):
    """
    image_file: st.file_uploader로부터 받은 파일 객체
    """
    try:
        # 파일을 FastAPI가 원하는 형식(Multipart)으로 패킹
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        
        # FastAPI에 POST 요청 (AI 연산 시간을 고려해 timeout은 넉넉히 또는 None)
        response = requests.post(API_URL, files=files, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"서버 오류: {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "message": f"통신 실패: {str(e)}"}

# 3. Streamlit UI 배치
st.title("🎨 Object Odyssey: AI 에셋 생성")

uploaded_file = st.file_uploader("사진을 올려주세요", type=["jpg", "png"])

if uploaded_file:
    # 화면 왼쪽: 원본 표시
    st.image(uploaded_file, caption="업로드된 이미지", width=300)
    
    # 4. 실제 통신 실행 시점 (버튼 클릭 시)
    if st.button("🚀 AI 누끼 따기 실행"):
        with st.spinner("백엔드 GPU 서버에서 처리 중..."):
            
            # API 호출
            result = request_nukki_api(uploaded_file)
            
            # 결과 처리
            if result["status"] == "success":
                st.success(f"탐지 완료: {result['target']}")
                
                # [중요] FastAPI가 로컬 경로를 주면 직접 읽어서 표시
                # 서버가 다른 PC라면 이미지를 직접 리턴받는 로직이 추가로 필요합니다.
                if "output_path" in result:
                    st.image(result["output_path"], caption="결과 이미지")
            else:
                st.error(result["message"])