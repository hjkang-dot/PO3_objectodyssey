import streamlit as st
from streamlit_mic_recorder import mic_recorder
from streamlit_lottie import st_lottie
import datetime
import os
import time
import json
import requests

#====================================================
#설정
#====================================================

st.set_page_config(page_title="AI 동화", page_icon="🙂")


#======================================================

st.title("🎨 오브젝트 오디세이: 오디오 파트")
st.subheader("부모의 목소리를 들려주세요!")

# 저장할 폴더 생성 (없으면 만들기)
save_dir = "recordings"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 오디오 입력 섹션
st.write("아래 버튼을 눌러 녹음을 시작하세요.")

# 녹음기 컴포넌트 배치
audio = mic_recorder(
    start_prompt="🔴 녹음 시작",
    stop_prompt="⏹️ 녹음 중지",
    just_once=True,
    use_container_width=True,
    key='recorder'
)
            
if audio:
    # 녹음된 오디오 재생
    st.audio(audio['bytes'])

    # 1. 고유한 파일명 생성 (예: audio_20231027_143005.wav)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{now}.wav"
    file_path = os.path.join(save_dir, file_name)

    # 파일로 저장하고 싶다면?
    with open(file_path, "wb") as f:
        f.write(audio['bytes'])
    st.success("오디오 파일이 성공적으로 저장되었습니다!")