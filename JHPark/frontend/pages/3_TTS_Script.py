"""TTS 스크립트 전용 페이지."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="TTS Script", page_icon="OO", layout="wide")

st.title("TTS 스크립트")
st.caption("동화 생성이 완료되면 전체 분량의 TTS 대본이 자동으로 표시됩니다. 여기의 tone은 동화 분위기가 아니라 실제 낭독 톤 가이드입니다.")

story_package = st.session_state.get("latest_story_package")

if not story_package:
    st.info("아직 생성된 동화가 없습니다. Story Generation 페이지에서 먼저 동화를 생성해 주세요.")
else:
    st.subheader(story_package.get("title", ""))
    st.write("전체 TTS 스크립트")
    for idx, item in enumerate(story_package.get("tts_script", []), start=1):
        st.write(f"{idx}. [{item.get('tone', '')}] {item.get('line', '')}")

    st.divider()
    st.write("JSON 구조")
    st.json(story_package.get("tts_script", []))
