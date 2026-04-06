import streamlit as st


st.title("TTS Script")
st.caption("동화 생성이 끝나면 전체 분량의 TTS 대본이 자동으로 표시됩니다. 여기의 tone은 실제 낭독 톤 가이드입니다.")

story_package = st.session_state.get("latest_story_package")

if not story_package:
    st.info("아직 생성된 동화가 없습니다. Story Generation 페이지에서 먼저 동화를 생성해 주세요.")
else:
    st.subheader(story_package.get("title", ""))
    st.write("전체 TTS Script")
    for idx, item in enumerate(story_package.get("tts_script", []), start=1):
        st.write(f"{idx}. [{item.get('tone', '')}] {item.get('line', '')}")

    st.divider()
    st.json(story_package.get("tts_script", []))
