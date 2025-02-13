import pydub
import streamlit as st

audio_value = st.audio_input("Record a voice message")

if audio_value:
    #st.audio(audio_value)
    audio_file = pydub.AudioSegment.from_wav(audio_value)
    audio_file.export("audio.wav", format="wav")
