
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

client11labs = ElevenLabs(
  api_key="sk_66d9e83921aa83adabe069d35f0fcbc5f645f55d405f01c8", # Defaults to ELEVEN_API_KEY or ELEVENLABS_API_KEY
)

from groq import Groq

# Initialize the Groq client
client = Groq(api_key='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')

def convert_audio_to_transcript(filename):

    # Open the audio file
    with open(filename, "rb") as file:
        # Create a translation of the audio file
        translation = client.audio.translations.create(
            file=(filename, file.read()), # Required audio file
            model="whisper-large-v3", # Required model to use for translation
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            temperature=0.0  # Optional
        )
    # Print the translation text
    return(translation.text)


from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a English tutor and assist new leaners in learning English.

			Have a casual conversation and start introducing yourself being an English tutor.
			
			You can talk different subjects in order to help learner express him/herself in English.

			You are the specialist in English grammar and provide any advice when learner does some mistake.

			Start conversation with a greeting and ask learner´s name.


			- Be sure to be kind of funny and witty!
			- Provide an advise the learner does a mistake but do not overwhelm the learner.
			- Keep all your responses short and simple. Use casual language, phrases like "Umm..", "Well...", and "I mean" are preferred
			- This is a voice conversation, so keep your responses short, like in a real conversation. Don't ramble for too long.
			- Don't make any jokes. Never say "haha" or "good one".
			""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192", api_key="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")

runnable = prompt | llm

from langchain_core.runnables.history import RunnableWithMessageHistory

runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

import pydub
import streamlit as st

audio_value = st.audio_input("Record a voice message")

if audio_value:
    #st.audio(audio_value)
    audio_wav = pydub.AudioSegment.from_wav(audio_value)
    audio_wav.export("audio.wav", format="wav")

    audio_file = "audio.wav"

    question = convert_audio_to_transcript(audio_file)

    answer = runnable_with_history.invoke(
                {"language": "english", "input": question},
                config={"configurable": {"session_id": "1"}},
             ).content

    response = client11labs.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=answer,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)
    # Generating a unique file name for the output MP3 file
    save_file_path = "audio.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")

    st.audio("audio.mp3")

    st.code(answer, wrap_lines=True)
