import os
import shutil

import streamlit as st

from langchain_helper import LangchainHelper
from process_transcript import (
    download_audio,
    upload_audio,
    get_transcript_from_audio,
    read_transcript_from_text,
)


def main():

    if 'qna_memory' not in st.session_state:
        st.session_state.qna_memory = []

    with st.sidebar:
        st.header("Upload")
        st.subheader("Provide youtube url, audio or text file and click 'Process'")
        youtube_url = st.text_input("Enter Youtube url")
        audio_file = st.file_uploader("Upload audio transcript as mp3 file", type=["mp3"])
        text_file = st.file_uploader("Upload text transcript as txt file", type=["txt"])
        process_button = st.button("Process")

        if process_button:

            transcript = None

            with st.spinner("Processing..."):

                if text_file:
                    transcript = read_transcript_from_text(text_file)

                else:
                    if youtube_url:
                        audio_file_path = download_audio(youtube_url)

                    elif audio_file:
                        audio_file_path = upload_audio(audio_file)
                    
                    transcript = get_transcript_from_audio(audio_file_path)

                if transcript:
                    # Remove vectordb folder before initialising lch object
                    if os.path.isdir('vectordb'):
                        shutil.rmtree('vectordb')
                    st.session_state.lch = LangchainHelper(transcript)

        st.header("Summarise")
        summary_choice = st.radio("Choose summary output format and click 'Generate'", ["one sentence", "bullet points", "short", "long"])
        summary_button = st.button("Generate")

    st.header("Summary")
    # Generates summary if user makes selection on summary options on sidebar
    if summary_button and 'lch' in st.session_state:
        with st.spinner("Processing..."):
            st.session_state.summary = st.session_state.lch.summarise(summary_choice)

    if 'summary' in st.session_state:
        st.markdown(st.session_state.summary)

        col1, col2 = st.columns(2) 
        with col2:
            st.header("History")
            for pair in st.session_state.qna_memory:
                st.markdown("Question")
                st.markdown(pair.question)
                st.markdown("Answer")
                st.markdown(pair.answer)

        with col1:
            st.header("Ask a question")
            if prompt := st.text_area("Ask a question on the transcript: ", max_chars=200):
                with st.spinner("Processing..."):
                    response = st.session_state.lch.get_response(prompt)
                st.markdown("Answer: ")
                st.markdown(response)
                st.session_state.qna_memory.append(QnAPair(prompt, response))

   
class QnAPair:

    def __init__(self, question, answer):
        self.question = question
        self.answer = answer


if __name__ == "__main__":
    main()


