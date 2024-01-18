# audio_transcription


This is a personal project that builds a simple Python application that does Speech to text transcription, custom summarisation, and question and answer over meeting and audio calls using large language models.

Concepts used in this project:
- Speech to text transcription of an uploaded audio mp3 file or youtube video url link using OpenAI whisper model.
- Content summarisation using two common approaches: Stuff and Map-reduce. Stuff generally gives better summarisation results than map reduce. Stuff is used prefentially if transcript token length is within the language model context window. Map reduce is used when the token count is greater.
- Retriever for the Question and Answer over transcript. The transcript is split into documents and embedded into a Chroma vectorstore for retrieval.
- LangChain as framework and Streamlit for UI


