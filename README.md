## Audio Transcription


A Python application that does Speech to text transcription, custom summarisation, and question and answer over meeting and audio calls using large language models.

The application accepts an uploaded audio file and text transcript, or youtube url. The user can choose various summarisation output formats like one sentence, bullet points, short or long summary. The summary is generated as per desired output format and the user can then ask questions about the transcript to obtain further finer details contained in the transcript.

Concepts used in this project:
- Speech to text transcription of an uploaded audio mp3 file or youtube video url link using OpenAI whisper model.
- Natural Language Processing: Content summarisation using two common approaches: Stuff and Map-reduce. Stuff generally gives better summarisation results than map reduce. Stuff is used prefentially if transcript token length is within the language model context window. Map reduce is used when the token count is greater.
- Retriever for the Question and Answer over transcript. The transcript is split into documents and embedded into a Chroma vectorstore for retrieval.
- LangChain as framework and Streamlit for UI


