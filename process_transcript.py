from dotenv import load_dotenv
import os 
import openai
from pytube import YouTube 

load_dotenv()


def download_audio(url: str) -> str:
    """Takes in youtube url as input, downloads as audio in .mp3 format using pytube and returns download file path as str"""
    
    yt = YouTube(url) 
    video = yt.streams.filter(only_audio=True).first() 
    destination = 'audio_files'
    out_file = video.download(output_path=destination) 
    base, ext = os.path.splitext(out_file) 
    new_file = base + '.mp3'
    os.rename(out_file, new_file) 
    
    return os.path.join(destination, new_file)


def upload_audio(audio_file) -> str:
    """Saves the uploaded audio file to the file folders and returns audio file path"""

    filepath = os.path.join('audio_files', audio_file.name)
    with open(filepath, mode='wb') as f:
        f.write(audio_file.getvalue())
    return filepath


def get_transcript_from_audio(audio_file_path: str) -> str:
    """Transcribe audio to text using whisper. Input param is audio file path as str and returns text transcript as str"""

    openai.api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI()
    transcript = client.audio.transcriptions.create(
      model="whisper-1", 
      file=open(audio_file_path, "rb") # r refers to read and b refers to binary
    )

    return transcript.text


def read_transcript_from_text(text_file) -> str:
    """Read and returns text transcript from an uploaded text file"""

    lines = [line.decode() for line in text_file]
    transcript = ' '.join(lines)

    return transcript


if __name__ == "__main__":
    pass
