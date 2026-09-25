import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None

def load_model():
    global _model
    if _model is None:
        print("Loading model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Model loaded successfully.")
    return _model


def transcribe_chunk(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path)
    return result["text"]


def transcribe_all(chunks: list) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk}")
        text = transcribe_chunk(chunk)
        full_transcript += text + " "
    print("Transcription complete.")
    return full_transcript