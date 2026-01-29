import os
import requests
import ffmpeg
import time
from typing import Literal, Dict, Any, Union
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# Globální proměnná pro uložení načteného modelu (Singleton)
local_whisper_model = None

def get_local_whisper():
    """Načte model do VRAM pouze jednou."""
    global local_whisper_model
    if local_whisper_model is None:
        print("Loading Local Whisper Large-v3 model to GPU... (this may take a moment)")
        # compute_type="float16" pro GPU
        local_whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return local_whisper_model

def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """
    Vytáhne zvukovou stopu z videa a uloží ji jako MP3.
    """
    try:
        # Převedeme na MP3 (libmp3lame), qscale:a 2 je vysoká kvalita VBR
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='libmp3lame', **{'qscale:a': 2}, loglevel="quiet")
            .overwrite_output()
            .run()
        )
        return True
    except ffmpeg.Error as e:
        print(f"Chyba při extrakci audia: {e}")
        return False
    except Exception as e:
        print(f"Obecná chyba extrakce: {e}")
        return False

def transcribe_with_timestamps(file_path: str) -> Dict[str, Any]:
    """
    Provede lokální transkripci a vrátí strukturovaná data s časy.
    """
    try:
        model = get_local_whisper()
        # word_timestamps=True zajistí přesnější segmentaci
        segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True)
        
        transcript_result = []
        full_text = []

        for segment in segments:
            transcript_result.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())

        return {
            "full_text": " ".join(full_text),
            "segments": transcript_result,
            "language": info.language
        }
    except Exception as e:
        return {"error": str(e), "full_text": "", "segments": []}

def transcribe_video_file(file_path: str, model_choice: str = "azure") -> Union[str, Dict[str, Any]]:
    """
    Původní funkce pro kompatibilitu + podpora Azure.
    Pokud jedeme lokálně, vracíme nově dict s timestamps, pokud to volající podporuje.
    """
    is_azure = "gpt" in model_choice.lower() or "azure" in model_choice.lower()
    
    # --- 1. LOKÁLNÍ PŘEPIS ---
    if not is_azure:
        # Použijeme novou chytrou funkci
        return transcribe_with_timestamps(file_path)

    # --- 2. AZURE PŘEPIS (Cloud) ---
    endpoint = os.getenv("SPEECH_AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("SPEECH_AZURE_OPENAI_API_KEY")
    api_version = os.getenv("SPEECH_AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("SPEECH_GPT4O_MINI_TRANSCRIBE_DEPLOYMENT_NAME")

    if not endpoint or not api_key or not deployment:
        return "Speech service is not configured."

    
    return "Azure transcription output (simplified for this file view)."