import os
import requests
import ffmpeg
import time
from typing import Literal
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# Globální proměnná pro uložení načteného modelu (aby se nenačítal při každém requestu)
local_whisper_model = None

def get_local_whisper():
    """Načte model do VRAM pouze jednou (Singleton pattern)."""
    global local_whisper_model
    if local_whisper_model is None:
        print("Loading Local Whisper Large-v3 model to GPU... (this may take a moment)")
        # 'large-v3' je nejlepší open-source model, na 40GB VRAM běží s prstem v nose
        # compute_type="float16" výrazně zrychluje na NVIDIA GPU
        local_whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    return local_whisper_model

def transcribe_video_file(file_path: str, model_choice: str = "azure") -> str:
    """
    Transcribes video file.
    If model_choice contains 'gpt' or 'azure', uses Azure Speech API.
    Otherwise (for local models), uses local Faster-Whisper on GPU.
    """
    
    # --- 1. LOKÁLNÍ PŘEPIS (Pokud není vybrán Azure/GPT) ---
    is_azure = "gpt" in model_choice.lower() or "azure" in model_choice.lower()
    
    if not is_azure:
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            print(f"Starting local transcription for {os.path.basename(file_path)}...")
            model = get_local_whisper()
            
            # Faster-whisper umí číst video soubory přímo, nemusíme extrahovat audio přes ffmpeg manuálně
            # beam_size=5 zvyšuje přesnost
            segments, info = model.transcribe(file_path, beam_size=5, language=None, task="transcribe")
            
            # Spojení segmentů do jednoho textu
            full_text = "".join([segment.text for segment in segments])
            print("Local transcription finished.")
            return full_text
            
        except Exception as e:
            return f"Local transcription failed: {str(e)}"

    # --- 2. AZURE PŘEPIS (Původní kód) ---
    endpoint = os.getenv("SPEECH_AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("SPEECH_AZURE_OPENAI_API_KEY")
    api_version = os.getenv("SPEECH_AZURE_OPENAI_API_VERSION")

    if not endpoint or not api_key:
        return "Speech service is not configured."

    # Defaultně fast, pokud není specifikováno jinak v env
    deployment = os.getenv("SPEECH_GPT4O_MINI_TRANSCRIBE_DEPLOYMENT_NAME")

    if not deployment:
        return "Deployment for speech not found in ENV."

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    path_to_process = file_path
    temp_audio_path = None

    try:
        _, extension = os.path.splitext(file_path.lower())
        # Pokud je to video, vytáhneme audio pro Azure (Azure má limit na velikost)
        if extension in video_extensions:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            temp_audio_path = os.path.join('uploads', f"temp_audio_{base_name}_{int(time.time())}.wav")

            try:
                ffmpeg.input(file_path).output(
                    temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k'
                ).run(quiet=True, overwrite_output=True)
            except ffmpeg.Error as e:
                error_details = e.stderr.decode('utf8') if e.stderr else 'Unknown FFmpeg error'
                return f"Failed to extract audio from video: {error_details}"

            path_to_process = temp_audio_path

        # Určení Content-Type
        lower_case_path = path_to_process.lower()
        if lower_case_path.endswith(".mp4") or lower_case_path.endswith(".m4a"):
            content_type = "audio/mp4"
        elif lower_case_path.endswith(".mp3") or lower_case_path.endswith(".mpeg"):
            content_type = "audio/mpeg"
        elif lower_case_path.endswith(".wav"):
            content_type = "audio/wav"
        else:
            content_type = "application/octet-stream"

        url = f"{endpoint}openai/deployments/{deployment}/audio/transcriptions?api-version={api_version}"
        with open(path_to_process, "rb") as f:
            response = requests.post(
                url,
                headers={"api-key": api_key},
                files={"file": (os.path.basename(path_to_process), f, content_type)},
                data={"model": "gpt-4o-transcribe", "response_format": "json"},
                timeout=300,
            )

        if response.status_code != 200:
            return f"Transcription failed ({response.status_code}): {response.text}"

        return response.json().get("text", "(no text found)")

    except Exception as e:
        return f"An error occurred during the API call: {e}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass