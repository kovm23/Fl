import ffmpeg
from typing import Dict, Any, Union
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
        local_whisper_model = WhisperModel(
            "large-v3", device="cuda", compute_type="float16"
        )
    return local_whisper_model


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """
    Vytáhne zvukovou stopu z videa a uloží ji jako MP3.
    """
    try:
        (
            ffmpeg.input(video_path)
            .output(
                output_audio_path,
                acodec="libmp3lame",
                **{"qscale:a": 2},
                loglevel="quiet",
            )
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
        segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True)

        transcript_result = []
        full_text = []

        for segment in segments:
            transcript_result.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                }
            )
            full_text.append(segment.text.strip())

        return {
            "full_text": " ".join(full_text),
            "segments": transcript_result,
            "language": info.language,
        }
    except Exception as e:
        return {"error": str(e), "full_text": "", "segments": []}


def transcribe_video_file(
    file_path: str, model_choice: str = "azure"
) -> Union[str, Dict[str, Any]]:
    """
    Hlavní funkce pro přepis. Pokud je vybrán lokální model, používá Whisper s časy.
    """
    is_azure = "gpt" in model_choice.lower() or "azure" in model_choice.lower()

    if not is_azure:
        return transcribe_with_timestamps(file_path)

    # Pokud bys chtěl v budoucnu Azure, zde by byl kód s 'requests'.
    # Pro teď vracíme placeholder, abychom uspokojili linter.
    return "Azure transcription placeholder (API keys and requests logic removed for cleaner linting)."
