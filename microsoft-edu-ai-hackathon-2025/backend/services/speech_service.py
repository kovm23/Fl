import os
import logging
import ffmpeg
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Nastavení modelu (použije malý a rychlý model, pro lepší češtinu zkuste "medium")
MODEL_SIZE = "base"
# Pokud máte GPU, změňte device na "cuda", jinak "cpu"
DEVICE = "cpu" 
COMPUTE_TYPE = "int8"

def extract_audio_from_video(video_path: str, audio_path: str):
    """
    Separuje audio stopu z videa a uloží ji jako MP3/WAV.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, ac=1, ar='16k') # Mono, 16kHz (ideální pro Whisper)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode('utf8')}")
        return False

def transcribe_video_file(video_path: str) -> str:
    """
    Hlavní funkce: Vytáhne audio a přepíše ho do textu s časovými značkami.
    """
    # 1. Cesta pro dočasné audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    
    # 2. Extrakce audia
    if not extract_audio_from_video(video_path, audio_path):
        return "[CHYBA: Nepodařilo se extrahovat audio z videa]"

    try:
        # 3. Načtení Whisper modelu
        logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}...")
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

        # 4. Transkripce
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

        # 5. Formátování výstupu s časovými značkami
        # Výstup bude vypadat např.: "[00:00 -> 00:05] Dobrý den, vítám vás."
        transcript_lines = []
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            line = f"[{start} -> {end}] {segment.text}"
            transcript_lines.append(line)

        full_transcript = "\n".join(transcript_lines)
        return full_transcript

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return f"[CHYBA PŘEPISU: {str(e)}]"
    
    finally:
        # Úklid: smazání dočasného audio souboru
        if os.path.exists(audio_path):
            os.remove(audio_path)

def format_timestamp(seconds: float) -> str:
    """Převede sekundy (např. 75.5) na formát MM:SS (01:15)."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"