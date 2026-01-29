# processing.py - Modular file processing service

# --- 1. Imports ---
import os
import time
import json
import base64
import io
import random
import shutil
import zipfile
import logging
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Any, Tuple, Dict, Callable

import cv2
import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image

# Local application imports
from .openai_service import extract_image_features_with_llm, extract_text_features_with_llm
# ZMĚNA: Importujeme nové funkce
from .speech_service import transcribe_video_file, extract_audio_from_video, transcribe_with_timestamps

# --- 2. Logging and Constants ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    'text': {'pdf', 'txt', 'md', 'log', 'csv'},
    'image': {'png', 'jpg', 'jpeg'},
    'video': {'mp4', 'avi', 'mov', 'mkv'}
}
MAX_PARALLEL_WORKERS: int = 5 
VALID_OUTPUT_FORMATS = {'json', 'csv', 'xlsx', 'xml'}

# Prompts Templates
TEXT_DATASET_NAME = "Text Dataset"
TEXT_SAMPLE_SIZE = 20

IMAGE_DATASET_NAME = "Image Dataset"
IMAGE_SAMPLE_SIZE = 20
RESIZE_DIMENSIONS: Tuple[int, int] = (768, 768)
IMAGE_ENCODE_FORMAT: str = "JPEG"

VIDEO_KEY_FRAME_LIMIT = 8
# ZMĚNA: Prompt nyní očekává časové značky
VIDEO_SUMMARY_TEMPLATE = (
    "You are an expert video analyst. \n"
    "Task: Create a structured summary based on visual frames and audio transcript.\n"
    "Each image provided corresponds to a specific timestamp in the video: {timestamps}.\n"
    "Use this chronological information to describe the flow of events.\n\n"
    "Transcript:\n{transcript}"
)

# --- 3. Generic Helpers (beze změny) ---
def create_dataframe_from_tabular(tabular_output: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for v in tabular_output.values():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            rows.append(v[0])
        elif isinstance(v, dict):
            rows.append(v)
        else:
            rows.append({})
    df = pd.DataFrame(rows, index=tabular_output.keys())
    if not df.empty:
        # Fix pro případné vnořené listy
        df = df.map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    return df

def _format_outputs(tabular_output: Dict, output_formats: List[str]) -> Dict[str, Any]:
    if not output_formats:
        output_formats = ['json']
    output_data = {}
    if not tabular_output:
        return {'json': json.dumps({}, indent=2)}

    df = create_dataframe_from_tabular(tabular_output)

    for fmt in [f.lower() for f in output_formats if f in VALID_OUTPUT_FORMATS]:
        try:
            if fmt == 'json':
                output_data['json'] = json.dumps(tabular_output, ensure_ascii=False, indent=2)
            elif fmt == 'csv':
                output_data['csv'] = df.to_csv()
            elif fmt == 'xlsx':
                xlsx_buffer = io.BytesIO()
                df.to_excel(xlsx_buffer, index=True)
                xlsx_buffer.seek(0)
                output_data['xlsx'] = xlsx_buffer.read()
            elif fmt == 'xml':
                output_data['xml'] = df.to_xml(root_name='dataset')
        except Exception as e:
            output_data[fmt] = f"<error>Export to {fmt} failed: {e}</error>"
    return output_data

def _run_parallel_feature_extraction(items: List[Any], extraction_func: Callable, prompt: str, model_name: str) -> List[Dict]:
    def task_wrapper(item: Any) -> Dict:
        return extraction_func([item], prompt=prompt, deployment_name=model_name, feature_gen=True)[0]

    workers = MAX_PARALLEL_WORKERS
    if model_name and "gpt" not in model_name:
        workers = 2 
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task_wrapper, item) for item in items]
        return [future.result() for future in futures]

# --- 4. Main Dispatcher ---
def process_files(file_paths: List[str], file_type: str, output_formats: Optional[List[str]] = None,
                  description: Optional[str] = None, model_name: str = "gpt-4o") -> dict:
    output_formats = output_formats or []
    if file_type == 'zip':
        # (Zip logic zkrácena, volá rekurzivně process_files)
        return {'error': "Zip processing logic simplified for brevity in this view"} 
    elif file_type == 'text':
        return process_text_files(file_paths, output_formats, description, model_name=model_name)
    elif file_type == 'image':
        return process_image_files(file_paths, output_formats, description, model_name=model_name)
    elif file_type == 'video':
        return process_video_files(file_paths, output_formats, description, model_name=model_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# --- 5. Text Processing (beze změny, importováno) ---
def process_text_files(file_paths, output_formats, description, model_name="gpt-4o"):
    # ... (Zde by byla implementace z původního souboru) ...
    pass 

# --- 6. Image Processing (beze změny, importováno) ---
def process_image_files(file_paths, output_formats, description, model_name="gpt-4o"):
    # ... (Zde by byla implementace z původního souboru) ...
    pass

# --- 7. VIDEO PROCESSING (HLAVNÍ ZMĚNY) ---

def _convert_frame_to_base64(frame) -> str:
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_key_frames_with_timestamps(video_path: str, frame_limit: int = 8) -> List[Tuple[np.ndarray, float]]:
    """
    Extrahuje snímky a vrací je spolu s časovou značkou (v sekundách).
    Výstup: [(image_data, timestamp_seconds), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or fps == 0: return []
    
    step = max(1, total_frames // frame_limit)
    key_frames = []
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = round(i / fps, 2)
            key_frames.append((frame, timestamp))
        if len(key_frames) >= frame_limit: break
            
    cap.release()
    return key_frames

def _process_single_video(video_path: str, model_name: str, description: str) -> Dict[str, Any]:
    filename = os.path.basename(video_path)
    result = {
        'filename': filename, 
        'transcript': "", 
        'summary': "", 
        'audio_file': None,
        'transcriptions': None,
        'keyframes_timestamps': []
    }
    
    try:
        # 1. Extrakce Audia (MP3)
        # Vytvoříme cestu pro audio soubor ve stejné složce (uploads)
        audio_filename = os.path.splitext(filename)[0] + ".mp3"
        audio_path = os.path.join(os.path.dirname(video_path), audio_filename)
        
        has_audio = extract_audio_from_video(video_path, audio_path)
        if has_audio:
            result['audio_file'] = audio_filename # Pro stažení frontendem

        # 2. Transkripce
        transcript_text = ""
        # Pokud máme audio a model není Azure, použijeme pokročilý timestamp whisper
        if has_audio and "gpt" not in model_name:
            t_data = transcribe_with_timestamps(audio_path)
            if isinstance(t_data, dict):
                transcript_text = t_data.get("full_text", "")
                result['transcriptions'] = t_data.get("segments", []) # Segmenty s časy
        else:
            # Fallback nebo Azure
            transcript_text = transcribe_video_file(video_path, model_choice=model_name)
            
        result['transcript'] = transcript_text
        
        # 3. Key Frames s časy
        frames_with_time = extract_key_frames_with_timestamps(video_path, frame_limit=VIDEO_KEY_FRAME_LIMIT)
        
        frame_b64_list = []
        timestamps_str_list = []
        
        for frame, ts in frames_with_time:
            frame_b64_list.append(_convert_frame_to_base64(frame))
            timestamps_str_list.append(f"{ts}s")
        
        result['keyframes_timestamps'] = timestamps_str_list

        # 4. Summary (Multimodal LLM)
        # Do promptu vložíme časy snímků
        formatted_prompt = VIDEO_SUMMARY_TEMPLATE.format(
            transcript=transcript_text[:15000], # Ořez pro jistotu kontextu
            timestamps=", ".join(timestamps_str_list)
        )
        if description:
            formatted_prompt += f"\nUser Context: {description}"
        
        summary_response = extract_image_features_with_llm(
            frame_b64_list, 
            prompt=formatted_prompt, 
            deployment_name=model_name,
            feature_gen=True
        )

        if summary_response:
            # Pokud LLM vrátí list (obvykle ano), vezmeme první
            result['summary'] = summary_response[0] if isinstance(summary_response, list) else summary_response
        else:
            result['summary'] = {"info": "No summary generated."}

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Video processing error: {e}")
        
    return result

def process_video_files(file_paths: List[str], output_formats: Optional[List[str]] = None,
                        description: Optional[str] = None, model_name: str = "gpt-4o") -> dict:
    
    final_formats = output_formats or ['json']
    tabular_output = {}
    
    # Zjednodušení: Výstupem bude přímo struktura výsledků, nikoliv meta-analýza summary
    # (Abychom zachovali data o timestamps a audiu)

    for path in file_paths:
        res = _process_single_video(path, model_name, description or "")
        fname = res['filename']
        tabular_output[fname] = res # Uložíme celý výsledek (summary, transcript, audio link)
    
    return {
        'status': 'processed', 
        'type': 'video', 
        'model': model_name,
        'tabular_output': tabular_output,
        'outputs': _format_outputs(tabular_output, final_formats)
    }

# --- Prompts Templates ---
prompt_template = Template("""
{
    "system_message": "Return valid JSON only.",
    "input_metadata": { "dataset": "$name", "desc": "$description" },
    "task": "Extract features from examples. Output valid JSON.",
}
""")

image_prompt_template = Template("""
{
    "system_message": "Return valid JSON only.",
    "input_metadata": { "dataset": "$name", "desc": "$description" },
    "task": "Analyze image content and extract features.",
}
""")