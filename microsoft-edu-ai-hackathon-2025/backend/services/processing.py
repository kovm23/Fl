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

# Third-party libraries
import cv2
import ffmpeg
import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image

# Local application imports
from .openai_service import extract_image_features_with_llm, extract_text_features_with_llm
from .speech_service import transcribe_video_file

# --- 2. Logging and Constants Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    'text': {'pdf', 'txt', 'md', 'log', 'csv'},
    'image': {'png', 'jpg', 'jpeg'},
    'video': {'mp4', 'avi', 'mov', 'mkv'}
}
MAX_PARALLEL_WORKERS: int = 5 # Sníženo pro lokální modely kvůli VRAM, zvedni na 10, pokud to stíhá
VALID_OUTPUT_FORMATS = {'json', 'csv', 'xlsx', 'xml'}

TEXT_DATASET_NAME = "Text Dataset"
TEXT_SAMPLE_SIZE = 20
DEFAULT_TARGET_VARIABLE = "<target>"

IMAGE_DATASET_NAME = "Image Dataset"
IMAGE_SAMPLE_SIZE = 20
RESIZE_DIMENSIONS: Tuple[int, int] = (768, 768)
IMAGE_ENCODE_FORMAT: str = "JPEG"

VIDEO_KEY_FRAME_LIMIT = 8
VIDEO_SUMMARY_TEMPLATE = (
    "You are an expert video analyst. "
    "Task: Create a structured summary of the video based on the Transcript and Key Frames.\n"
    "Transcript:\n{transcript}"
)


# --- 3. Generic Helper Functions ---

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
        df = df.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
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
            logger.error(f"Output format '{fmt}' generation failed: {e}")
            output_data[fmt] = f"<error>Export to {fmt} failed: {e}</error>"
    return output_data

def _run_parallel_feature_extraction(items: List[Any], extraction_func: Callable, prompt: str, model_name: str) -> List[Dict]:
    """Spustí extrakci paralelně, ale předává 'model_name'."""
    def task_wrapper(item: Any) -> Dict:
        # Tady předáváme deployment_name=model_name
        return extraction_func([item], prompt=prompt, deployment_name=model_name, feature_gen=True)[0]

    # Pro lokální modely můžeme chtít snížit paralelismus, aby nedošlo k OOM
    workers = MAX_PARALLEL_WORKERS
    if model_name and "gpt" not in model_name:
        workers = 2 # Opatrně s lokálními modely
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task_wrapper, item) for item in items]
        return [future.result() for future in futures]

# --- 4. Main Dispatcher ---

def _process_zip_archive(zip_path: str, output_formats: List[str], description: Optional[str], model_name: str) -> dict:
    temp_dir = os.path.join('uploads', f"temp_unzip_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(temp_dir)
            for name in files if not name.startswith(('__MACOSX', '.'))
        ]
        if not extracted_files: return {'status': 'error', 'message': 'ZIP empty'}

        inner_file_types = {
            ftype for f in extracted_files
            for ftype, exts in ALLOWED_EXTENSIONS.items()
            if os.path.splitext(f)[1].lower().replace('.', '') in exts
        }

        if len(inner_file_types) != 1:
            return {'status': 'error', 'message': f'Mixed file types in ZIP: {list(inner_file_types)}'}

        actual_file_type = inner_file_types.pop()
        
        if actual_file_type == 'text':
            return process_text_files(extracted_files, output_formats, description, model_name=model_name)
        elif actual_file_type == 'image':
            return process_image_files(extracted_files, output_formats, description, model_name=model_name)
        elif actual_file_type == 'video':
            return process_video_files(extracted_files, output_formats, description, model_name=model_name)
        else:
            return {'status': 'error', 'message': 'Unsupported type'}
    finally:
        shutil.rmtree(temp_dir)

def process_files(file_paths: List[str], file_type: str, output_formats: Optional[List[str]] = None,
                  description: Optional[str] = None, model_name: str = "gpt-4o") -> dict:
    output_formats = output_formats or []
    if file_type == 'zip':
        return _process_zip_archive(file_paths[0], output_formats, description, model_name)
    elif file_type == 'text':
        return process_text_files(file_paths, output_formats, description, model_name=model_name)
    elif file_type == 'image':
        return process_image_files(file_paths, output_formats, description, model_name=model_name)
    elif file_type == 'video':
        return process_video_files(file_paths, output_formats, description, model_name=model_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# --- 5. Text Processing ---

def _extract_texts_from_files(file_paths: List[str]) -> Dict[str, str]:
    extracted_texts = {}
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            _, ext = os.path.splitext(file_path)
            if ext.lower() == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted_texts[filename] = ''.join(page.extract_text() or '' for page in reader.pages)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_texts[filename] = f.read()
        except Exception as e:
            extracted_texts[filename] = f'<error: {e}>'
    return extracted_texts

def process_text_files(file_paths: List[str], output_formats: List[str], description: str,
                       target: Optional[str] = None, model_name: str = "gpt-4o") -> dict:
    extracted_texts = _extract_texts_from_files(file_paths)
    texts_list = list(extracted_texts.values())
    rep_texts = select_representative_images(texts_list, sample_size=TEXT_SAMPLE_SIZE)

    prompt = prompt_template.substitute(
        name=TEXT_DATASET_NAME,
        description=description or "",
        target=target or DEFAULT_TARGET_VARIABLE,
        examples='\n---\n'.join(rep_texts[:3]) # Jen pár příkladů do promptu
    )
    
    # 1. Získání struktury (Schema)
    feature_spec = extract_text_features_with_llm(rep_texts[:1], prompt=prompt, deployment_name=model_name)
    feature_prompt = str(feature_spec[0]) if feature_spec else ""

    # 2. Extrakce hodnot
    # Použijeme model_name
    all_features = _run_parallel_feature_extraction(texts_list, extract_text_features_with_llm, feature_prompt, model_name)
    
    tabular_output = dict(zip(extracted_texts.keys(), all_features))
    output_data = _format_outputs(tabular_output, output_formats)

    return {
        'status': 'processed', 'type': 'text', 'model': model_name,
        'tabular_output': tabular_output, 'outputs': output_data
    }

# --- 6. Image Processing ---

def _preprocess_images(file_paths: List[str]) -> Dict[str, Optional[str]]:
    processed_images = {}
    for path in file_paths:
        try:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                resized_img = img_rgb.resize(RESIZE_DIMENSIONS)
                buffered = io.BytesIO()
                resized_img.save(buffered, format=IMAGE_ENCODE_FORMAT)
                processed_images[path] = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            processed_images[path] = None
    return processed_images

def process_image_files(file_paths: List[str], output_formats: List[str], description: Optional[str] = None, model_name: str = "gpt-4o") -> Dict:
    processed_images = _preprocess_images(file_paths)
    valid_base64_list = [b64 for b64 in processed_images.values() if b64 is not None]

    if not valid_base64_list:
        return {'status': 'error', 'message': 'No valid images.'}

    rep_images = select_representative_images(valid_base64_list, sample_size=IMAGE_SAMPLE_SIZE)
    prompt = image_prompt_template.substitute(name=IMAGE_DATASET_NAME, description=description or "")

    # 1. Schema
    feature_spec = extract_image_features_with_llm(rep_images[:1], prompt=prompt, deployment_name=model_name)
    feature_prompt = str(feature_spec[0]) if feature_spec else ""

    # 2. Values
    # Musíme upravit _run_parallel_feature_extraction aby brala model_name
    # Zde voláme přímo funkci s list comp, nebo upravíme wrapper
    
    # Paralelní extrakce s ohledem na model
    all_features = _run_parallel_feature_extraction(valid_base64_list, extract_image_features_with_llm, feature_prompt, model_name)

    tabular_output = {}
    feature_iterator = iter(all_features)
    for path, b64 in processed_images.items():
        if b64 is not None:
            tabular_output[os.path.basename(path)] = next(feature_iterator)
        else:
            tabular_output[os.path.basename(path)] = {"error": "Load failed"}

    output_data = _format_outputs(tabular_output, output_formats)

    return {
        'status': 'processed', 'type': 'image', 'model': model_name,
        'tabular_output': tabular_output, 'outputs': output_data
    }

# --- 7. Video Processing ---

def _convert_frames_to_base64(key_frames: List[Any]) -> List[str]:
    frame_b64_list = []
    for frame in key_frames:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            frame_b64_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        except Exception:
            pass
    return frame_b64_list

def _process_single_video(video_path: str, model_name: str) -> Dict[str, Any]:
    filename = os.path.basename(video_path)
    result = {'filename': filename, 'transcript': "", 'summary': "", 'error': None}
    try:
        # 1. Transkripce (Local nebo Azure podle model_name)
        result['transcript'] = transcribe_video_file(video_path, model_choice=model_name)
        
        # 2. Key Frames
        key_frames = extract_key_frames(video_path, frame_limit=VIDEO_KEY_FRAME_LIMIT)
        frame_b64_list = _convert_frames_to_base64(key_frames)

        # 3. Summary (Multimodal LLM)
        summary_prompt = VIDEO_SUMMARY_TEMPLATE.format(transcript=result['transcript'])
        
        # Voláme LLM (Llava nebo GPT) s obrázky i textem
        summary_response = extract_image_features_with_llm(
            frame_b64_list, 
            prompt=summary_prompt, 
            deployment_name=model_name, # <-- Důležité
            feature_gen=True
        )

        if summary_response:
            result['summary'] = str(summary_response[0])
        else:
            result['summary'] = "No summary generated."

    except Exception as e:
        result['error'] = str(e)
    return result

def _analyze_video_summaries(summaries: Dict[str, str], output_formats: List[str], description: str, model_name: str) -> Dict:
    # Analyzujeme textové shrnutí videa jako text dataset
    temp_files = []
    try:
        for fname, text in summaries.items():
            tpath = os.path.join('uploads', f"summary_{fname}.txt")
            with open(tpath, 'w') as f: f.write(text)
            temp_files.append(tpath)
            
        return process_text_files(temp_files, output_formats, description, model_name=model_name)
    finally:
        for f in temp_files: 
            if os.path.exists(f): os.remove(f)

def process_video_files(file_paths: List[str], output_formats: Optional[List[str]] = None,
                        description: Optional[str] = None, model_name: str = "gpt-4o") -> dict:
    
    final_formats = output_formats or ['json']
    consolidated_output = {}
    all_summaries = {}

    for path in file_paths:
        res = _process_single_video(path, model_name)
        fname = res['filename']
        all_summaries[fname] = res['summary']
        
        consolidated_output[fname] = {
            "transcript_preview": res['transcript'][:200] + "...",
            "summary_raw": res['summary']
        }
    
    # 4. Strukturovaná analýza z textových shrnutí
    analysis = _analyze_video_summaries(all_summaries, final_formats, description or "", model_name)
    
    # Merge výsledků
    final_tabular = analysis.get('tabular_output', {})
    
    return {
        'status': 'processed', 'type': 'video', 'model': model_name,
        'tabular_output': final_tabular,
        'outputs': _format_outputs(final_tabular, final_formats)
    }

# --- 8. Utils ---

def select_representative_images(items: List[Any], sample_size: int) -> List[Any]:
    if len(items) <= sample_size: return items
    return random.sample(items, sample_size)

def extract_key_frames(video_path: str, frame_limit: int = 8) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    # Jednoduchá extrakce v pravidelných intervalech
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return []
    
    step = max(1, total_frames // frame_limit)
    frames = []
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= frame_limit: break
            
    cap.release()
    return frames

# --- 9. Prompts ---
# Zkráceno pro přehlednost, templates zůstávají stejné jako v původním souboru
prompt_template = Template("""
{
    "system_message": "Return valid JSON only.",
    "input_metadata": { "dataset": "$name", "desc": "$description" },
    "task": "Extract features from examples. Output valid JSON.",
    "output_format": { "features": [ {"feature_name": "...", "value": "..."} ] }
}
""")

image_prompt_template = Template("""
{
    "system_message": "Return valid JSON only.",
    "input_metadata": { "dataset": "$name", "desc": "$description" },
    "task": "Analyze image content and extract features.",
    "output_format": { "features": [ {"feature_name": "...", "value": "..."} ] }
}
""")