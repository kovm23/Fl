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
import numpy as np
import pandas as pd
import PyPDF2
from PIL import Image

# Local application imports
from .openai_service import (
    extract_image_features_with_llm,
    extract_text_features_with_llm,
)
from .speech_service import transcribe_video_file

# --- 2. Logging and Constants Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# General constants
ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt", "md", "log", "csv"},
    "image": {"png", "jpg", "jpeg", "heic", "webp", "gif"},
    "video": {"mp4", "avi", "mov", "mkv"},
}
MAX_PARALLEL_WORKERS: int = 10
VALID_OUTPUT_FORMATS = {"json", "csv", "xlsx", "xml"}

# Text processing constants
TEXT_DATASET_NAME = "Text Dataset"
TEXT_SAMPLE_SIZE = 20
DEFAULT_TARGET_VARIABLE = "<target>"

# Image processing constants
IMAGE_DATASET_NAME = "Image Dataset"
IMAGE_SAMPLE_SIZE = 20
RESIZE_DIMENSIONS: Tuple[int, int] = (768, 768)
IMAGE_ENCODE_FORMAT: str = "JPEG"

# Video processing constants
KEY_FRAME_LIMIT = 8
# ZJEDNODUŠENÝ PROMPT PRO VIDEO (aby nedocházelo k chybám s JSONem)
SUMMARY_PROMPT_TEMPLATE = (
    "Analyze this video based on its transcript and key frames.\n"
    "Transcript (with timestamps):\n{transcript}\n\n"
    "Key Frames Context:\n{frame_context}\n\n"
    "TASK: Write a detailed descriptive summary of the video content, focusing on visual elements, "
    "speakers, topics, and overall atmosphere. This summary will be used to extract features later."
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
        output_formats = ["json"]

    output_data = {}
    if not tabular_output:
        return {"json": json.dumps({}, indent=2)}

    df = create_dataframe_from_tabular(tabular_output)

    for fmt in [f.lower() for f in output_formats if f in VALID_OUTPUT_FORMATS]:
        try:
            if fmt == "json":
                output_data["json"] = json.dumps(
                    tabular_output, ensure_ascii=False, indent=2
                )
            elif fmt == "csv":
                output_data["csv"] = df.to_csv()
            elif fmt == "xlsx":
                xlsx_buffer = io.BytesIO()
                df.to_excel(xlsx_buffer, index=True)
                xlsx_buffer.seek(0)
                output_data["xlsx"] = xlsx_buffer.read()
            elif fmt == "xml":
                output_data["xml"] = df.to_xml(root_name="dataset")
        except Exception as e:
            logger.error(f"Output format '{fmt}' generation failed: {e}")
            output_data[fmt] = f"<error>Export to {fmt} failed: {e}</error>"
    return output_data


# --- 4. Main Dispatcher and ZIP Processor ---


def _process_zip_archive(
    zip_path: str, output_formats: List[str], description: Optional[str], model_provider: str
) -> dict:
    temp_dir = os.path.join("uploads", f"temp_unzip_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(temp_dir)
            for name in files
            if not name.startswith(("__MACOSX", "."))
        ]

        if not extracted_files:
            return {"status": "error", "message": "The ZIP file is empty."}

        inner_file_types = {
            ftype
            for f in extracted_files
            for ftype, exts in ALLOWED_EXTENSIONS.items()
            if os.path.splitext(f)[1].lower().replace(".", "") in exts
        }

        if len(inner_file_types) > 1:
            return {
                "status": "error",
                "message": f"ZIP file contains multiple file types: {list(inner_file_types)}",
            }
        if not inner_file_types:
            return {
                "status": "error",
                "message": "No supported file types found in the ZIP file.",
            }

        actual_file_type = inner_file_types.pop()
        logger.info(
            f"Processing {len(extracted_files)} files of type '{actual_file_type}' from ZIP archive..."
        )

        if actual_file_type == "text":
            return process_text_files(extracted_files, output_formats, description, model_provider=model_provider)
        elif actual_file_type == "image":
            return process_image_files(extracted_files, output_formats, description, model_provider=model_provider)
        elif actual_file_type == "video":
            return process_video_files(extracted_files, output_formats, description, model_provider=model_provider)
        else:
            return {
                "status": "error",
                "message": f'File type "{actual_file_type}" in ZIP is not supported.',
            }
    finally:
        shutil.rmtree(temp_dir)


def process_files(
    file_paths: List[str],
    file_type: str,
    output_formats: Optional[List[str]] = None,
    description: Optional[str] = None,
    model_provider: str = "azure",
) -> dict:
    output_formats = output_formats or []
    if file_type == "zip":
        return _process_zip_archive(file_paths[0], output_formats, description, model_provider)
    elif file_type == "text":
        return process_text_files(
            file_paths, output_formats, description, model_provider=model_provider
        )
    elif file_type == "image":
        return process_image_files(
            file_paths, output_formats, description, model_provider=model_provider
        )
    elif file_type == "video":
        return process_video_files(
            file_paths, output_formats, description, model_provider=model_provider
        )
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


# --- 5. Text File Processing ---

def _extract_texts_from_files(file_paths: List[str]) -> Dict[str, str]:
    extracted_texts = {}
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            _, ext = os.path.splitext(file_path)
            if ext.lower() == ".pdf":
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted_texts[filename] = "".join(
                        page.extract_text() or "" for page in reader.pages
                    )
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_texts[filename] = f.read()
        except Exception as e:
            extracted_texts[filename] = f"<error extracting text: {e}>"
    return extracted_texts

def process_text_files(file_paths, output_formats, description, target=None, model_provider="azure"):
    logger.info(f"Processing text files with provider: {model_provider}")
    extracted_texts = _extract_texts_from_files(file_paths)
    texts_list = list(extracted_texts.values())
    rep_texts = select_representative_images(texts_list, sample_size=TEXT_SAMPLE_SIZE)

    prompt = prompt_template.substitute(
        name=TEXT_DATASET_NAME,
        description=description or "",
        target=target or DEFAULT_TARGET_VARIABLE,
        examples="\n---\n".join(rep_texts),
    )
    feature_spec = extract_text_features_with_llm(
        rep_texts, prompt=prompt, provider=model_provider
    )
    feature_prompt = str(feature_spec[0]) if feature_spec else ""

    def extraction_func(text, prompt=feature_prompt):
        return extract_text_features_with_llm(
            [text], prompt=prompt, provider=model_provider
        )[0]

    workers = 3 if model_provider in ["ollama_qwen", "ollama_llama"] else 10
    all_features = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extraction_func, text) for text in texts_list]
        for future in as_completed(futures):
            all_features.append(future.result())
    tabular_output = dict(zip(extracted_texts.keys(), all_features))
    output_data = _format_outputs(tabular_output, output_formats)
    return {
        "status": "processed", "type": "text", "files": file_paths,
        "output_formats": output_formats, "description": description,
        "tabular_output": tabular_output, "outputs": output_data,
        "feature_specification": feature_prompt,
    }


# --- 6. Image File Processing ---

def _preprocess_images(file_paths: List[str]) -> Dict[str, Optional[str]]:
    processed_images = {}
    for path in file_paths:
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
                resized_img = img_rgb.resize(RESIZE_DIMENSIONS)
                buffered = io.BytesIO()
                resized_img.save(buffered, format=IMAGE_ENCODE_FORMAT)
                processed_images[path] = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.warning(f"Could not process image {path}. Error: {e}")
            processed_images[path] = None
    return processed_images

def process_image_files(file_paths, output_formats, description=None, model_provider="azure"):
    logger.info(f"Processing image files with provider: {model_provider}")
    processed_images = _preprocess_images(file_paths)
    valid_base64_list = [b64 for b64 in processed_images.values() if b64 is not None]

    if not valid_base64_list:
        return {"status": "error", "message": "No valid images could be processed."}

    rep_images = select_representative_images(valid_base64_list, sample_size=IMAGE_SAMPLE_SIZE)
    prompt = image_prompt_template.substitute(name=IMAGE_DATASET_NAME, description=description or "")
    
    feature_spec = extract_image_features_with_llm(rep_images, prompt=prompt, provider=model_provider)
    feature_prompt = str(feature_spec[0]) if feature_spec else ""
    
    def extraction_func(img_b64, prompt=feature_prompt):
        return extract_image_features_with_llm([img_b64], prompt=prompt, provider=model_provider)[0]

    workers = 3 if model_provider in ["ollama_qwen", "ollama_llama"] else 10
    all_features = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extraction_func, img_b64) for img_b64 in valid_base64_list]
        for future in as_completed(futures):
            all_features.append(future.result())

    tabular_output = {}
    feature_iterator = iter(all_features)
    for path, b64 in processed_images.items():
        filename = os.path.basename(path)
        if b64 is not None:
            tabular_output[filename] = next(feature_iterator)
        else:
            tabular_output[filename] = {"error": "Failed to process image."}

    output_data = _format_outputs(tabular_output, output_formats)
    return {
        "status": "processed", "type": "image", "original_files": file_paths,
        "output_formats": output_formats, "description": description,
        "tabular_output": tabular_output, "outputs": output_data,
        "feature_specification": feature_prompt,
    }


# --- 7. Video File Processing ---

def _convert_frames_to_base64(key_frames: List[Tuple[str, np.ndarray]], filename: str) -> List[str]:
    """Converts a list of (timestamp, frame) tuples to base64 strings."""
    frame_b64_list = []
    # key_frames je nyní seznam (timestamp_str, image_array)
    for i, (ts, frame) in enumerate(key_frames):
        try:
            if isinstance(frame, np.ndarray):
                if frame.size == 0: continue
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                continue

            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            frame_b64_list.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to process frame {i} in {filename}: {e}")
    return frame_b64_list


def _process_single_video(video_path: str, model_provider: str) -> Dict[str, Any]:
    """Processes one video file to get its transcript and summary."""
    filename = os.path.basename(video_path)
    result = {"filename": filename, "transcript": "", "summary": "", "error": None}
    try:
        # 1. Získání transkriptu s časovými značkami
        result["transcript"] = transcribe_video_file(video_path)
        
        # 2. Získání klíčových snímků s časovými značkami
        # key_frames je list tuplů: [("00:05", img1), ("00:45", img2), ...]
        key_frames_with_ts = extract_key_frames(video_path, frame_limit=KEY_FRAME_LIMIT)
        
        if not key_frames_with_ts:
            raise ValueError("No key frames extracted.")

        # Rozbalení dat pro LLM
        frame_b64_list = _convert_frames_to_base64(key_frames_with_ts, filename)
        timestamps_list = [item[0] for item in key_frames_with_ts]

        if not frame_b64_list:
            raise ValueError("No valid frames could be processed.")

        # 3. Vytvoření kontextu pro prompt ("Image 1 is at 00:05", etc.)
        frame_context_str = ""
        for idx, ts in enumerate(timestamps_list):
            frame_context_str += f"Image {idx+1}: Timestamp {ts}\n"

        # 4. Finální prompt
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            transcript=result["transcript"],
            frame_context=frame_context_str
        )

        # 5. Volání LLM (posíláme obrázky + text s časy)
        summary_response = extract_image_features_with_llm(
            frame_b64_list, prompt=summary_prompt, feature_gen=True, provider=model_provider
        )

        if summary_response and isinstance(summary_response, list):
            result["summary"] = str(summary_response[0])
        else:
            result["summary"] = "[ERROR: Invalid or empty summary response]"

    except Exception as e:
        logger.error(f"Failed to process video {filename}: {e}")
        result["error"] = str(e)
    return result


def _analyze_video_summaries(
    summaries: Dict[str, str], output_formats: List[str], description: str, model_provider: str
) -> Dict:
    temp_summary_files = []
    os.makedirs("uploads", exist_ok=True)
    try:
        for filename, summary_text in summaries.items():
            temp_path = os.path.join(
                "uploads", f"summary_{os.path.splitext(filename)[0]}.txt"
            )
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            temp_summary_files.append(temp_path)

        text_analysis = process_text_files(
            temp_summary_files, output_formats, description, model_provider=model_provider
        )
        tabular_output = {}
        analysis_data = text_analysis.get("tabular_output", {})
        for path in temp_summary_files:
            original_fname = next(
                (fname for fname in summaries if os.path.splitext(fname)[0] in os.path.basename(path)),
                None,
            )
            if original_fname:
                tabular_output[original_fname] = analysis_data.get(
                    os.path.basename(path), {"error": "No data returned"}
                )
        return {
            "tabular_output": tabular_output,
            "feature_specification": text_analysis.get("feature_specification"),
        }
    finally:
        for f in temp_summary_files:
            if os.path.exists(f):
                os.remove(f)


def process_video_files(file_paths, output_formats=None, description=None, model_provider="azure"):
    logger.info(f"Processing video files with provider: {model_provider}")
    if not file_paths:
        return {"status": "error", "error": "file_paths must be non-empty", "type": "video"}

    final_formats = [fmt.lower() for fmt in (output_formats or []) if fmt in VALID_OUTPUT_FORMATS] or ["json"]
    all_transcripts, all_summaries, consolidated_output = {}, {}, {}

    for path in file_paths:
        if not (isinstance(path, str) and os.path.isfile(path)):
            continue

        result = _process_single_video(path, model_provider)
        all_transcripts[result["filename"]] = result["transcript"]
        all_summaries[result["filename"]] = result["summary"]
        if result["error"]:
            consolidated_output[result["filename"]] = {"error": result["error"]}

    valid_summaries = {k: v for k, v in all_summaries.items() if v and not v.startswith("[ERROR")}
    feature_spec = None
    if valid_summaries:
        analysis_result = _analyze_video_summaries(
            valid_summaries, final_formats, description, model_provider
        )
        consolidated_output.update(analysis_result.get("tabular_output", {}))
        feature_spec = analysis_result.get("feature_specification")

    output_data = _format_outputs(consolidated_output, final_formats)

    return {
        "status": "processed", "type": "video", "original_files": file_paths,
        "output_formats": final_formats, "description": description or "",
        "tabular_output": consolidated_output, "outputs": output_data,
        "feature_specification": feature_spec,
        "transcripts": all_transcripts,
        "summaries": all_summaries,
    }


# --- 8. Miscellaneous Utilities ---

def select_representative_images(items: List[Any], sample_size: int) -> List[Any]:
    if len(items) <= sample_size:
        return items
    return random.sample(items, sample_size)

def format_timestamp_cv(seconds: float) -> str:
    """Helper pro formátování času z OpenCV."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def extract_key_frames(
    video_path: str, frame_limit: int = 8, sharpness_threshold: float = 100.0
) -> List[Tuple[str, np.ndarray]]:
    """
    Extracts key frames AND their timestamps from a video.
    Returns: List of tuples (timestamp_string, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    candidates = [] # (change_score, timestamp_str, frame)
    frame_skip = 15
    frame_count = 0
    prev_gray = None

    while True:
        is_read, frame = cap.read()
        if not is_read:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        # Získání časové značky v milisekundách
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_str = format_timestamp_cv(msec / 1000.0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray_frame, cv2.CV_64F).var() < sharpness_threshold:
            continue

        change_score = (
            cv2.absdiff(prev_gray, gray_frame).sum() if prev_gray is not None else 0
        )
        candidates.append((change_score, timestamp_str, frame))
        prev_gray = gray_frame

    cap.release()
    
    # Seřadit podle skóre změny a vzít top N
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_candidates = candidates[:frame_limit]
    
    # Vrátit seřazené chronologicky (podle času), ne podle skóre
    # Protože chceme, aby LLM viděl příběh popořadě
    # top_candidates je list (score, timestamp, frame)
    # seřadíme podle timestamp (string porovnání funguje u MM:SS dobře)
    top_candidates.sort(key=lambda item: item[1])

    # Vrátit jen (timestamp, frame)
    return [(item[1], item[2]) for item in top_candidates]


# --- 9. Prompt Templates (Opravené pro zamezení Prompt Leaku) ---

prompt_template = Template(
    """
    You are an expert data analyst. Your task is to analyze the provided text (which is a summary of a video/document) and extract a structured dataset of features.

    INPUT CONTEXT:
    Dataset Name: $name
    Description: $description
    Target Variable: $target
    Examples: $examples

    INSTRUCTIONS:
    1. Analyze the content deeply.
    2. Extract 15-20 distinct, meaningful features (categorical or descriptive).
    3. Focus on visual style, content, tone, sentiment, and specific details.
    4. RETURN ONLY RAW JSON. Do not use Markdown (```json). Do not add explanations.

    REQUIRED OUTPUT FORMAT:
    {
        "Feature Name 1": "Value 1",
        "Feature Name 2": "Value 2",
        "Tone": "Serious/Humorous/etc",
        "Main Topic": "...",
        "Visual Style": "..."
    }
    """
)

image_prompt_template = Template(
    """
    You are an expert computer vision analyst. Analyze the provided image(s) and metadata to construct a tabular dataset row.

    INPUT CONTEXT:
    Dataset Name: $name
    Description: $description

    INSTRUCTIONS:
    1. Extract 15-20 distinct visual features.
    2. Focus on colors, objects, lighting, composition, text presence, and atmosphere.
    3. RETURN ONLY RAW JSON. Do not use Markdown.

    REQUIRED OUTPUT FORMAT:
    {
        "Dominant Color": "...",
        "Lighting": "...",
        "Number of People": "...",
        "Setting": "...",
        "Object 1": "Present",
        "Text Content": "..."
    }
    """
)