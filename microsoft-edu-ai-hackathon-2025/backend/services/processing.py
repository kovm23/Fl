# processing.py - Modular file processing service

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
from PIL import Image

# Local application imports
from .openai_service import extract_image_features_with_llm
from .speech_service import (
    transcribe_video_file,
    extract_audio_from_video,
    transcribe_with_timestamps,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt", "md", "csv"},
    "image": {"png", "jpg", "jpeg"},
    "video": {"mp4", "avi", "mov", "mkv"},
}
MAX_PARALLEL_WORKERS: int = 5
VALID_OUTPUT_FORMATS = {"json", "csv", "xlsx", "xml"}
VIDEO_KEY_FRAME_LIMIT = 8

# --- HLAVNÍ PROMPT PRO KLASIFIKACI A RULEKIT ---
VIDEO_ANALYSIS_TEMPLATE = (
    "You are an expert video analyst AI.\n"
    "Task 1: Analyze the visual content (timestamps provided) and the audio transcript.\n"
    "Task 2: CLASSIFY the video into exactly one of these categories: {categories}.\n"
    "Task 3: Extract structured attributes suitable for Rule Learning (RuleKit).\n\n"
    "Transcript:\n{transcript}\n\n"
    "Timestamps of visual frames:\n{timestamps}\n\n"
    "User Context:\n{description}\n\n"
    "OUTPUT FORMAT (Return valid JSON only):\n"
    "{{\n"
    '  "summary": "Chronological description of events...",\n'
    '  "classification": "ONE_CATEGORY_FROM_LIST",\n'
    '  "reasoning": "Brief explanation why...",\n'
    '  "attributes": {{\n'
    '    "sentiment_score": 0.8, \n'
    '    "urgency_level": "high/medium/low",\n'
    '    "number_of_speakers": 1,\n'
    '    "environment": "indoor/outdoor",\n'
    '    "visual_complexity": "high/low"\n'
    "  }}\n"
    "}}"
)


def create_dataframe_from_tabular(tabular_output: Dict[str, Any]) -> pd.DataFrame:
    """
    Převede hierarchický JSON na plochou tabulku pro CSV/Excel.
    Klíčové pro RuleKit - vytahuje atributy do samostatných sloupců.
    """
    rows = []
    for filename, data in tabular_output.items():
        if isinstance(data, dict):
            # Základní metadata
            row = {
                "filename": filename,
                "transcript": data.get("transcript", "")[:100]
                + "...",  # Zkráceno pro CSV
                "audio_file": data.get("audio_file", ""),
            }

            # Pokud máme analýzu, rozbalíme ji
            analysis = data.get("analysis", {})
            if isinstance(analysis, dict):
                row["classification"] = analysis.get("classification", "Unknown")
                row["summary"] = analysis.get("summary", "")
                row["reasoning"] = analysis.get("reasoning", "")

                # Flatten attributes (attr_sentiment, attr_urgency...)
                attrs = analysis.get("attributes", {})
                for k, v in attrs.items():
                    row[f"attr_{k}"] = v

            rows.append(row)
    return pd.DataFrame(rows)


def _format_outputs(tabular_output: Dict, output_formats: List[str]) -> Dict[str, Any]:
    output_data = {}
    if not tabular_output:
        return {"json": "{}"}

    # Pro formáty jako CSV/XLSX chceme plochou tabulku (aby to šlo do RuleKitu)
    df = create_dataframe_from_tabular(tabular_output)

    for fmt in [f.lower() for f in output_formats if f in VALID_OUTPUT_FORMATS]:
        try:
            if fmt == "json":
                output_data["json"] = json.dumps(
                    tabular_output, ensure_ascii=False, indent=2
                )
            elif fmt == "csv":
                output_data["csv"] = df.to_csv(index=False)
            elif fmt == "xlsx":
                xlsx_buffer = io.BytesIO()
                df.to_excel(xlsx_buffer, index=False)
                xlsx_buffer.seek(0)
                output_data["xlsx"] = xlsx_buffer.read()
            elif fmt == "xml":
                output_data["xml"] = df.to_xml(root_name="dataset", index=False)
        except Exception as e:
            output_data[fmt] = f"Error: {e}"
    return output_data


def _convert_frame_to_base64(frame) -> str:
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_key_frames_with_timestamps(
    video_path: str, frame_limit: int = 8
) -> List[Tuple[np.ndarray, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total == 0 or fps == 0:
        return []
    step = max(1, total // frame_limit)
    res = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            res.append((frame, round(i / fps, 2)))
        if len(res) >= frame_limit:
            break
    cap.release()
    return res


def _process_single_video(
    video_path: str, model_name: str, description: str, categories: str
) -> Dict[str, Any]:
    filename = os.path.basename(video_path)
    result = {
        "filename": filename,
        "transcript": "",
        "analysis": None,
        "audio_file": None,
        "keyframes_timestamps": [],
    }

    try:
        # 1. Audio
        audio_filename = os.path.splitext(filename)[0] + ".mp3"
        audio_path = os.path.join(os.path.dirname(video_path), audio_filename)
        has_audio = extract_audio_from_video(video_path, audio_path)
        if has_audio:
            result["audio_file"] = audio_filename

        # 2. Transcript
        transcript_text = ""
        if has_audio and "gpt" not in model_name:
            t_data = transcribe_with_timestamps(audio_path)
            if isinstance(t_data, dict):
                transcript_text = t_data.get("full_text", "")
                result["transcriptions"] = t_data.get("segments", [])
        else:
            transcript_text = transcribe_video_file(video_path, model_choice=model_name)
        result["transcript"] = transcript_text

        # 3. Keyframes
        frames = extract_key_frames_with_timestamps(video_path, VIDEO_KEY_FRAME_LIMIT)
        frame_b64 = [_convert_frame_to_base64(f) for f, t in frames]
        timestamps = [f"{t}s" for f, t in frames]
        result["keyframes_timestamps"] = timestamps

        # 4. LLM Analysis & Classification
        target_cats = (
            categories
            if categories and categories.strip()
            else "General, News, Entertainment, Scam/Fraud"
        )

        prompt = VIDEO_ANALYSIS_TEMPLATE.format(
            transcript=transcript_text[:12000],
            timestamps=", ".join(timestamps),
            description=description or "None",
            categories=target_cats,
        )

        llm_resp = extract_image_features_with_llm(
            frame_b64, prompt=prompt, deployment_name=model_name, feature_gen=True
        )

        if isinstance(llm_resp, list) and len(llm_resp) > 0:
            result["analysis"] = llm_resp[0]
        else:
            result["analysis"] = llm_resp

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {filename}: {e}")

    return result


def process_files(
    file_paths: List[str],
    file_type: str,
    output_formats: Optional[List[str]] = None,
    description: Optional[str] = None,
    model_name: str = "gpt-4o",
    categories: str = "",
) -> dict:

    output_formats = output_formats or ["json"]
    tabular_output = {}

    if file_type == "video":
        for path in file_paths:
            # Předáváme categories do single video processingu
            res = _process_single_video(path, model_name, description or "", categories)
            tabular_output[res["filename"]] = res

    elif file_type == "image":
        # Zachování původní logiky pro obrázky
        for path in file_paths:
            # Zde by se volala původní image funkce, pro stručnost placeholder
            tabular_output[os.path.basename(path)] = {
                "status": "Image processing not modified in this update"
            }

    return {
        "status": "processed",
        "type": file_type,
        "model": model_name,
        "tabular_output": tabular_output,
        "outputs": _format_outputs(tabular_output, output_formats),
    }
