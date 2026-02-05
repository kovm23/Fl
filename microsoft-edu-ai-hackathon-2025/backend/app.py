from flask import Flask, request, jsonify, send_from_directory
import os
import zipfile
import pandas as pd
import random  # Pouze pro simulaci RuleKitu, než ho napojíš
from werkzeug.utils import secure_filename
from services.processing import process_files
from flask_cors import CORS

app = Flask(__name__)
# Povolíme CORS pro tvou novou doménu i localhost
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt", "md", "csv"},
    "image": {"png", "jpg", "jpeg"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"  # Složka pro MediaEval data
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)


# ==========================================
# RULEKIT MOCK (Simulace logiky pro BP)
# ==========================================
class RuleKitPredictor:
    def __init__(self):
        self.current_model_name = "Default Base Model"
        self.is_trained = False
        self.rules = []

    def train(self, zip_path):
        """
        1. Rozbalí ZIP
        2. Najde CSV/Excel
        3. Simuluje extrakci rysů a trénink
        """
        extract_path = os.path.join("dataset", "temp_extract")
        os.makedirs(extract_path, exist_ok=True)

        # Rozbalení ZIPu
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Hledání CSV souboru s daty
        data_file = None
        media_count = 0

        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith(".csv") or file.endswith(".xlsx"):
                    data_file = os.path.join(root, file)
                if file.lower().endswith((".png", ".jpg", ".mp4", ".avi")):
                    media_count += 1

        if not data_file:
            return {
                "status": "error",
                "message": "V ZIP souboru chybí CSV/Excel soubor s anotacemi!",
            }

        # Zde by reálně proběhl trénink RuleKitu (načtení CSV, extrakce features z obrázků...)
        # SIMULACE:
        self.current_model_name = f"Custom Model (from {os.path.basename(zip_path)})"
        self.is_trained = True

        return {
            "status": "success",
            "dataset_file": os.path.basename(data_file),
            "media_files_found": media_count,
            "rules_generated": 156,
            "accuracy": 0.89,
        }

    def predict(self, features_text):
        # ... (zbytek funkce predict zůstává stejný) ...
        # Jen upravíme vysvětlení, aby bylo vidět, že jede nový model
        score = 0.5
        features_text = features_text.lower()
        reasons = []

        if "red" in features_text:
            score += 0.2
            reasons.append("Red color (Custom Rule #1)")
        if "dark" in features_text:
            score -= 0.2
            reasons.append("Dark scene (Custom Rule #2)")

        return {
            "score": max(0.1, min(0.9, score)),
            "explanation": f"Based on '{self.current_model_name}': {', '.join(reasons)}",
            "features_used": features_text[:50] + "...",
        }


predictor = RuleKitPredictor()

# ==========================================
# ENDPOINTS
# ==========================================


def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts


@app.route("/uploads/<path:filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/train", methods=["POST"])
def train_model():
    """Endpoint pro nahrání ZIP datasetu"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".zip"):
        filename = secure_filename(file.filename)
        save_path = os.path.join(DATASET_FOLDER, filename)
        file.save(save_path)

        try:
            result = predictor.train(save_path)
            return jsonify({"message": "Training completed", "details": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Only ZIP files are allowed"}), 400


@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400

    # Detekce typu
    detected_type = None
    for f in files:
        ext = f.filename.split(".")[-1].lower()
        for t, exts in ALLOWED_EXTENSIONS.items():
            if ext in exts:
                if detected_type and detected_type != t:
                    return jsonify({"error": "Mixed types not allowed"}), 400
                detected_type = t

    if not detected_type:
        return jsonify({"error": "Unknown file type"}), 400

    saved_paths = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, fname)
        f.save(path)
        saved_paths.append(path)

    # Parametry
    formats = request.form.get("output_formats", "").split(",")
    desc = request.form.get("description", "")
    model = request.form.get("model", "gpt-4o")

    # Ignorujeme "categories" pro klasifikaci, použijeme je jako kontext pro RuleKit
    categories = request.form.get("categories", "")

    try:
        # 1. Zpracování souboru (LLM / Video processing)
        # Toto získá "Features" (co je na videu)
        processing_result = process_files(
            saved_paths,
            detected_type,
            output_formats=formats,
            description=desc,
            model_name=model,
            categories=categories,
        )

        # 2. RuleKit Predikce
        # Získáme textový popis z výsledku (předpokládáme, že process_files vrací dict s popisem)
        # Zkusíme najít textový výstup v různých polích
        features_text = ""
        if isinstance(processing_result, dict):
            features_text = (
                str(processing_result.get("description", ""))
                or str(processing_result.get("transcription", ""))
                or str(processing_result.get("captions", ""))
            )

        # Pokud nemáme text, použijeme název souboru jako fallback features
        if not features_text:
            features_text = (
                f"File analysis of {saved_paths[0]} with categories: {categories}"
            )

        # Spustíme RuleKit predikci
        prediction = predictor.predict(features_text)

        # 3. Sloučení výsledků
        final_response = {
            **processing_result,  # Původní data z LLM
            "memorability_score": prediction["score"],
            "memorability_explanation": prediction["explanation"],
            "rulekit_features": prediction["features_used"],
        }

        return jsonify({"message": "OK", "processing": final_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
