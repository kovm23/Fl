from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from services.processing import process_files
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt", "md", "csv"},
    "image": {"png", "jpg", "jpeg"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts


@app.route("/uploads/<path:filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


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

    # NOVÉ: Kategorie pro RuleKit
    categories = request.form.get("categories", "")

    try:
        result = process_files(
            saved_paths,
            detected_type,
            output_formats=formats,
            description=desc,
            model_name=model,
            categories=categories,  # Předáváme dál
        )
        return jsonify({"message": "OK", "processing": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
