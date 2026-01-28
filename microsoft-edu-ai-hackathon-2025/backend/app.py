from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from services.processing import process_files

app = Flask(__name__)

# Allowed extensions for each type
ALLOWED_EXTENSIONS = {
    "text": {"pdf", "txt"},
    "image": {"png", "jpg", "jpeg"},
    "video": {"mp4", "avi", "mov", "mkv"},
    "zip": {"zip"},
}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Helper to check allowed file
def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_exts


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "No files uploaded"}), 400
    # Infer file type from extensions
    file_types_detected = set()
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        for type_name, allowed_exts in ALLOWED_EXTENSIONS.items():
            if ext in allowed_exts:
                file_types_detected.add(type_name)
                break
    if len(file_types_detected) == 0:
        return jsonify({"error": "No valid file types detected"}), 400
    if len(file_types_detected) > 1:
        return jsonify({"error": "Multiple file types detected in one upload"}), 400
    file_type = file_types_detected.pop()
    allowed_exts = ALLOWED_EXTENSIONS[file_type]
    # Validate all files
    for f in files:
        if not allowed_file(f.filename, allowed_exts):
            return (
                jsonify(
                    {"error": f"File {f.filename} is not a valid {file_type} file"}
                ),
                400,
            )
    saved_files = []
    for f in files:
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)
        saved_files.append(filename)
    # Parse output_formats (checkboxes)
    output_formats = request.form.get("output_formats", "")
    if output_formats:
        output_formats = [
            fmt.strip() for fmt in output_formats.split(",") if fmt.strip()
        ]
    else:
        output_formats = []
    # Parse optional description
    description = request.form.get("description", None)
    # Parse model provider (azure/local)
    model_provider = request.form.get("model_provider", "azure")
    # Process files using modular service
    processing_result = process_files(
        [os.path.join(UPLOAD_FOLDER, fname) for fname in saved_files],
        file_type,
        output_formats=output_formats,
        description=description,
        model_provider=model_provider,
    )
    return (
        jsonify(
            {
                "message": f"{len(saved_files)} {file_type} files uploaded",
                "files": saved_files,
                "processing": processing_result,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True)
