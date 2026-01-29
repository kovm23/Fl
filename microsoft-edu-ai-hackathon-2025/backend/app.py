from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from services.processing import process_files

app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    'text': {'pdf', 'txt'},
    'image': {'png', 'jpg', 'jpeg'},
    'video': {'mp4', 'avi', 'mov', 'mkv'},
    'zip': {'zip'}
}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='ok'), 200
    
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files uploaded'}), 400
        
    # Detekce typu
    file_types_detected = set()
    for f in files:
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        for type_name, allowed_exts in ALLOWED_EXTENSIONS.items():
            if ext in allowed_exts:
                file_types_detected.add(type_name)
                break
                
    if len(file_types_detected) != 1:
        return jsonify({'error': 'Single file type allowed per upload'}), 400
        
    file_type = file_types_detected.pop()
    
    # Save files
    saved_files = []
    for f in files:
        if allowed_file(f.filename, ALLOWED_EXTENSIONS[file_type]):
            filename = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(save_path)
            saved_files.append(filename)

    # Parametry z formuláře
    output_formats = request.form.get('output_formats', '')
    if output_formats:
        output_formats = [fmt.strip() for fmt in output_formats.split(',') if fmt.strip()]
    
    description = request.form.get('description', None)
    
    # --- NOVÉ: Načtení vybraného modelu ---
    # Default je 'gpt-4o', pokud frontend nic nepošle
    selected_model = request.form.get('model', 'gpt-4o') 

    # Process
    try:
        processing_result = process_files(
            [os.path.join(UPLOAD_FOLDER, fname) for fname in saved_files],
            file_type,
            output_formats=output_formats,
            description=description,
            model_name=selected_model  # Předáváme dál
        )
        return jsonify({
            'message': 'Success', 
            'model_used': selected_model,
            'processing': processing_result
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)