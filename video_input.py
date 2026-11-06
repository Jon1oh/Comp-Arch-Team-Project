import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_DIR = "temp_video_staging"
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# The HTML interface
UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Upload Cat Video</title></head>
<body style="font-family: Arial, sans-serif; padding: 2rem;">
    <h2>Cat Behavior AI: Video Upload</h2>
    <form action="/api/ingest_video" method="POST" enctype="multipart/form-data">
        <input type="file" name="video" accept="video/mp4,video/x-m4v,video/*" required>
        <br><br>
        <button type="submit" style="padding: 10px 20px;">Upload to Pipeline</button>
    </form>
</body>
</html>
"""

def stage_uploaded_video(file_object):
    if not file_object or file_object.filename == '':
        raise ValueError("No video file provided.")

    filename = secure_filename(file_object.filename)
    name, ext = os.path.splitext(filename)
    
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    unique_id = uuid.uuid4().hex[:8]
    unique_filename = f"{name}_{unique_id}{ext}"
    safe_path = os.path.abspath(os.path.join(UPLOAD_DIR, unique_filename))
    
    file_object.save(safe_path)
    return safe_path

@app.route('/', methods=['GET'])
def index():
    # Renders the HTML page at http://127.0.0.1:5001/
    return render_template_string(UPLOAD_PAGE)

@app.route('/api/ingest_video', methods=['POST'])
def ingest_video():
    # Receives the video from either the HTML page or the iOS app
    if 'video' not in request.files:
        return jsonify({"error": "Missing 'video' in form data"}), 400
        
    try:
        staged_filepath = stage_uploaded_video(request.files['video'])
        print(f"[IO Manager] Video ready for Compute Team at: {staged_filepath}")
        return jsonify({"status": "success", "file_path": staged_filepath}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)