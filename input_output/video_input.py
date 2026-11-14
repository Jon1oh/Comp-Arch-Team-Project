import os, uuid, subprocess, video_output
from . import shared_state
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_DIR = "temp_video_staging"
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.m4v', '.mkv'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Upload Cat Video</title></head>
<body style="font-family: Arial, sans-serif; padding: 2rem;">
    <h2>Cat Behavior AI: Video Upload</h2>
    <form action="/api/ingest_video" method="POST" enctype="multipart/form-data">
        <input type="file" name="video" accept="video/mp4,video/quicktime,video/x-msvideo,video/*" required>
        <br><br>
        <button type="submit" style="padding: 10px 20px;">Upload to Pipeline</button>
    </form>
</body>
</html>
"""

def convert_to_mp4(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-y', output_path 
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def stage_uploaded_video(file_object):
    if not file_object or file_object.filename == '':
        raise ValueError("No video file provided.")

    filename = secure_filename(file_object.filename)
    name, ext = os.path.splitext(filename)
    
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    unique_id = uuid.uuid4().hex[:8]
    temp_raw_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"raw_{unique_id}{ext}"))
    file_object.save(temp_raw_path)

    final_mp4_path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{name}_{unique_id}.mp4"))
    
    if ext.lower() == '.mp4':
        os.rename(temp_raw_path, final_mp4_path)
    else:
        print(f"Converting {ext} to .mp4...")
        convert_to_mp4(temp_raw_path, final_mp4_path)
        os.remove(temp_raw_path) 
        
    return final_mp4_path

@app.route('/', methods=['GET'])
def index():
    return render_template_string(UPLOAD_PAGE)

@app.route('/api/ingest_video', methods=['POST'])
def ingest_video():
    if 'video' not in request.files:
        return jsonify({"error": "Missing 'video' in form data"}), 400
        
    try:
        # 1. Instantly trigger the TinyScreen to show "ANALYSING"
        video_output.broadcast_results("analysing")
        
        # 2. Save and convert the video
        staged_filepath = stage_uploaded_video(request.files['video'])
        print(f"Video ready at: {staged_filepath}")
        
        return jsonify({"status": "success", "file_path": staged_filepath}), 200
        
    except subprocess.CalledProcessError:
        return jsonify({"error": "Failed to convert video format on server."}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/api/get_analysis', methods=['GET'])
def get_analysis():
    # Allow the iOS app to poll the shared state
    return jsonify(shared_state.latest_analysis)

if __name__ == "__main__":
    # Ensure shared_state has default values on startup
    if not hasattr(shared_state, 'latest_analysis'):
        shared_state.latest_analysis = {"status": "waiting", "behavior": "", "description": ""}
        
    app.run(host='0.0.0.0', port=5001, debug=True)