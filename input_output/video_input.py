import uuid, subprocess, threading
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from . import shared_state
from . import video_output
import traceback

app = Flask(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "temp_video_staging"
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.m4v', '.mkv'}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_mp4(input_path, output_path):
    command = ['ffmpeg', '-i', str(input_path), '-vcodec', 'libx264', '-acodec', 'aac', '-y', str(output_path)]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def stage_uploaded_video(file_object):
    if not file_object or file_object.filename == '':
        raise ValueError("No video file provided.")

    filename = secure_filename(file_object.filename)
    if not filename:
        raise ValueError("Invalid video filename.")
    name, ext = Path(filename).stem, Path(filename).suffix
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    unique_id = uuid.uuid4().hex[:8]
    temp_raw_path = UPLOAD_DIR / f"raw_{unique_id}{ext}"
    file_object.save(str(temp_raw_path))

    final_mp4_path = UPLOAD_DIR / f"{name}_{unique_id}.mp4"

    if ext.lower() == '.mp4':
        temp_raw_path.rename(final_mp4_path)
    else:
        convert_to_mp4(temp_raw_path, final_mp4_path)
        temp_raw_path.unlink()
    return str(final_mp4_path)

@app.route('/api/ingest_video', methods=['POST'])
def ingest_video():
    if 'video' not in request.files:
        return jsonify({"error": "Missing 'video' in form data"}), 400

    try:
        video_output.broadcast_results("analysing")
        staged_filepath = stage_uploaded_video(request.files['video'])
        print(f"Video ready at: {staged_filepath}")

        # Import and run the pipeline here to avoid circular imports at the top
        from run import run_pipeline
        thread = threading.Thread(target=run_pipeline, args=(staged_filepath,))
        thread.start()

        return jsonify({"status": "success", "file_path": staged_filepath}), 200

    except ValueError as e:
        video_output.broadcast_results("error")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        video_output.broadcast_results("error")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/api/get_analysis', methods=['GET'])
def get_analysis():
    return jsonify(shared_state.latest_analysis)

if not hasattr(shared_state, 'latest_analysis'):
    shared_state.latest_analysis = {"status": "waiting", "behavior": "", "description": ""}