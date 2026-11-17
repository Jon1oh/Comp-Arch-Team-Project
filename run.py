import sys, argparse, logging
from pathlib import Path

# Add directories so Python can locate the modules
PROJECT_ROOT = Path(__file__).resolve().parent
DLC_DIR = PROJECT_ROOT / "analysis" / "DeepLabCut-Processing"
INPUT_OUTPUT_DIR = PROJECT_ROOT / "input_output"

sys.path.insert(0, str(DLC_DIR))
sys.path.insert(0, str(INPUT_OUTPUT_DIR))

# Import your pipeline components
from analysis.dlc_video_processor import process_video
from logic.movement_summary import analyze_dataset, to_ai_evidence
from ai.ai import analyze_data
from input_output import video_output

def run_pipeline(video_path):
    try:
        print("\n========== CAT BEHAVIOUR ANALYSER ==========\n")
        
        # Step 0: Notify displays
        video_output.broadcast_results("analysing")

        # Step 1: DeepLabCut (Raw Coordinates)
        print("[1/4] Processing video with DeepLabCut...")
        raw_keypoints = process_video(video_path)

        # Step 2: Movement Summary (Behavior Logic)
        print("[2/4] Generating movement logic summary...")
        full_summary = analyze_dataset(raw_keypoints)
        ai_evidence = to_ai_evidence(full_summary)

        # Step 3: Gemini AI Interpretation
        print("\n[3/4] Analysing animal behaviour...")
        ai_result = analyze_data(ai_evidence)
        if ai_result is None:
            raise RuntimeError("AI analysis failed.")

        print("\n[4/4] Analysis complete! Here is the output:")
        print("Behaviour:", ai_result.get("behaviour"))
        print("Explanation:", ai_result.get("explanation"))

        # Step 4: Route outputs (TinyScreen gets detail, App gets summary)
        video_output.broadcast_results("complete", ai_result)
        
        print("\n========== PIPELINE FINISHED ==========\n")
        
        return ai_result

    except Exception:
        logging.exception("Pipeline failed")
        video_output.broadcast_results("error")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat Behaviour Analysis System")
    parser.add_argument("--video", type=Path, help="Path to the input video")
    args = parser.parse_args()

    if args.video:
        if not args.video.is_file():
            parser.error(f"Video not found: {args.video}")
        logging.basicConfig(level=logging.INFO)
        run_pipeline(args.video)
    else:
        from input_output.video_input import app
        print("\nStarting macOS Flask Server on http://192.168.1.79:5001 ...")
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)