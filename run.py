import sys, argparse, logging
from pathlib import Path
from analysis.dlc_video_processor import process_video
from ai.ai import analyze_data
from input_output import video_output

PROJECT_ROOT = Path(__file__).resolve().parent
DLC_DIR = PROJECT_ROOT / "analysis" / "DeepLabCut-Processing"
INPUT_OUTPUT_DIR = PROJECT_ROOT / "input_output"

# Add directories so Python can locate the modules
sys.path.insert(0, str(DLC_DIR))
sys.path.insert(0, str(INPUT_OUTPUT_DIR))

def run_pipeline(video_path):
    try:
        print("\n========== CAT BEHAVIOUR ANALYSER ==========\n")
        
        # Step 1: Notify TinyScreen that analysis has started
        video_output.broadcast_results("analysing")

        # Step 2: Process video using DeepLabCut
        print("[1/3] Processing video with DeepLabCut...")

        keypoint_data = process_video(video_path)

        # Step 3: Send keypoint data for AI analysis
        print("\n[2/3] Analysing animal behaviour...")
        
        ai_result = analyze_data(keypoint_data)
        if ai_result is None:
            raise RuntimeError("AI analysis failed.")

        # Step 4: Display results
        print("\n[3/3] Analysis complete! Here is the output:")
        print("Behaviour:", ai_result["behaviour"])
        print("Body Language:", ai_result["body_language_meaning"])
        print("Explanation:", ai_result["explanation"])
        print("Confidence:", ai_result["confidence"])
        print("Data Quality:", ai_result["data_quality"])

        # Step 5: Send results to output component
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

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")

    logging.basicConfig(level=logging.INFO)
    run_pipeline(args.video)
