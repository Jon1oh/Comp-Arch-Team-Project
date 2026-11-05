
"""
test_cat.py

DEMONSTRATION VIDEO GENERATOR

Purpose:
- Generate an annotated video showing detected cat keypoints.
- Reuse video preparation and model settings from dlc_video_processor.py.
- Generate an MP4 demonstration video only.

Does NOT:
- Export JSON.
- Generate a separate keypoint dataset.
"""

from pathlib import Path
import time
import logging

import deeplabcut
import torch

from dlc_video_processor import (
    BASE_DIR,
    OUTPUT_DIR,
    SUPERANIMAL_NAME,
    MODEL_NAME,
    DETECTOR_NAME,
    VIDEO_ADAPT,
    POSE_BATCH_SIZE,
    DETECTOR_BATCH_SIZE,
    DEVICE,
    MAX_INDIVIDUALS,
    prepare_video,
    validate_video
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

VIDEO_PATH = BASE_DIR / "test3_cat_grooming.mp4"

DEMO_DIR = OUTPUT_DIR / "demonstrations"

DEMO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# 2. GENERATE ANNOTATED VIDEO
# =========================================================

def generate_demo_video(video_path):

    video_path = Path(video_path).resolve()

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Validate uploaded video.
    metadata = validate_video(video_path)

    print("\nOriginal video information:")

    print("FPS:", metadata["fps"])

    print("Duration:", metadata["duration"])

    print(
        "Resolution:",
        metadata["width"],
        "x",
        metadata["height"]
    )

    # Prepare video using shared production settings.
    prepared_video = prepare_video(video_path)

    print("\nPrepared video:", prepared_video)

    # Create a unique output directory for this run.
    run_folder = (
        DEMO_DIR /
        f"{video_path.stem}_{time.time_ns()}"
    )

    run_folder.mkdir(
        parents=True,
        exist_ok=False
    )

    # Run inference and generate annotated video.
    start_time = time.perf_counter()

    print("\nGenerating annotated demonstration video...")

    try:

        deeplabcut.video_inference_superanimal(

            videos=[str(prepared_video)],

            superanimal_name=SUPERANIMAL_NAME,

            model_name=MODEL_NAME,

            detector_name=DETECTOR_NAME,

            video_adapt=VIDEO_ADAPT,

            batch_size=POSE_BATCH_SIZE,

            detector_batch_size=DETECTOR_BATCH_SIZE,

            device=DEVICE,

            max_individuals=MAX_INDIVIDUALS,

            create_labeled_video=True,

            plot_bboxes=False,

            dest_folder=str(run_folder)

        )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        # Locate annotated video only.
        generated_videos = [

            path for path in run_folder.rglob("*")

            if path.suffix.lower() in (
                ".mp4",
                ".avi",
                ".mov"
            )

        ]

        if not generated_videos:

            raise FileNotFoundError(
                "DLC did not generate an annotated video."
            )

        latest_video = max(
            generated_videos,
            key=lambda path: path.stat().st_mtime
        )

        # Move/copy final demonstration video.
        final_video = (
            DEMO_DIR /
            f"{video_path.stem}_annotated.mp4"
        )

        import shutil

        shutil.copy2(
            latest_video,
            final_video
        )

        # Remove temporary DLC prediction files and folders.
        try:
            shutil.rmtree(run_folder)
            logger.info("Temporary folder removed: %s", run_folder)

        except OSError as exc:
            logger.warning(
                "Could not remove temporary folder: %s",
                exc
            )

        print("\n========== DEMONSTRATION RESULTS ==========")

        print("Input video:", video_path.name)

        print("Prepared video:", prepared_video.name)

        print("Annotated video:", final_video)

        print(
            "Processing time:",
            round(elapsed_time, 2),
            "seconds"
        )

        print("===========================================")

        return final_video

    except Exception:

        logger.exception(
            "Annotated video generation failed."
        )

        raise


# =========================================================
# 3. MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    print(
        "DeepLabCut version:",
        deeplabcut.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print("Selected device:", DEVICE)

    try:

        result = generate_demo_video(
            VIDEO_PATH
        )

        print(
            "\nAnnotated demonstration generated successfully!"
        )

    except Exception:

        logger.exception(
            "Demonstration processing failed."
        )

        raise SystemExit(1)