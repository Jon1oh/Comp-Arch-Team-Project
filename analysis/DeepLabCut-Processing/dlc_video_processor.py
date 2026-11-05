
"""
dlc_video_processor.py

DeepLabCut 3.0.2
SuperAnimal Quadruped
HRNet-W32
Faster R-CNN MobileNet V3 Large FPN

Input requirements:
- Maximum duration: 10 seconds
- 30 FPS
- Maximum resolution: 1280 x 720

Processing:
- No FPS conversion
- No frame duplication
- Resize only if above 720p
- CPU inference
- No labeled video
- JSON output only
"""

# =========================================================
# 1. IMPORTS
# =========================================================

from pathlib import Path
import json
import math
import logging
import time
import shutil

import cv2
import pandas as pd
import torch
import deeplabcut


# =========================================================
# 2. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
PREPARED_DIR = OUTPUT_DIR / "prepared_videos"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

MAX_DURATION = 10.0
REQUIRED_FPS = 30.0

MAX_WIDTH = 1280
MAX_HEIGHT = 720

SUPERANIMAL_NAME = "superanimal_quadruped"
MODEL_NAME = "hrnet_w32"
DETECTOR_NAME = "fasterrcnn_mobilenet_v3_large_fpn"

VIDEO_ADAPT = False

POSE_BATCH_SIZE = 2
DETECTOR_BATCH_SIZE = 1

DEVICE = "cpu"

MAX_INDIVIDUALS = 1

CONFIDENCE_THRESHOLD = 0.60
FPS_TOLERANCE = 0.05

KEEP_H5 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# 3. VALIDATE UPLOADED VIDEO
# =========================================================


def validate_video(video_path):

    video_path = Path(video_path).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to open uploaded video."
        )

    try:

        fps = capture.get(cv2.CAP_PROP_FPS)

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if (
            not math.isfinite(fps)
            or fps <= 0
            or frame_count <= 0
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                "Invalid video metadata."
            )

        duration = frame_count / fps

        if duration > MAX_DURATION + (1 / fps):

            raise ValueError(
                f"Video exceeds {MAX_DURATION} seconds. "
                f"Detected duration: {duration:.2f} seconds."
            )

        logger.info("Video validation successful.")

        logger.info("Original FPS: %.2f", fps)

        logger.info(
            "Original resolution: %s x %s",
            width,
            height
        )

        logger.info(
            "Duration: %.2f seconds",
            duration
        )

        if fps > REQUIRED_FPS:
            logger.info(
                "FPS exceeds 30. Conversion to 30 FPS required."
            )

        elif fps < REQUIRED_FPS:
            logger.info(
                "FPS below 30. Original FPS will be retained."
            )

        else:
            logger.info(
                "Video already at 30 FPS. No FPS conversion needed."
            )

        return {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height
        }

    finally:
        capture.release()


# =========================================================
# 4. RESIZE VIDEO ONLY IF ABOVE 720P
# =========================================================


def prepare_video(video_path):

    metadata = validate_video(video_path)

    video_path = metadata["path"]

    original_fps = metadata["fps"]

    width = metadata["width"]
    height = metadata["height"]

    # Only convert FPS when original FPS exceeds 30.
    output_fps = (
        REQUIRED_FPS
        if original_fps > REQUIRED_FPS
        else original_fps
    )

    # Resize only when resolution exceeds 720p.
    needs_resize = (
        width > MAX_WIDTH
        or height > MAX_HEIGHT
    )

    needs_fps_conversion = (
        original_fps > REQUIRED_FPS
    )

    # No conversion needed.
    if not needs_resize and not needs_fps_conversion:

        logger.info(
            "No video conversion required."
        )

        return video_path

    PREPARED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PREPARED_DIR /
        f"{video_path.stem}_prepared.mp4"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Unable to open video for preparation."
        )

    writer = None

    try:

        # Calculate output dimensions.
        scale = min(
            1.0,
            MAX_WIDTH / width,
            MAX_HEIGHT / height
        )

        output_width = max(
            2,
            int(width * scale) // 2 * 2
        )

        output_height = max(
            2,
            int(height * scale) // 2 * 2
        )

        writer = cv2.VideoWriter(

            str(output_path),

            cv2.VideoWriter_fourcc(*"mp4v"),

            output_fps,

            (output_width, output_height)

        )

        if not writer.isOpened():

            raise RuntimeError(
                "Unable to create prepared video."
            )

        # Determine which original frames to retain.
        if needs_fps_conversion:

            output_frame_count = math.ceil(
                metadata["frame_count"]
                * output_fps
                / original_fps
            )

            selected_frames = {

                round(
                    index * original_fps / output_fps
                )

                for index in range(output_frame_count)

            }

        else:

            selected_frames = None

        frame_index = 0
        written_frames = 0

        while True:

            success, frame = capture.read()

            if not success:
                break

            # Retain every frame below or at 30 FPS.
            # For higher FPS, retain selected frames only.
            if (
                selected_frames is not None
                and frame_index not in selected_frames
            ):

                frame_index += 1
                continue

            # Resize only when necessary.
            if needs_resize:

                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA
                )

            writer.write(frame)

            written_frames += 1
            frame_index += 1

    finally:

        capture.release()

        if writer is not None:
            writer.release()

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
        or written_frames == 0
    ):

        raise RuntimeError(
            "Prepared video was not generated correctly."
        )

    logger.info(
        "Video preparation completed."
    )

    logger.info(
        "Original FPS: %.2f",
        original_fps
    )

    logger.info(
        "Output FPS: %.2f",
        output_fps
    )

    logger.info(
        "Original frames: %s",
        metadata["frame_count"]
    )

    logger.info(
        "Output frames: %s",
        written_frames
    )

    logger.info(
        "Output resolution: %s x %s",
        output_width,
        output_height
    )

    logger.info(
        "Prepared video: %s",
        output_path
    )

    return output_path


# =========================================================
# 5. RUN DEEPLABCUT SUPERANIMAL INFERENCE
# =========================================================

def run_inference(
    video_path,
    output_folder,
    create_visual=False
):

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Starting SuperAnimal inference."
    )

    logger.info(
        "Selected device: %s",
        DEVICE
    )

    start_time = time.perf_counter()

    try:

        deeplabcut.video_inference_superanimal(

            videos=[str(video_path)],

            superanimal_name=SUPERANIMAL_NAME,

            model_name=MODEL_NAME,

            detector_name=DETECTOR_NAME,

            video_adapt=VIDEO_ADAPT,

            batch_size=POSE_BATCH_SIZE,

            detector_batch_size=DETECTOR_BATCH_SIZE,

            device=DEVICE,

            max_individuals=MAX_INDIVIDUALS,

            create_labeled_video=create_visual,

            plot_bboxes=False,

            dest_folder=str(output_folder)

        )

    except Exception as exc:

        logger.exception(
            "DeepLabCut inference failed."
        )

        raise RuntimeError(
            f"DLC inference failed: {exc}"
        ) from exc

    inference_time = (
        time.perf_counter() - start_time
    )

    matching_files = list(
        output_folder.glob("*.h5")
    )

    if not matching_files:

        raise FileNotFoundError(
            "No HDF5 prediction file was generated. "
            "Check the DLC inference output."
        )

    h5_path = max(
        matching_files,
        key=lambda path: path.stat().st_mtime
    )

    logger.info(
        "Inference completed in %.2f seconds.",
        inference_time
    )

    return h5_path, inference_time


# =========================================================
# 6. LOAD DEEPLABCUT PREDICTIONS
# =========================================================

def load_predictions(h5_path):

    try:
        dataframe = pd.read_hdf(h5_path)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read HDF5 predictions: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError(
            "Prediction dataframe is empty."
        )

    if not isinstance(
        dataframe.columns,
        pd.MultiIndex
    ):
        raise ValueError(
            "Unexpected DLC prediction format. "
            "Expected MultiIndex columns."
        )

    columns = dataframe.columns

    coordinate_level = None

    for level in range(columns.nlevels):

        values = {
            str(value).lower()
            for value in columns.get_level_values(level)
        }

        if {"x", "y", "likelihood"}.issubset(values):
            coordinate_level = level
            break

    if coordinate_level is None:
        raise ValueError(
            "Could not identify x, y and likelihood columns."
        )

    bodypart_level = None

    for level, name in enumerate(columns.names):

        if name and "bodypart" in str(name).lower():
            bodypart_level = level
            break

    if bodypart_level is None:

        candidates = [
            level
            for level in range(columns.nlevels)
            if level != coordinate_level
        ]

        if not candidates:
            raise ValueError(
                "Could not identify body-part column level."
            )

        bodypart_level = max(
            candidates,
            key=lambda level: len(set(
                str(value)
                for value in columns.get_level_values(level)
            ))
        )

    bodyparts = list(dict.fromkeys(
        str(value)
        for value in columns.get_level_values(bodypart_level)
    ))

    logger.info(
        "Detected body parts: %s",
        bodyparts
    )

    return (
        dataframe,
        bodyparts,
        bodypart_level,
        coordinate_level
    )


# =========================================================
# 7. SAFE NUMBER CONVERSION
# =========================================================

def safe_number(value):

    try:

        number = float(value)

        if not math.isfinite(number):
            return None

        return round(number, 3)

    except (TypeError, ValueError, OverflowError):

        return None


# =========================================================
# 8. CONVERT PREDICTIONS TO JSON
# =========================================================

def convert_predictions_to_json(
    h5_path,
    video_path,
    json_path
):

    (
        dataframe,
        bodyparts,
        bodypart_level,
        coordinate_level
    ) = load_predictions(h5_path)

    predictions = {}

    # Extract coordinates for every detected body part.
    for bodypart in bodyparts:

        predictions[bodypart] = {}

        for coordinate in (
            "x",
            "y",
            "likelihood"
        ):

            matching_columns = [

                column

                for column in dataframe.columns

                if (
                    str(column[bodypart_level]) == bodypart

                    and str(
                        column[coordinate_level]
                    ).lower() == coordinate
                )

            ]

            if not matching_columns:
                continue

            predictions[bodypart][coordinate] = (
                pd.to_numeric(
                    dataframe[matching_columns[0]],
                    errors="coerce"
                ).to_numpy()
            )

    # Read video metadata.
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to read video metadata."
        )

    try:

        fps = capture.get(cv2.CAP_PROP_FPS)

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video FPS.")

    if len(dataframe) != frame_count:
        raise ValueError(
            f"Frame count mismatch: "
            f"video={frame_count}, "
            f"predictions={len(dataframe)}."
        )

    frames = []

    # Convert every prediction frame into JSON-compatible data.
    for frame_index in range(len(dataframe)):

        keypoints = {}

        for bodypart in bodyparts:

            values = predictions[bodypart]

            if not all(
                coordinate in values
                for coordinate in (
                    "x",
                    "y",
                    "likelihood"
                )
            ):

                keypoints[bodypart] = {
                    "x": None,
                    "y": None,
                    "likelihood": None,
                    "valid": False
                }

                continue

            x = safe_number(
                values["x"][frame_index]
            )

            y = safe_number(
                values["y"][frame_index]
            )

            likelihood = safe_number(
                values["likelihood"][frame_index]
            )

            valid = (
                x is not None
                and y is not None
                and likelihood is not None
                and 0 <= likelihood <= 1
                and likelihood >= CONFIDENCE_THRESHOLD
            )

            keypoints[bodypart] = {
                "x": x,
                "y": y,
                "likelihood": likelihood,
                "valid": valid
            }

        frames.append({

            "frame": frame_index,

            "timestamp_seconds": round(
                frame_index / fps,
                4
            ),

            "keypoints": keypoints

        })

    # Assemble final dataset.
    dataset = {

        "metadata": {

            "original_video": Path(
                video_path
            ).stem,

            "fps": fps,

            "duration_seconds": round(
                frame_count / fps,
                3
            ),

            "frame_count": frame_count,

            "processed_frame_count": len(frames),

            "width": width,

            "height": height,

            "model": SUPERANIMAL_NAME,

            "pose_model": MODEL_NAME,

            "detector": DETECTOR_NAME,

            "device": DEVICE,

            "confidence_threshold": CONFIDENCE_THRESHOLD

        },

        "keypoint_names": bodyparts,

        "frames": frames

    }

    # Export JSON.
    json_path = Path(json_path)

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        with json_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                dataset,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

    except (OSError, TypeError, ValueError) as exc:

        raise RuntimeError(
            f"JSON export failed: {exc}"
        ) from exc

    logger.info(
        "JSON saved successfully: %s",
        json_path
    )

    return dataset


# =========================================================
# 9. COMPLETE VIDEO PROCESSING PIPELINE
# =========================================================

def process_video(video_path):

    total_start = time.perf_counter()

    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREPARED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    import uuid

    video_prediction_folder = (
        PREDICTION_DIR /
        f"{video_path.stem}_{uuid.uuid4().hex[:8]}"
    )

    video_prediction_folder.mkdir(
        parents=True,
        exist_ok=False
    )

    # Step 1: Validate and prepare video.
    preparation_start = time.perf_counter()

    prepared_video = prepare_video(
        video_path
    )

    preparation_time = (
        time.perf_counter() - preparation_start
    )

    # Step 2: Run DeepLabCut inference.
    h5_path, inference_time = run_inference(

        video_path=prepared_video,

        output_folder=video_prediction_folder,

        create_visual=False

    )

    # Step 3: Convert predictions into JSON.
    export_start = time.perf_counter()

    json_path = (
        OUTPUT_DIR /
        f"{video_path.stem}_keypoints.json"
    )

    dataset = convert_predictions_to_json(

        h5_path,

        prepared_video,

        json_path

    )

    export_time = (
        time.perf_counter() - export_start
    )

    total_time = (
        time.perf_counter() - total_start
    )

    # Add processing statistics.
    dataset["processing_time"] = {

        "preparation_seconds": round(
            preparation_time, 2
        ),

        "inference_seconds": round(
            inference_time, 2
        ),

        "export_seconds": round(
            export_time, 2
        ),

        "total_seconds": round(
            total_time, 2
        )

    }

    # Rewrite JSON with processing statistics.
    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

    # Delete intermediate HDF5 only after successful export.
    if not KEEP_H5:

        h5_path.unlink(
            missing_ok=True
        )

    # Display results.
    print("\n========== RESULTS ==========")

    print("Video:", video_path.name)

    print("Frames:", len(dataset["frames"]))

    print(
        "Resolution:",
        dataset["metadata"]["width"],
        "x",
        dataset["metadata"]["height"]
    )

    print(
        "FPS:",
        dataset["metadata"]["fps"]
    )

    print("\nProcessing times:")

    for name, value in dataset["processing_time"].items():

        print(
            f"{name}: {value} seconds"
        )

    print("\nJSON output:", json_path)

    print("=============================")

    return dataset


# =========================================================
# 10. MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":

    VIDEO_PATH = BASE_DIR / "test3_cat_grooming.mp4"

    print(
        "DeepLabCut version:",
        deeplabcut.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "Selected device:",
        DEVICE
    )

    try:

        result = process_video(
            VIDEO_PATH
        )

        print(
            "\nCat video processing completed!"
        )

    except Exception:

        logger.exception(
            "Video processing failed."
        )

        raise SystemExit(1)
