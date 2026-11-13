"""Convert DeepLabCut keypoints into conservative, measurable AI evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


SCHEMA_VERSION = "behavior-summary/1"
LANDMARK_SOURCES = {
    "torso_front": "back_base",
    "torso_middle": "back_middle",
    "torso_rear": "back_end",
    "nose": "nose",
    "tail_base": "tail_base",
    "tail_tip": "tail_end",
    "front_left_paw": "front_left_paw",
    "front_right_paw": "front_right_paw",
    "back_left_paw": "back_left_paw",
    "back_right_paw": "back_right_paw",
}
PAW_LANDMARKS = (
    "front_left_paw",
    "front_right_paw",
    "back_left_paw",
    "back_right_paw",
)
CANDIDATE_DESCRIPTIONS = {
    "body_displacement": (
        "Movement of the midpoint of available torso landmarks across the image; "
        "does not establish locomotion or intent."
    ),
    "head_relative_motion": (
        "Movement of the nose relative to the torso center; does not establish gaze or emotion."
    ),
    "tail_relative_motion": (
        "Movement of the tail tip relative to its base; does not establish a tail signal or emotion."
    ),
    "paw_relative_motion": (
        "Movement of detected paws relative to the torso; does not identify walking, play, or grooming."
    ),
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class PreprocessConfig:
    confidence_threshold: float = 0.60
    segment_seconds: float = 3.0
    minimum_duration_seconds: float = 1.0
    minimum_coverage: float = 0.50
    minimum_valid_pairs: int = 2
    maximum_gap_factor: float = 1.5
    body_motion_threshold_diagonals_per_second: float = 0.05
    relative_motion_threshold_diagonals_per_second: float = 0.05

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if _number(value) is None:
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.segment_seconds <= 0 or self.minimum_duration_seconds <= 0:
            raise ValueError("Window and minimum durations must be positive")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be greater than 0 and at most 1")
        if type(self.minimum_valid_pairs) is not int or self.minimum_valid_pairs < 1:
            raise ValueError("minimum_valid_pairs must be a positive integer")
        if self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be at least 1")
        if (self.body_motion_threshold_diagonals_per_second <= 0
                or self.relative_motion_threshold_diagonals_per_second <= 0):
            raise ValueError("Motion thresholds must be positive")


def _validate_dataset(dataset: Any) -> tuple[dict[str, Any], list[dict[str, Any]], float, float]:
    if not isinstance(dataset, dict):
        raise ValueError("DLC dataset must be a JSON object")
    metadata = dataset.get("metadata")
    frames = dataset.get("frames")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list")
    keypoint_names = dataset.get("keypoint_names", [])
    if not isinstance(keypoint_names, list) or any(not isinstance(name, str) for name in keypoint_names):
        raise ValueError("keypoint_names must be a list of strings")

    info: dict[str, Any] = {}
    for field in ("fps", "width", "height"):
        value = _number(metadata.get(field))
        if value is None or value <= 0:
            raise ValueError(f"metadata.{field} must be a positive finite number")
        info[field] = value
    info["diagonal_pixels"] = math.hypot(info["width"], info["height"])

    normalized: list[dict[str, Any]] = []
    previous_frame = -1
    previous_time = -1.0
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{position}] must be an object")
        frame_index = frame.get("frame", position)
        if type(frame_index) is not int or frame_index < 0 or frame_index <= previous_frame:
            raise ValueError("Frame indices must be non-negative and increasing")
        timestamp = _number(frame.get("timestamp_seconds", frame_index / info["fps"]))
        if timestamp is None or timestamp < 0 or timestamp <= previous_time:
            raise ValueError("Frame timestamps must be finite, non-negative and increasing")
        keypoints = frame.get("keypoints")
        if not isinstance(keypoints, dict) or any(not isinstance(name, str) for name in keypoints):
            raise ValueError(f"frames[{position}].keypoints must be an object")
        normalized.append({
            "frame": frame_index,
            "timestamp_seconds": timestamp,
            "keypoints": keypoints,
        })
        previous_frame, previous_time = frame_index, timestamp

    start = normalized[0]["timestamp_seconds"]
    inferred_end = normalized[-1]["timestamp_seconds"] + 1 / info["fps"]
    declared_duration = _number(metadata.get("duration_seconds"))
    end = max(inferred_end, declared_duration) if declared_duration and declared_duration > 0 else inferred_end
    return info, normalized, start, end


def _resolve_sources(
    dataset: dict[str, Any], overrides: dict[str, str | None] | None
) -> dict[str, str | None]:
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("landmark_map must be a dictionary of canonical names to source labels")
    overrides = overrides or {}
    unknown = set(overrides) - LANDMARK_SOURCES.keys()
    if unknown:
        raise ValueError(f"Unknown canonical landmarks: {', '.join(sorted(unknown))}")
    names = set(dataset.get("keypoint_names", []))
    for frame in dataset["frames"]:
        names.update(frame["keypoints"])
    result: dict[str, str | None] = {}
    for canonical, default_source in LANDMARK_SOURCES.items():
        source = overrides.get(canonical, default_source)
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError(f"Source for {canonical} must be a non-empty string or None")
        result[canonical] = source if source in names else None
    return result


def _valid_point(
    point: Any, width: float, height: float, threshold: float
) -> dict[str, float] | None:
    if not isinstance(point, dict) or point.get("valid", True) is not True:
        return None
    x, y, likelihood = (_number(point.get(key)) for key in ("x", "y", "likelihood"))
    if (x is None or y is None or likelihood is None
            or not 0 <= x < width or not 0 <= y < height
            or not threshold <= likelihood <= 1):
        return None
    return {"x": x, "y": y, "likelihood": likelihood}


def _midpoint(points: list[dict[str, float]]) -> dict[str, float]:
    return {
        "x": statistics.fmean(point["x"] for point in points),
        "y": statistics.fmean(point["y"] for point in points),
    }


def _distance(first: dict[str, float], second: dict[str, float]) -> float:
    return math.hypot(second["x"] - first["x"], second["y"] - first["y"])


def _relative(point: dict[str, float], anchor: dict[str, float]) -> dict[str, float]:
    return {"x": point["x"] - anchor["x"], "y": point["y"] - anchor["y"]}


def _coverage(count: int, total: int) -> float:
    return count / total if total else 0.0


def _quality(frames: list[dict[str, Any]], sources: dict[str, str | None]) -> dict[str, Any]:
    coverage = {
        canonical: 100 * _coverage(
            sum(canonical in frame["keypoints"] for frame in frames), len(frames)
        )
        for canonical in sources
    }
    average = statistics.fmean(coverage.values()) if coverage else 0.0
    return {
        "rating": "good" if average >= 80 else "limited" if average >= 50 else "poor",
        "average_landmark_coverage_percent": round(average, 2),
        "landmark_coverage_percent": {name: round(value, 2) for name, value in coverage.items()},
        "unavailable_landmarks": [name for name, source in sources.items() if source is None],
    }


def _motion(
    frames: list[dict[str, Any]],
    vectors: list[dict[str, float] | None],
    body_lengths: list[float | None],
    config: PreprocessConfig,
    threshold: float,
    *,
    minimum_landmarks: int = 1,
    measured_landmarks: set[str] | None = None,
) -> dict[str, Any]:
    valid_frames = sum(vector is not None for vector in vectors)
    scale_values = [value for value in body_lengths if value is not None]
    scale_coverage = _coverage(len(scale_values), len(frames))
    normalizer = frames[0]["diagonal_pixels"]
    speeds: list[float] = []
    for index in range(1, len(frames)):
        first, second = vectors[index - 1], vectors[index]
        dt = frames[index]["timestamp_seconds"] - frames[index - 1]["timestamp_seconds"]
        consecutive = frames[index]["frame"] == frames[index - 1]["frame"] + 1
        if first is None or second is None:
            continue
        if not consecutive or dt <= 0 or dt > config.maximum_gap_factor / frames[index]["fps"]:
            continue
        speeds.append(_distance(first, second) / dt / normalizer)

    median_speed = statistics.median(speeds) if speeds else None
    pair_coverage = _coverage(len(speeds), max(len(frames) - 1, 0))
    duration = (
        frames[-1]["timestamp_seconds"] - frames[0]["timestamp_seconds"]
        if len(frames) > 1 else 0.0
    )
    enough = (
        duration >= config.minimum_duration_seconds
        and _coverage(valid_frames, len(frames)) >= config.minimum_coverage
        and len(speeds) >= config.minimum_valid_pairs
        and len(measured_landmarks or {"one"}) >= minimum_landmarks
    )
    status = "insufficient_data" if not enough else (
        "detected" if median_speed is not None and median_speed >= threshold else "not_detected"
    )
    return {
        "status": status,
        "median_speed_image_diagonals_per_second": median_speed,
        "mean_speed_image_diagonals_per_second": statistics.fmean(speeds) if speeds else None,
        "pairs_over_threshold_percent": 100 * _coverage(sum(value >= threshold for value in speeds), len(speeds)),
        "threshold_image_diagonals_per_second": threshold,
        "valid_frame_percent": 100 * _coverage(valid_frames, len(frames)),
        "body_scale_valid_frame_percent": 100 * scale_coverage,
        "valid_pair_count": len(speeds),
        "valid_pair_percent": 100 * pair_coverage,
        "minimum_coverage_percent": 100 * config.minimum_coverage,
        "minimum_duration_seconds": config.minimum_duration_seconds,
        "landmarks_with_valid_pairs": sorted(measured_landmarks or []),
    }


def _prepare_frames(
    frames: list[dict[str, Any]],
    sources: dict[str, str | None],
    info: dict[str, Any],
    config: PreprocessConfig,
) -> list[dict[str, Any]]:
    result = []
    for frame in frames:
        points = {
            canonical: point
            for canonical, source in sources.items()
            if source is not None
            and (point := _valid_point(
                frame["keypoints"].get(source),
                info["width"],
                info["height"],
                config.confidence_threshold,
            )) is not None
        }
        torso = [points[name] for name in ("torso_front", "torso_middle", "torso_rear") if name in points]
        center = _midpoint(torso) if len(torso) >= 2 else None
        body_length = (
            _distance(points["torso_front"], points["torso_rear"])
            if "torso_front" in points and "torso_rear" in points else None
        )
        vectors: dict[str, dict[str, float] | None] = {
            "body": center,
            "head_relative": _relative(points["nose"], center) if center and "nose" in points else None,
            "tail_relative": (
                _relative(points["tail_tip"], points["tail_base"])
                if "tail_tip" in points and "tail_base" in points else None
            ),
        }
        for paw in PAW_LANDMARKS:
            vectors[paw] = _relative(points[paw], center) if center and paw in points else None
        result.append({
            "frame": frame["frame"],
            "timestamp_seconds": frame["timestamp_seconds"],
            "fps": info["fps"],
            "diagonal_pixels": info["diagonal_pixels"],
            "keypoints": points,
            "body_center": center,
            "body_length": body_length if body_length and body_length > 0 else None,
            "vectors": vectors,
        })
    return result


def _analyze_window(
    frames: list[dict[str, Any]],
    start: float,
    end: float,
    config: PreprocessConfig,
) -> dict[str, Any]:
    if not frames:
        empty_motion = {
            "status": "insufficient_data",
            "median_speed_image_diagonals_per_second": None,
            "valid_frame_percent": 0.0,
            "valid_pair_count": 0,
            "minimum_coverage_percent": 100 * config.minimum_coverage,
            "minimum_duration_seconds": config.minimum_duration_seconds,
            "landmarks_with_valid_pairs": [],
        }
        return {
            "start_seconds": start,
            "end_seconds": end,
            "frames_processed": 0,
            "measurements": {},
            "candidates": [
                {
                    "label": label,
                    "description": CANDIDATE_DESCRIPTIONS[label],
                    **empty_motion,
                }
                for label in CANDIDATE_DESCRIPTIONS
            ],
        }

    lengths = [frame["body_length"] for frame in frames]
    threshold = config.relative_motion_threshold_diagonals_per_second
    body = _motion(
        frames, [frame["vectors"]["body"] for frame in frames], lengths,
        config, config.body_motion_threshold_diagonals_per_second,
    )
    head = _motion(
        frames, [frame["vectors"]["head_relative"] for frame in frames], lengths,
        config, threshold,
    )
    tail = _motion(
        frames, [frame["vectors"]["tail_relative"] for frame in frames], lengths,
        config, threshold,
    )

    paw_speeds: list[float] = []
    paw_frame_count = 0
    paw_landmarks: set[str] = set()
    for index, frame in enumerate(frames):
        if sum(frame["vectors"][paw] is not None for paw in PAW_LANDMARKS) >= 2:
            paw_frame_count += 1
        if index == 0:
            continue
        previous = frames[index - 1]
        dt = frame["timestamp_seconds"] - previous["timestamp_seconds"]
        if frame["frame"] != previous["frame"] + 1 or dt <= 0 or dt > config.maximum_gap_factor / frame["fps"]:
            continue
        normalizer = frame["diagonal_pixels"]
        for paw in PAW_LANDMARKS:
            first = previous["vectors"][paw]
            second = frame["vectors"][paw]
            if first is not None and second is not None:
                paw_speeds.append(_distance(first, second) / dt / normalizer)
                paw_landmarks.add(paw)
    scale_coverage = _coverage(sum(length is not None for length in lengths), len(frames))
    paw_coverage = _coverage(paw_frame_count, len(frames))
    window_duration = end - start
    enough_paws = (
        window_duration >= config.minimum_duration_seconds
        and paw_coverage >= config.minimum_coverage
        and len(paw_speeds) >= config.minimum_valid_pairs
        and len(paw_landmarks) >= 2
    )
    paw_median = statistics.median(paw_speeds) if paw_speeds else None
    paw_candidate = {
        "status": "insufficient_data" if not enough_paws else (
            "detected" if paw_median is not None and paw_median >= threshold else "not_detected"
        ),
        "median_speed_image_diagonals_per_second": paw_median,
        "threshold_image_diagonals_per_second": threshold,
        "valid_frame_percent": 100 * paw_coverage,
        "body_scale_valid_frame_percent": 100 * scale_coverage,
        "valid_pair_count": len(paw_speeds),
        "minimum_coverage_percent": 100 * config.minimum_coverage,
        "minimum_duration_seconds": config.minimum_duration_seconds,
        "landmarks_with_valid_pairs": sorted(paw_landmarks),
    }
    lengths_valid = [length for length in lengths if length is not None]

    candidates = [
        {"label": "body_displacement", "description": CANDIDATE_DESCRIPTIONS["body_displacement"], **body},
        {"label": "head_relative_motion", "description": CANDIDATE_DESCRIPTIONS["head_relative_motion"], **head},
        {"label": "tail_relative_motion", "description": CANDIDATE_DESCRIPTIONS["tail_relative_motion"], **tail},
        {"label": "paw_relative_motion", "description": CANDIDATE_DESCRIPTIONS["paw_relative_motion"], **paw_candidate},
    ]
    return {
        "start_seconds": start,
        "end_seconds": end,
        "frames_processed": len(frames),
        "measurements": {
            "median_body_length_pixels": statistics.median(lengths_valid) if lengths_valid else None,
            "body_length_valid_frame_percent": 100 * scale_coverage,
            "body_center_valid_frame_percent": body["valid_frame_percent"],
            "head_relative_valid_frame_percent": head["valid_frame_percent"],
            "tail_relative_valid_frame_percent": tail["valid_frame_percent"],
            "paw_relative_valid_frame_percent": 100 * paw_coverage,
        },
        "candidates": candidates,
    }


def prepare_dataset(
    dataset: dict[str, Any],
    *,
    config: PreprocessConfig | None = None,
    landmark_map: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Return a compact summary; candidates describe motion, not confirmed behavior."""
    settings = config or PreprocessConfig()
    if not isinstance(settings, PreprocessConfig):
        raise TypeError("config must be a PreprocessConfig instance")
    info, raw_frames, start, end = _validate_dataset(dataset)
    sources = _resolve_sources(dataset, landmark_map)
    frames = _prepare_frames(raw_frames, sources, info, settings)
    quality = _quality(frames, sources)

    warnings: list[str] = []
    declared_count = dataset["metadata"].get("frame_count")
    if declared_count is not None and declared_count != len(raw_frames):
        warnings.append(f"Declared frame_count={declared_count}; received {len(raw_frames)} frames.")
    declared_duration = _number(dataset["metadata"].get("duration_seconds"))
    frame_derived_duration = (
        raw_frames[-1]["timestamp_seconds"] + 1 / info["fps"] - start
    )
    if declared_duration and abs(frame_derived_duration - declared_duration) > 1 / info["fps"]:
        warnings.append("Declared duration and frame timestamps do not agree.")

    interval_count = max(1, math.ceil((end - start) / settings.segment_seconds))
    intervals = []
    for index in range(interval_count):
        interval_start = start + index * settings.segment_seconds
        interval_end = min(end, interval_start + settings.segment_seconds)
        selected = [
            frame for frame in frames
            if interval_start <= frame["timestamp_seconds"] < interval_end
            or (index == interval_count - 1 and frame["timestamp_seconds"] == interval_end)
        ]
        intervals.append(_analyze_window(selected, interval_start, interval_end, settings))

    return {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "name": dataset["metadata"].get("original_video"),
            "duration_seconds": round(end - start, 6),
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "model": dataset["metadata"].get("model"),
            "pose_model": dataset["metadata"].get("pose_model"),
            "detector": dataset["metadata"].get("detector"),
        },
        "data_quality": quality,
        "intervals": intervals,
        "analysis_settings": {
            **asdict(settings),
            "landmark_sources": sources,
            "distance_normalization": "image diagonal; speed thresholds are image diagonals per second",
            "candidate_interpretation": "measured motion only; not a confirmed behavior or emotional state",
        },
        "warnings": warnings,
        "limitations": [
            "Measurements use 2D image coordinates and can be affected by camera motion, viewpoint, and occlusion.",
            "Motion candidates do not establish intent, emotion, health, or a specific behavior.",
            "Thresholds are configurable starting values and require validation on labeled videos.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="DeepLabCut keypoint JSON")
    parser.add_argument("--output", type=Path, help="Write summary to this JSON file")
    parser.add_argument("--confidence", type=float, default=PreprocessConfig.confidence_threshold)
    parser.add_argument("--segment-seconds", type=float, default=PreprocessConfig.segment_seconds)
    parser.add_argument(
        "--body-motion-threshold",
        type=float,
        default=PreprocessConfig.body_motion_threshold_diagonals_per_second,
        help="Image diagonals per second",
    )
    parser.add_argument(
        "--relative-motion-threshold",
        type=float,
        default=PreprocessConfig.relative_motion_threshold_diagonals_per_second,
        help="Image diagonals per second",
    )
    args = parser.parse_args(argv)
    try:
        summary = prepare_dataset(
            json.loads(args.input.read_text(encoding="utf-8-sig")),
            config=PreprocessConfig(
                confidence_threshold=args.confidence,
                segment_seconds=args.segment_seconds,
                body_motion_threshold_diagonals_per_second=args.body_motion_threshold,
                relative_motion_threshold_diagonals_per_second=args.relative_motion_threshold,
            ),
        )
        output = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
