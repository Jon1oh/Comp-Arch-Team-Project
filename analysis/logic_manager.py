"""Convert single-animal DLC predictions into compact, measurable AI input.

Supports both this project's SuperAnimal JSON and the proposed standardized JSON.
No DeepLabCut, OpenCV, pandas, network access, or AI calls are needed here.

    from analysis.logic_manager import analyze_dataset
    summary = analyze_dataset(dataset)  # dictionary returned by process_video()

    python analysis/logic_manager.py input_keypoints.json --output summary.json

Angles are in the image plane: 0 degrees points right, +90 points down.
Activity labels are configurable engineering heuristics, not validated behaviors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys


# Choose ONE source label per landmark for the whole video. The trunk aliases
# are geometric proxies: back_base -> shoulder, back_end -> hip. Antler labels
# from the quadruped model are deliberately excluded from feline features.
KEYPOINT_ALIASES = {
    "nose": ("nose",),
    "left_eye": ("left_eye",),
    "right_eye": ("right_eye",),
    "left_ear_base": ("left_ear_base", "left_earbase"),
    "right_ear_base": ("right_ear_base", "right_earbase"),
    "left_ear_tip": ("left_ear_tip", "left_earend"),
    "right_ear_tip": ("right_ear_tip", "right_earend"),
    "shoulder": ("shoulder", "back_base"),
    "spine_mid": ("spine_mid", "back_middle"),
    "hip": ("hip", "back_end"),
    "tail_base": ("tail_base",),
    "tail_mid": ("tail_mid",),
    "tail_tip": ("tail_tip", "tail_end"),
}


def _number(value):
    """Accept finite JSON numbers, excluding booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class LogicConfig:
    confidence_threshold: float = 0.60
    smoothing_window: int = 5
    segment_seconds: float = 3.0
    # Both thresholds use fractions of the image diagonal per second.
    low_speed_threshold: float = 0.02
    high_speed_threshold: float = 0.10
    minimum_coverage: float = 0.50
    direction_change_degrees: float = 60.0
    angular_deadband_degrees: float = 2.0
    maximum_gap_factor: float = 1.5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if _number(value) is None:
                raise ValueError(f"{name} must be a finite number.")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if (type(self.smoothing_window) is not int
                or self.smoothing_window < 1 or self.smoothing_window % 2 == 0):
            raise ValueError("smoothing_window must be a positive odd integer.")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive.")
        if not 0 < self.low_speed_threshold < self.high_speed_threshold:
            raise ValueError("Speed thresholds must satisfy 0 < low < high.")
        if not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be greater than 0 and at most 1.")
        if not 0 < self.direction_change_degrees <= 180:
            raise ValueError("direction_change_degrees must be in (0, 180].")
        if not 0 < self.angular_deadband_degrees <= 180:
            raise ValueError("angular_deadband_degrees must be in (0, 180].")
        if self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be at least 1.")


def load_dataset(input_path):
    """Read a UTF-8 JSON file; structure is validated during analysis."""
    path = Path(input_path).expanduser()
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid DLC JSON in {path}: {exc}") from exc


def _validate_dataset(dataset):
    if not isinstance(dataset, dict):
        raise ValueError("The dataset must be a dictionary/JSON object.")
    metadata = dataset.get("metadata")
    frames = dataset.get("frames")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object.")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty list.")
    if metadata.get("animal_count", 1) != 1 or isinstance(
        metadata.get("animal_count"), bool
    ):
        raise ValueError("Only single-animal datasets are supported.")

    info = {}
    for name in ("fps", "width", "height"):
        value = _number(metadata.get(name, metadata.get(f"video_{name}")))
        if value is None or value <= 0:
            raise ValueError(f"metadata.{name} (or video_{name}) must be positive.")
        info[name] = value
    info["diagonal"] = math.hypot(info["width"], info["height"])
    if not math.isfinite(info["diagonal"]):
        raise ValueError("Video dimensions are too large.")

    available = set()
    names = dataset.get("keypoint_names", [])
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise ValueError("keypoint_names must be a list of strings.")
    available.update(names)
    normalized_frames = []
    previous_index, previous_time = -1, -1.0
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{position}] must be an object.")
        index = frame.get("frame", position)
        if type(index) is not int or index < 0 or index <= previous_index:
            raise ValueError("Frame indices must be non-negative, increasing integers.")
        timestamp = _number(frame.get("timestamp_seconds", index / info["fps"]))
        if timestamp is None or timestamp < 0 or timestamp <= previous_time:
            raise ValueError("Timestamps must be finite, non-negative and increasing.")
        points = frame.get("keypoints")
        if not isinstance(points, dict) or any(not isinstance(n, str) for n in points):
            raise ValueError(f"frames[{position}].keypoints must be an object with string keys.")
        available.update(points)
        normalized_frames.append({
            "frame": index, "timestamp_seconds": timestamp, "keypoints": points,
        })
        previous_index, previous_time = index, timestamp

    warnings = []
    for field in ("processed_frame_count", "frame_count", "video_frame_count"):
        count = metadata.get(field)
        if count is not None and count != len(frames):
            warnings.append(f"metadata.{field}={count}; received {len(frames)} records.")
    info["start"] = normalized_frames[0]["timestamp_seconds"]
    info["end"] = normalized_frames[-1]["timestamp_seconds"] + 1 / info["fps"]
    duration = _number(metadata.get("duration_seconds", metadata.get("video_duration_seconds")))
    if duration is not None and duration > 0:
        info["duration"] = duration
        if abs(duration - info["end"]) > 1 / info["fps"]:
            warnings.append("Video duration and the supplied frame timestamps do not agree.")
    else:
        info["duration"] = info["end"]
    info["video_name"] = metadata.get("video_name", metadata.get("original_video"))
    info["warnings"] = warnings
    return info, normalized_frames, available


def _resolve_mapping(available, keypoint_map):
    overrides = {} if keypoint_map is None else keypoint_map
    if not isinstance(overrides, dict):
        raise ValueError("keypoint_map must be a dictionary of canonical -> source names.")
    if set(overrides) - KEYPOINT_ALIASES.keys():
        raise ValueError("keypoint_map contains unknown canonical landmarks.")
    mapping = {}
    for canonical, aliases in KEYPOINT_ALIASES.items():
        if canonical in overrides:
            source = overrides[canonical]
            if source is not None and (not isinstance(source, str) or not source):
                raise ValueError("Mapped labels must be non-empty strings or None.")
            mapping[canonical] = source if source in available else None
        else:
            mapping[canonical] = next((name for name in aliases if name in available), None)
    return mapping


def clean_keypoints(frames, mapping, width, height, confidence_threshold=0.60):
    """Keep reliable, in-bounds observations; retain frames with missing points.

    The source valid=False flag is respected even if the configured threshold
    is lower. Coordinates are never interpolated and the input is not mutated.
    """
    result = []
    for frame in frames:
        points = {}
        for canonical, source in mapping.items():
            point = frame["keypoints"].get(source) if source is not None else None
            if not isinstance(point, dict) or point.get("valid", True) is not True:
                continue
            x, y, confidence = (_number(point.get(key)) for key in ("x", "y", "likelihood"))
            if (x is None or y is None or confidence is None
                    or not confidence_threshold <= confidence <= 1
                    or not 0 <= x < width or not 0 <= y < height):
                continue
            points[canonical] = {"x": x, "y": y, "likelihood": confidence}
        result.append({
            "frame": frame["frame"],
            "timestamp_seconds": frame["timestamp_seconds"],
            "keypoints": points,
        })
    return result


def _adjacent(previous, current, info, config):
    dt = current["timestamp_seconds"] - previous["timestamp_seconds"]
    return (current["frame"] == previous["frame"] + 1
            and 0 < dt <= config.maximum_gap_factor / info["fps"])


def smooth_keypoints(frames, info, config):
    """Centered moving average within each uninterrupted valid landmark run."""
    result = [dict(frame, keypoints={name: dict(point) for name, point in frame["keypoints"].items()})
              for frame in frames]
    if config.smoothing_window == 1:
        return result
    radius = config.smoothing_window // 2
    for name in KEYPOINT_ALIASES:
        start = 0
        while start < len(frames):
            if name not in frames[start]["keypoints"]:
                start += 1
                continue
            end = start + 1
            while (end < len(frames) and name in frames[end]["keypoints"]
                   and _adjacent(frames[end - 1], frames[end], info, config)):
                end += 1
            for coordinate in ("x", "y"):
                prefix = [0.0]
                for index in range(start, end):
                    prefix.append(prefix[-1] + frames[index]["keypoints"][name][coordinate])
                for index in range(start, end):
                    left, right = max(start, index - radius), min(end, index + radius + 1)
                    result[index]["keypoints"][name][coordinate] = (
                        prefix[right - start] - prefix[left - start]
                    ) / (right - left)
            start = end
    return result


def calculate_distance(point1, point2):
    return math.hypot(point2["x"] - point1["x"], point2["y"] - point1["y"])


def _mean(values):
    return statistics.fmean(values) if values else None


def _percent(count, total):
    return 100.0 * count / total if total else 0.0


def _activity(speed, valid_count, frame_count, pair_count, config):
    if speed is None or pair_count < 2 or valid_count / max(frame_count, 1) < config.minimum_coverage:
        return "unknown"
    if speed < config.low_speed_threshold:
        return "low"
    return "moderate" if speed < config.high_speed_threshold else "high"


def _movement_stats(frames, series, info, config):
    distances, speeds = [], []
    observed_seconds = 0.0
    direction_changes, bouts = 0, 0
    previous_direction, previously_active = None, False
    for index in range(1, len(frames)):
        previous, current = series[index - 1], series[index]
        if previous is None or current is None or not _adjacent(frames[index - 1], frames[index], info, config):
            previous_direction, previously_active = None, False
            continue
        dt = frames[index]["timestamp_seconds"] - frames[index - 1]["timestamp_seconds"]
        distance = calculate_distance(previous, current)
        speed = distance / dt
        normalized = speed / info["diagonal"]
        distances.append(distance)
        speeds.append(speed)
        observed_seconds += dt
        active = normalized >= config.low_speed_threshold
        if active and not previously_active:
            bouts += 1
        if active:
            direction = math.degrees(math.atan2(current["y"] - previous["y"], current["x"] - previous["x"]))
            if previous_direction is not None and abs(_angle_delta(direction, previous_direction)) >= config.direction_change_degrees:
                direction_changes += 1
            previous_direction = direction
        else:
            previous_direction = None
        previously_active = active
    valid_count = sum(point is not None for point in series)
    # Time-weighted speed avoids bias when frame intervals differ slightly.
    mean_speed = sum(distances) / observed_seconds if observed_seconds else None
    normalized_speed = mean_speed / info["diagonal"] if mean_speed is not None else None
    return {
        "movement": _activity(normalized_speed, valid_count, len(frames), len(distances), config),
        "average_displacement_pixels": _mean(distances),
        "maximum_displacement_pixels": max(distances) if distances else None,
        "average_speed_pixels_per_second": mean_speed,
        "maximum_speed_pixels_per_second": max(speeds) if speeds else None,
        "average_speed_diagonals_per_second": normalized_speed,
        "valid_measurement_count": len(distances),
        "valid_frame_percentage": _percent(valid_count, len(frames)),
        "observed_pair_seconds": observed_seconds,
        "movement_direction_changes": direction_changes,
        "movement_bout_count": bouts,
    }


def calculate_movement(frames, info, config):
    """Image-plane displacement and speed for each canonical landmark."""
    return {name: _movement_stats(frames, [frame["keypoints"].get(name) for frame in frames], info, config)
            for name in KEYPOINT_ALIASES}


def _angle_delta(current, previous):
    return (current - previous + 180) % 360 - 180


def _angle(origin, tip):
    if calculate_distance(origin, tip) < 1e-9:
        return None
    return math.degrees(math.atan2(tip["y"] - origin["y"], tip["x"] - origin["x"]))


def _midpoint(first, second):
    return {"x": (first["x"] + second["x"]) / 2, "y": (first["y"] + second["y"]) / 2}


def _angular_stats(frames, angles, info, config):
    values = [angle for angle in angles if angle is not None]
    mean_angle = concentration = None
    if values:
        x = statistics.fmean(math.cos(math.radians(value)) for value in values)
        y = statistics.fmean(math.sin(math.radians(value)) for value in values)
        concentration = math.hypot(x, y)
        if concentration > 1e-6:
            mean_angle = math.degrees(math.atan2(y, x))
    total_change, observed_seconds = 0.0, 0.0
    reversals, previous_sign, pairs = 0, None, 0
    for index in range(1, len(frames)):
        if angles[index - 1] is None or angles[index] is None or not _adjacent(frames[index - 1], frames[index], info, config):
            previous_sign = None
            continue
        change = _angle_delta(angles[index], angles[index - 1])
        total_change += abs(change)
        observed_seconds += frames[index]["timestamp_seconds"] - frames[index - 1]["timestamp_seconds"]
        pairs += 1
        if abs(change) >= config.angular_deadband_degrees:
            sign = 1 if change > 0 else -1
            if previous_sign is not None and sign != previous_sign:
                reversals += 1
            previous_sign = sign
    enough = (len(values) / max(len(frames), 1) >= config.minimum_coverage and pairs >= 2)
    return {
        "mean_angle_degrees": mean_angle,
        "circular_concentration": concentration,
        "average_angular_speed_degrees_per_second": total_change / observed_seconds if observed_seconds else None,
        "angular_direction_changes": reversals,
        "angular_direction_changes_per_second": reversals / observed_seconds if enough and observed_seconds else None,
        "valid_frame_percentage": _percent(len(values), len(frames)),
        "valid_measurement_count": pairs,
    }


def _quality(frames, mapping):
    counts = {name: sum(name in frame["keypoints"] for frame in frames) for name in mapping}
    mapped = sum(source is not None for source in mapping.values())
    percentage = _percent(sum(counts.values()), len(frames) * mapped)
    return {
        "rating": "good" if percentage >= 80 else "limited" if percentage >= 50 else "poor",
        "reliable_detection_percentage": percentage,
        "mapped_keypoint_count": mapped,
        "missing_keypoints": [name for name, source in mapping.items() if source is None],
        "keypoint_coverage_percentage": {name: _percent(count, len(frames)) for name, count in counts.items()},
    }


def extract_features(frames, info, config, movement=None):
    """Compute geometry without assigning emotions, gaze, or physical posture."""
    movement = calculate_movement(frames, info, config) if movement is None else movement
    body, tail_relative, tail_angles, tail_positions = [], [], [], []
    head_angles, spine_angles, eye_distances = [], [], []
    ear_relative = {side: [] for side in ("left", "right")}
    ear_angles = {side: [] for side in ear_relative}
    ear_head_angles = {side: [] for side in ear_relative}

    for frame in frames:
        points = frame["keypoints"]
        shoulder, hip, spine = (points.get(name) for name in ("shoulder", "hip", "spine_mid"))
        body.append(_midpoint(shoulder, hip) if shoulder and hip else None)
        nose, left_base, right_base = (points.get(name) for name in ("nose", "left_ear_base", "right_ear_base"))
        head_angle = _angle(_midpoint(left_base, right_base), nose) if nose and left_base and right_base else None
        head_angles.append(head_angle)
        for side in ear_relative:
            base, tip = points.get(f"{side}_ear_base"), points.get(f"{side}_ear_tip")
            angle = _angle(base, tip) if base and tip else None
            ear_angles[side].append(angle)
            ear_relative[side].append({"x": tip["x"] - base["x"], "y": tip["y"] - base["y"]} if base and tip else None)
            ear_head_angles[side].append(abs(_angle_delta(angle, head_angle)) if angle is not None and head_angle is not None else None)
        base, tip = points.get("tail_base"), points.get("tail_tip")
        tail_relative.append({"x": tip["x"] - base["x"], "y": tip["y"] - base["y"]} if base and tip else None)
        tail_angles.append(_angle(base, tip) if base and tip else None)
        if base and tip and calculate_distance(base, tip) > 1e-9:
            vertical = (tip["y"] - base["y"]) / calculate_distance(base, tip)
            tail_positions.append("above_base_in_image" if vertical < -0.15 else "below_base_in_image" if vertical > 0.15 else "level_with_base_in_image")
        if shoulder and hip and spine:
            first, second = _angle(spine, shoulder), _angle(spine, hip)
            if first is not None and second is not None:
                spine_angles.append(abs(_angle_delta(first, second)))
        left_eye, right_eye = points.get("left_eye"), points.get("right_eye")
        if left_eye and right_eye:
            eye_distances.append(calculate_distance(left_eye, right_eye))

    tail_pattern = "unknown"
    if tail_positions and len(tail_positions) / max(len(frames), 1) >= config.minimum_coverage:
        name, count = Counter(tail_positions).most_common(1)[0]
        tail_pattern = f"mostly_{name}" if count / len(tail_positions) >= 0.70 else "variable_in_image"
    body_stats = _movement_stats(frames, body, info, config)
    ears = {}
    for side in ear_relative:
        offsets = [value for value in ear_head_angles[side] if value is not None]
        ears[side] = {
            "relative_movement": _movement_stats(frames, ear_relative[side], info, config),
            "orientation_in_image": _angular_stats(frames, ear_angles[side], info, config),
            "mean_angle_to_head_axis_degrees": _mean(offsets),
            "head_axis_comparison_count": len(offsets),
        }
    return {
        "head": {"movement": movement["nose"]["movement"],
                 "movement_direction_changes": movement["nose"]["movement_direction_changes"],
                 "orientation_in_image": _angular_stats(frames, head_angles, info, config)},
        "ears": ears,
        "eyes": {"mean_inter_eye_distance_pixels": _mean(eye_distances),
                 "valid_frame_percentage": _percent(len(eye_distances), len(frames)),
                 "gaze_direction": "not_inferred_from_eye_centers"},
        "tail": {"movement": _movement_stats(frames, tail_relative, info, config),
                 "position_pattern": tail_pattern,
                 "orientation_in_image": _angular_stats(frames, tail_angles, info, config)},
        "body": {"movement": body_stats,
                 "position_pattern": "low_displacement_in_image" if body_stats["movement"] == "low" else "moving_in_image" if body_stats["movement"] in ("moderate", "high") else "unknown",
                 "mean_spine_interior_angle_degrees": _mean(spine_angles),
                 "spine_measurement_count": len(spine_angles)},
    }


def _activity_labels(features):
    return {
        "head": features["head"]["movement"],
        "left_ear": features["ears"]["left"]["relative_movement"]["movement"],
        "right_ear": features["ears"]["right"]["relative_movement"]["movement"],
        "tail": features["tail"]["movement"]["movement"],
        "body": features["body"]["movement"]["movement"],
    }


def analyze_temporal_patterns(frames, cleaned_frames, mapping, info, config):
    """Summarize up to 60 time windows, including windows with no valid points."""
    duration = info["end"] - info["start"]
    interval = max(config.segment_seconds, duration / 60)
    count = min(60, max(1, math.ceil(duration / interval)))
    groups, clean_groups = [[] for _ in range(count)], [[] for _ in range(count)]
    for frame, cleaned in zip(frames, cleaned_frames):
        index = min(count - 1, int((frame["timestamp_seconds"] - info["start"]) / interval))
        groups[index].append(frame)
        clean_groups[index].append(cleaned)
    result, previous_labels = [], None
    ranks = {"low": 0, "moderate": 1, "high": 2}
    for index, (group, clean_group) in enumerate(zip(groups, clean_groups)):
        features = extract_features(group, info, config)
        labels = _activity_labels(features)
        changes = []
        if previous_labels is not None:
            for name, label in labels.items():
                previous = previous_labels[name]
                if label in ranks and previous in ranks and label != previous:
                    changes.append(f"{name}_activity_{'increased' if ranks[label] > ranks[previous] else 'decreased'}")
        quality = _quality(clean_group, mapping)
        result.append({
            "start_seconds": info["start"] + index * interval,
            "end_seconds": min(info["end"], info["start"] + (index + 1) * interval),
            "frames_processed": len(group),
            "data_quality": quality["rating"],
            "reliable_detection_percentage": quality["reliable_detection_percentage"],
            "movement": labels,
            "tail_position_pattern": features["tail"]["position_pattern"],
            "changes_from_previous_interval": changes,
        })
        previous_labels = labels
    return result


def _rounded(value):
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


class LogicManager:
    """Coordinate validation, cleaning, smoothing, feature and time analysis."""

    def __init__(self, config=None, keypoint_map=None):
        self.config = LogicConfig() if config is None else config
        if not isinstance(self.config, LogicConfig):
            raise TypeError("config must be a LogicConfig instance.")
        self.keypoint_map = dict(keypoint_map) if isinstance(keypoint_map, dict) else keypoint_map

    def analyze(self, dataset):
        info, frames, available = _validate_dataset(dataset)
        mapping = _resolve_mapping(available, self.keypoint_map)
        cleaned = clean_keypoints(frames, mapping, info["width"], info["height"], self.config.confidence_threshold)
        smoothed = smooth_keypoints(cleaned, info, self.config)
        quality = _quality(cleaned, mapping)
        movement = calculate_movement(smoothed, info, self.config)
        features = extract_features(smoothed, info, self.config, movement)
        limitations = [
            "All measurements are 2D image-plane geometry; camera motion and perspective can affect them.",
            "Activity and quality thresholds are engineering heuristics requiring validation on annotated cat videos.",
            "Ear rotation, eye closure, gaze, emotions and diagnoses cannot be established from these landmarks alone.",
            "Tail angular direction-change rate is an exploratory activity measure, not a validated tail-swish frequency.",
        ]
        if mapping["shoulder"] == "back_base" or mapping["hip"] == "back_end":
            limitations.append("SuperAnimal back_base/back_end are front/rear trunk proxies for shoulder/hip.")
        if quality["missing_keypoints"]:
            limitations.append("Unavailable landmarks: " + ", ".join(quality["missing_keypoints"]) + ".")
        if quality["rating"] != "good":
            limitations.append("Incomplete reliable detections limit interpretation; check per-feature coverage.")
        unknown = [name for name, label in _activity_labels(features).items() if label == "unknown"]
        if unknown:
            limitations.append("Insufficient consecutive reliable samples for activity labels: " + ", ".join(unknown) + ".")
        gaps = sum(not _adjacent(first, second, info, self.config) for first, second in zip(cleaned, cleaned[1:]))
        if gaps:
            info["warnings"].append(f"{gaps} frame/time gaps were excluded from movement and smoothing.")
        return _rounded({
            "schema_version": "1.0",
            "video_summary": {
                "video_name": info["video_name"], "duration_seconds": info["duration"],
                "analysis_start_seconds": info["start"], "analysis_end_seconds": info["end"],
                "fps": info["fps"], "width": info["width"], "height": info["height"],
                "frames_processed": len(frames), "data_quality": quality["rating"],
            },
            "data_quality": quality,
            "observed_features": features,
            "movement_metrics": movement,
            "temporal_patterns": analyze_temporal_patterns(smoothed, cleaned, mapping, info, self.config),
            "analysis_settings": {
                **asdict(self.config), "keypoint_mapping": mapping,
                "normalization": "pixel distance / image diagonal; speed uses timestamp differences",
                "angle_convention": "degrees in image plane; 0=right, +90=down",
                "body_anchor": "midpoint of shoulder and hip; requires both",
                "tail_movement_reference": "tail_tip minus tail_base",
                "ear_movement_reference": "ear_tip minus its ear_base",
                "smoothing": "centered moving average; no interpolation across missing detections or gaps",
            },
            "warnings": info["warnings"], "limitations": limitations,
        })

    def analyze_file(self, input_path, output_path=None):
        summary = self.analyze(load_dataset(input_path))
        if output_path is not None:
            destination = Path(output_path).expanduser()
            if destination.resolve() == Path(input_path).expanduser().resolve():
                raise ValueError("The summary output path must differ from the input dataset.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
        return summary


def analyze_dataset(dataset, *, config=None, keypoint_map=None):
    """Return an AI-ready dictionary from process_video()'s returned dataset."""
    return LogicManager(config=config, keypoint_map=keypoint_map).analyze(dataset)


def analyze_file(input_path, output_path=None, *, config=None, keypoint_map=None):
    """Analyze saved DLC JSON; optionally also write the summary as JSON."""
    return LogicManager(config=config, keypoint_map=keypoint_map).analyze_file(input_path, output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="DLC keypoint JSON file")
    parser.add_argument("--output", type=Path, help="Write the AI-ready summary to this JSON file")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--smoothing-window", type=int, default=5, help="Positive odd number; 1 disables smoothing")
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        config = LogicConfig(confidence_threshold=args.confidence,
                             smoothing_window=args.smoothing_window,
                             segment_seconds=args.segment_seconds)
        summary = analyze_file(args.input, args.output, config=config)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Logic analysis failed: {exc}\n")
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
