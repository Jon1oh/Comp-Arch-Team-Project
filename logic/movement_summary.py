"""Build evidence for any supported single-cat clip, using only the standard library.

Run: python -m logic.movement_summary input_keypoints.json
Summaries default to analysis/DeepLabCut-Processing/outputs/summaries/.
Use --output to choose a different destination.
Rules describe projected movement, not confirmed behavior, emotion, or health.
"""

from __future__ import annotations
import argparse, json, math, sys, statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "movement-summary/1"
RULE_VERSION = "1"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "DeepLabCut-Processing"
    / "outputs"
)
ALIASES = {
    "torso_front": ("back_base", "shoulder"),
    "torso_rear": ("back_end", "hip"),
    "nose": ("nose",),
    "left_ear_base": ("left_earbase", "left_ear_base"),
    "right_ear_base": ("right_earbase", "right_ear_base"),
    "left_ear_tip": ("left_earend", "left_ear_tip"),
    "right_ear_tip": ("right_earend", "right_ear_tip"),
    "tail_base": ("tail_base",),
    "tail_tip": ("tail_end", "tail_tip"),
    **{name: (name,) for name in (
        "front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"
    )},
}
LIMITATIONS = [
    "All positions and angles are 2D projections; occlusion and viewpoint affect measurements.",
    "Camera motion and zoom are not estimated or corrected; body displacement is movement in the image.",
    "Image-diagonal normalization handles image resizing, not camera distance or real-world scale.",
    "Landmark likelihood is a model score, not a calibrated probability of correct behavior recognition.",
    "Rule thresholds are engineering starting values, not validated feline behavior standards.",
    "Median smoothing and isolated-spike filtering can suppress real rapid movements; estimates depend on these settings.",
    "These measurements cannot establish gaze, attention, emotion, intent, health, contact, or licking.",
]


@dataclass(frozen=True)
class MovementConfig:
    experimental_behavior_candidates: bool = False
    confidence_threshold: float = 0.60
    segment_seconds: float = 3.0
    max_intervals: int = 60
    smoothing_window: int = 3
    minimum_frame_coverage: float = 0.60
    minimum_pair_coverage: float = 0.50
    minimum_valid_pairs: int = 3
    minimum_observed_seconds: float = 0.50
    maximum_gap_factor: float = 1.5
    minimum_axis_diagonals: float = 0.005
    isolated_spike_speed_diagonals_per_second: float = 4.0
    body_motion_diagonals_per_second: float = 0.02
    relative_motion_diagonals_per_second: float = 0.01
    angular_motion_degrees_per_second: float = 15.0
    angular_deadband_degrees: float = 2.0
    direction_change_degrees: float = 60.0
    minimum_bout_seconds: float = 0.20
    bout_exit_threshold_ratio: float = 0.70
    stationary_duration_seconds: float = 1.0
    stationary_radius_diagonals: float = 0.01
    tail_minimum_reversals: int = 3
    tail_minimum_excursion_degrees: float = 20.0
    paw_minimum_direction_changes: int = 3
    play_minimum_direction_changes: int = 2
    nose_paw_distance_body_lengths: float = 0.25
    grooming_minimum_cooccurrence: float = 0.30

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "experimental_behavior_candidates":
                if type(value) is not bool:
                    raise ValueError("experimental_behavior_candidates must be a boolean")
                continue
            if _number(value) is None or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("max_intervals", "smoothing_window", "minimum_valid_pairs",
                     "tail_minimum_reversals", "paw_minimum_direction_changes", "play_minimum_direction_changes"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.smoothing_window % 2 != 1:
            raise ValueError("smoothing_window must be odd; 1 disables smoothing")
        for name in ("confidence_threshold", "minimum_frame_coverage", "minimum_pair_coverage",
                     "bout_exit_threshold_ratio", "grooming_minimum_cooccurrence"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed 1")
        if self.maximum_gap_factor < 1:
            raise ValueError("maximum_gap_factor must be at least 1")
        if self.direction_change_degrees > 180 or self.angular_deadband_degrees > 180:
            raise ValueError("Angular thresholds must not exceed 180 degrees")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _percent(count, total):
    return 100 * count / total if total else 0.0


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _delta(a, b):
    return (a - b + 180) % 360 - 180


def _axis(base, tip, minimum):
    if base is None or tip is None or _distance(base, tip) < minimum:
        return None
    return math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))


def _relative(point, anchor, angle):
    if point is None or anchor is None or angle is None:
        return None
    x, y = point[0] - anchor[0], point[1] - anchor[1]
    r = math.radians(angle)
    return (x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r))


def _validate(dataset):
    if not isinstance(dataset, dict) or not isinstance(dataset.get("metadata"), dict):
        raise ValueError("Expected raw keypoint JSON with metadata and frames")
    metadata, frames = dataset["metadata"], dataset.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a nonempty list")
    if type(metadata.get("animal_count", 1)) is not int or metadata.get("animal_count", 1) != 1:
        raise ValueError("Only single-animal datasets are supported")
    info = {}
    for name in ("fps", "width", "height"):
        value = _number(metadata.get(name))
        if value is None or value <= 0:
            raise ValueError(f"metadata.{name} must be positive and finite")
        info[name] = value
    info["diagonal"] = math.hypot(info["width"], info["height"])
    if not math.isfinite(info["diagonal"]):
        raise ValueError("Video dimensions are too large")
    names = dataset.get("keypoint_names", [])
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise ValueError("keypoint_names must contain strings")
    available = set(names)
    normalized, last_index, last_time = [], -1, -1.0
    for position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{position}] must be an object")
        index = frame.get("frame", position)
        time = _number(frame.get("timestamp_seconds", index / info["fps"] if type(index) is int else None))
        if type(index) is not int or index < 0 or index <= last_index:
            raise ValueError("Frame numbers must be nonnegative, strictly increasing integers")
        if time is None or time < 0 or time <= last_time:
            raise ValueError("Timestamps must be finite, nonnegative and strictly increasing")
        points = frame.get("keypoints")
        if not isinstance(points, dict) or any(not isinstance(n, str) for n in points):
            raise ValueError("Each frame needs keypoints with string names")
        normalized.append({"frame": index, "time": time, "keypoints": points})
        available.update(points)
        last_index, last_time = index, time
    info["start"] = normalized[0]["time"]
    info["end"] = normalized[-1]["time"] + 1 / info["fps"]
    warnings = []
    for field in ("frame_count", "processed_frame_count"):
        if field in metadata and metadata[field] != len(frames):
            warnings.append(f"{field} reports {metadata[field]}; received {len(frames)} frame records")
    duration = _number(metadata.get("duration_seconds"))
    if duration is not None and abs(duration - info["end"]) > 1 / info["fps"] + 1e-6:
        warnings.append("Declared duration differs from the supplied timestamps; analysis uses timestamps")
    return info, normalized, available, warnings


def _adjacent(a, b, info, config):
    return (b["frame"] == a["frame"] + 1
            and 0 < b["time"] - a["time"] <= config.maximum_gap_factor / info["fps"])


def _clean(frames, mapping, info, config):
    rejected = Counter()
    series = {name: [] for name in mapping}
    for frame in frames:
        for name, source in mapping.items():
            point = frame["keypoints"].get(source)
            reason = None
            if not isinstance(point, dict):
                reason = "missing"
            elif point.get("valid", True) is not True:
                reason = "exporter_invalid"
            else:
                x, y, score = (_number(point.get(k)) for k in ("x", "y", "likelihood"))
                if x is None or y is None or score is None:
                    reason = "nonfinite_or_missing_number"
                elif not config.confidence_threshold <= score <= 1:
                    reason = "confidence"
                elif not 0 <= x < info["width"] or not 0 <= y < info["height"]:
                    reason = "out_of_bounds"
            series[name].append(None if reason else (x, y))
            if reason:
                rejected[reason] += 1
    spikes = Counter()
    for name, points in series.items():
        original = points[:]
        for i in range(1, len(points) - 1):
            a, b, c = original[i - 1:i + 2]
            if any(p is None for p in (a, b, c)) or not (
                _adjacent(frames[i - 1], frames[i], info, config)
                and _adjacent(frames[i], frames[i + 1], info, config)
            ):
                continue
            dt1 = frames[i]["time"] - frames[i - 1]["time"]
            dt2 = frames[i + 1]["time"] - frames[i]["time"]
            limit = config.isolated_spike_speed_diagonals_per_second * info["diagonal"]
            if (_distance(a, b) / dt1 > limit and _distance(b, c) / dt2 > limit
                    and _distance(a, c) / (dt1 + dt2) <= limit):
                points[i] = None
                spikes[name] += 1
    # Preserve missing observations. Never smooth or interpolate across their gaps.
    smoothed = {name: points[:] for name, points in series.items()}
    radius = config.smoothing_window // 2
    for name, points in series.items():
        start = 0
        while start < len(points):
            if points[start] is None:
                start += 1
                continue
            end = start + 1
            while end < len(points) and points[end] is not None and _adjacent(frames[end - 1], frames[end], info, config):
                end += 1
            for i in range(start, end):
                window = points[max(start, i - radius):min(end, i + radius + 1)]
                smoothed[name][i] = tuple(statistics.median(p[j] for p in window) for j in (0, 1))
            start = end
    return series, smoothed, rejected, spikes


def _features(points, info, config):
    result = {name: [] for name in (
        "body", "head_relative", "head_orientation", "head_orientation_relative_to_body",
        "left_ear_orientation_relative_to_head", "right_ear_orientation_relative_to_head",
        "tail_relative", "tail_orientation_relative_to_body", *[n for n in ALIASES if n.endswith("paw")]
    )}
    proximity = []
    minimum = info["diagonal"] * config.minimum_axis_diagonals
    for i in range(len(points["nose"])):
        p = {name: values[i] for name, values in points.items()}
        front, rear = p["torso_front"], p["torso_rear"]
        body_angle = _axis(rear, front, minimum)
        body = tuple((front[j] + rear[j]) / 2 for j in (0, 1)) if body_angle is not None else None
        left, right = p["left_ear_base"], p["right_ear_base"]
        head_base = tuple((left[j] + right[j]) / 2 for j in (0, 1)) if left and right else None
        head_angle = _axis(head_base, p["nose"], minimum)
        result["body"].append(body)
        result["head_relative"].append(_relative(p["nose"], body, body_angle))
        result["head_orientation"].append(head_angle)
        result["head_orientation_relative_to_body"].append(
            _delta(head_angle, body_angle) if head_angle is not None and body_angle is not None else None)
        for side in ("left", "right"):
            ear = _axis(p[f"{side}_ear_base"], p[f"{side}_ear_tip"], minimum)
            result[f"{side}_ear_orientation_relative_to_head"].append(
                _delta(ear, head_angle) if ear is not None and head_angle is not None else None)
        base, tip = p["tail_base"], p["tail_tip"]
        result["tail_relative"].append(
            (tip[0] - base[0], tip[1] - base[1]) if base and tip else None)
        tail_angle = _axis(base, tip, minimum)
        result["tail_orientation_relative_to_body"].append(
            _delta(tail_angle, body_angle) if tail_angle is not None and body_angle is not None else None)
        for name in ALIASES:
            if name.endswith("paw"):
                result[name].append(_relative(p[name], body, body_angle))
        distances = [_distance(p["nose"], p[name]) / _distance(front, rear)
                     for name in ("front_left_paw", "front_right_paw")
                     if body is not None and p["nose"] and p[name]]
        proximity.append(min(distances) if distances else None)
    return result, proximity


def _stats(frames, values, info, config, *, angular=False, threshold=None):
    threshold = threshold or (config.angular_motion_degrees_per_second if angular
                              else config.relative_motion_diagonals_per_second)
    records, excursions = [], []
    run_start, unwrapped = None, 0.0
    for i, value in enumerate(values):
        if value is None:
            run_start = None
            continue
        if i == 0 or values[i - 1] is None or not _adjacent(frames[i - 1], frames[i], info, config):
            run_start, unwrapped = value, 0.0
            continue
        previous = values[i - 1]
        dt = frames[i]["time"] - frames[i - 1]["time"]
        change = _delta(value, previous) if angular else _distance(previous, value)
        if angular:
            unwrapped += change
            excursions.append(abs(unwrapped))
        else:
            excursions.append(_distance(run_start, value))
        speed = abs(change) / dt
        scaled = speed if angular else speed / info["diagonal"]
        direction = (1 if change > 0 else -1) if angular else math.degrees(math.atan2(
            value[1] - previous[1], value[0] - previous[0]))
        records.append({"i": i, "dt": dt, "distance": abs(change), "speed": speed,
                        "scaled": scaled, "direction": direction, "change": abs(change)})
    count = sum(v is not None for v in values)
    pair_seconds = sum(r["dt"] for r in records)
    elapsed = frames[-1]["time"] - frames[0]["time"] if len(frames) > 1 else 0.0
    frame_coverage = count / len(frames) if frames else 0.0
    pair_coverage = len(records) / max(1, len(frames) - 1)
    reasons = []
    if frame_coverage < config.minimum_frame_coverage:
        reasons.append("Required landmarks have insufficient joint frame coverage")
    if pair_coverage < config.minimum_pair_coverage:
        reasons.append("Insufficient consecutive-pair coverage")
    if len(records) < config.minimum_valid_pairs:
        reasons.append("Too few valid consecutive measurements")
    if pair_seconds + 1e-9 < config.minimum_observed_seconds:
        reasons.append("Too little observed pair time")
    if elapsed > 0 and pair_seconds / elapsed + 1e-9 < config.minimum_pair_coverage:
        reasons.append("Observed pair time covers too little of the interval")
    if elapsed + 1e-9 < config.minimum_observed_seconds:
        reasons.append("Observation interval is too short")
    bouts, direction_changes, active_seconds = 0, 0, 0.0
    active, bout_time, last_i, previous_direction = False, 0.0, None, None
    quiet_time, longest_quiet, quiet_anchor = 0.0, 0.0, None
    angular_reversals, angular_trend, angular_position, angular_extreme = 0, 0, 0.0, 0.0
    for record in records:
        i = record["i"]
        if last_i is not None and i != last_i + 1:
            bouts += int(active and bout_time + 1e-9 >= config.minimum_bout_seconds)
            active, bout_time, previous_direction = False, 0.0, None
            quiet_time, quiet_anchor = 0.0, None
            angular_trend, angular_position, angular_extreme = 0, 0.0, 0.0
        if angular:
            angular_position += _delta(values[i], values[i - 1])
            departure = angular_position - angular_extreme
            if angular_trend == 0:
                if abs(departure) >= config.angular_deadband_degrees:
                    angular_trend = 1 if departure > 0 else -1
                    angular_extreme = angular_position
            elif departure * angular_trend > 0:
                angular_extreme = angular_position
            elif abs(departure) >= config.angular_deadband_degrees:
                angular_reversals += 1
                angular_trend *= -1
                angular_extreme = angular_position
        active_now = record["scaled"] >= threshold * (config.bout_exit_threshold_ratio if active else 1)
        if active_now:
            bout_time += record["dt"]
            active_seconds += record["dt"]
            meaningful = record["change"] >= config.angular_deadband_degrees if angular else True
            if meaningful:
                direction = record["direction"]
                if previous_direction is not None and (
                    direction != previous_direction if angular else
                    abs(_delta(direction, previous_direction)) >= config.direction_change_degrees
                ):
                    direction_changes += 1
                previous_direction = direction
        else:
            bouts += int(active and bout_time + 1e-9 >= config.minimum_bout_seconds)
            bout_time = 0.0
        active = active_now
        if record["scaled"] < threshold:
            if quiet_anchor is None:
                quiet_anchor = values[i - 1]
            extent = abs(_delta(values[i], quiet_anchor)) if angular else _distance(values[i], quiet_anchor) / info["diagonal"]
            allowed = config.angular_deadband_degrees if angular else config.stationary_radius_diagonals
            if extent <= allowed:
                quiet_time += record["dt"]
                longest_quiet = max(longest_quiet, quiet_time)
            else:
                quiet_time, quiet_anchor = 0.0, values[i]
        else:
            quiet_time, quiet_anchor = 0.0, None
        last_i = i
    bouts += int(active and bout_time + 1e-9 >= config.minimum_bout_seconds)
    total = sum(r["distance"] for r in records)
    average = total / pair_seconds if pair_seconds else None
    speeds = [r["speed"] for r in records]
    stats = {
        "data_sufficient": not reasons,
        "insufficient_data_reasons": reasons,
        "valid_frame_percentage": 100 * frame_coverage,
        "valid_pair_percentage": 100 * pair_coverage,
        "valid_measurement_count": len(records),
        "observed_pair_seconds": pair_seconds,
        "observed_time_percentage": _percent(pair_seconds, elapsed),
        "movement_direction_changes": angular_reversals if angular else direction_changes,
        "movement_bout_count": bouts,
        "active_time_percentage": _percent(active_seconds, pair_seconds) if records else None,
        "longest_stationary_duration_seconds": longest_quiet if records else None,
        "motion_threshold": threshold,
        "threshold_units": "degrees/second" if angular else "image diagonals/second",
    }
    if angular:
        stats.update({"average_angular_speed_degrees_per_second": average,
                      "maximum_angular_speed_degrees_per_second": max(speeds) if speeds else None,
                      "maximum_excursion_degrees": max(excursions) if excursions else None,
                      "angular_direction_reversals": angular_reversals})
    else:
        stats.update({"average_displacement_pixels": total / len(records) if records else None,
                      "maximum_displacement_pixels": max(r["distance"] for r in records) if records else None,
                      "total_distance_pixels": total if records else None,
                      "maximum_excursion_pixels": max(excursions) if excursions else None,
                      "average_speed_pixels_per_second": average,
                      "maximum_speed_pixels_per_second": max(speeds) if speeds else None,
                      "median_speed_pixels_per_second": statistics.median(speeds) if speeds else None,
                      "average_speed_diagonals_per_second": average / info["diagonal"] if average is not None else None})
    return stats, records


def _candidate(label, rule, stats, detected, evidence, limitations):
    reasons = [reason for s in stats for reason in s["insufficient_data_reasons"]]
    return {"label": label, "rule_id": label, "rule_version": RULE_VERSION,
            "status": "insufficient_data" if reasons else "detected" if detected else "not_detected",
            "rule": rule, "score": None, "evidence": evidence,
            "limitations": limitations + sorted(set(reasons))}


def _motion_evidence(name, stat):
    return {"measurement": name, **{key: stat[key] for key in (
        "valid_frame_percentage", "valid_pair_percentage", "valid_measurement_count",
        "observed_pair_seconds", "movement_bout_count", "motion_threshold", "threshold_units",
    )}, "average_speed": stat.get("average_speed_diagonals_per_second",
                                   stat.get("average_angular_speed_degrees_per_second"))}


def _window(frames, series, proximity, info, config, start, end):
    measurements, records = {}, {}
    for name, values in series.items():
        angular = "orientation" in name
        measurements[name], records[name] = _stats(
            frames, values, info, config, angular=angular,
            threshold=config.body_motion_diagonals_per_second if name == "body" else None)
    body = measurements["body"]
    candidates = []
    for name, label in (("body", "body_moving"), ("head_relative", "head_relative_motion"),
                        ("head_orientation", "head_orientation_change_in_image"),
                        ("head_orientation_relative_to_body", "head_orientation_change_relative_to_body"),
                        ("left_ear_orientation_relative_to_head", "left_ear_relative_motion"),
                        ("right_ear_orientation_relative_to_head", "right_ear_relative_motion"),
                        ("tail_relative", "tail_relative_motion"),
                        *[(n, f"{n}_relative_motion") for n in ALIASES if n.endswith("paw")]):
        stat = measurements[name]
        candidates.append(_candidate(
            label, "At least one sustained movement bout crosses the recorded speed threshold.", [stat],
            stat["movement_bout_count"] > 0, _motion_evidence(name, stat),
            ["Projected motion is not a confirmed behavior."] + (
                ["Tail-base subtraction removes translation, but body rotation can still move this vector."]
                if name == "tail_relative" else [])))
    candidates.append(_candidate(
        "body_stationary", "A continuous low-speed run stays within the configured radius for the minimum duration.",
        [body], (body["longest_stationary_duration_seconds"] or 0) + 1e-9 >= config.stationary_duration_seconds,
        {"measurement": "body", "longest_stationary_duration_seconds": body["longest_stationary_duration_seconds"],
         "minimum_duration_seconds": config.stationary_duration_seconds,
         "maximum_radius_diagonals": config.stationary_radius_diagonals,
         "speed_threshold_diagonals_per_second": config.body_motion_diagonals_per_second,
         "valid_frame_percentage": body["valid_frame_percentage"]},
        ["This describes a stationary interval, not stillness throughout the clip or inactivity of all body parts."] ))
    tail = measurements["tail_orientation_relative_to_body"]
    candidates.append(_candidate(
        "tail_oscillation_candidate", "Body-relative tail angle has enough reversals, excursion and sustained motion.",
        [tail], tail["angular_direction_reversals"] >= config.tail_minimum_reversals
        and (tail["maximum_excursion_degrees"] or 0) >= config.tail_minimum_excursion_degrees
        and tail["movement_bout_count"] > 0,
        {**_motion_evidence("tail_orientation_relative_to_body", tail),
         "angular_direction_reversals": tail["angular_direction_reversals"],
         "maximum_excursion_degrees": tail["maximum_excursion_degrees"],
         "minimum_reversals": config.tail_minimum_reversals,
         "minimum_excursion_degrees": config.tail_minimum_excursion_degrees},
        ["An exploratory oscillation cue; it does not establish tail swishing, emotion, or intent."]))
    paws = [measurements[n] for n in ("front_left_paw", "front_right_paw")]
    candidates.append(_candidate(
        "repetitive_front_paw_motion_candidate", "Both torso-relative front paws repeatedly change direction in sustained movement bouts.",
        paws, all(s["movement_direction_changes"] >= config.paw_minimum_direction_changes
                  and s["movement_bout_count"] > 0 for s in paws),
        {"front_left_paw": {**_motion_evidence("front_left_paw", paws[0]),
                            "direction_changes": paws[0]["movement_direction_changes"]},
         "front_right_paw": {**_motion_evidence("front_right_paw", paws[1]),
                             "direction_changes": paws[1]["movement_direction_changes"]},
         "minimum_direction_changes_per_paw": config.paw_minimum_direction_changes},
        ["Alternating paw contact is not measured; this does not confirm kneading."]))
    if not config.experimental_behavior_candidates:
        for candidate in candidates:
            candidate["interval_seconds"] = {"start": start, "end": end}
        return {"start_seconds": start, "end_seconds": end, "frames_processed": len(frames),
                "measurements": measurements, "candidates": candidates}
    head = measurements["head_relative"]
    # Evaluate proximity and head motion on the SAME observed pairs, not independent window averages.
    body_pairs = {r["i"]: r for r in records["body"]}
    joint = [r for r in records["head_relative"] if r["i"] in body_pairs
             and proximity[r["i"]] is not None and proximity[r["i"] - 1] is not None]
    near_and_moving = [r for r in joint if max(proximity[r["i"]], proximity[r["i"] - 1])
                       <= config.nose_paw_distance_body_lengths
                       and r["scaled"] >= config.relative_motion_diagonals_per_second
                       and body_pairs[r["i"]]["scaled"] < config.body_motion_diagonals_per_second]
    joint_seconds = sum(r["dt"] for r in joint)
    joint_percent = _percent(len(joint), max(0, len(frames) - 1))
    cooccurrence = sum(r["dt"] for r in near_and_moving) / joint_seconds if joint_seconds else None
    joint_quality = {"insufficient_data_reasons": []}
    if (len(joint) < config.minimum_valid_pairs or joint_percent / 100 < config.minimum_pair_coverage
            or joint_seconds + 1e-9 < config.minimum_observed_seconds):
        joint_quality["insufficient_data_reasons"].append("Insufficient simultaneous nose, paw and torso observations")
    candidates.append(_candidate(
        "grooming_like_motion_candidate",
        "Nose is near a front paw while moving relative to the torso for the required share of joint observed time, with a stationary body interval.",
        [body, head, joint_quality], cooccurrence is not None and cooccurrence >= config.grooming_minimum_cooccurrence
        and (body["longest_stationary_duration_seconds"] or 0) >= config.stationary_duration_seconds,
        {"cooccurrence_percentage": 100 * cooccurrence if cooccurrence is not None else None,
         "minimum_cooccurrence_percentage": 100 * config.grooming_minimum_cooccurrence,
         "maximum_nose_paw_distance_body_lengths": config.nose_paw_distance_body_lengths,
         "joint_valid_pair_count": len(joint), "joint_pair_coverage_percentage": joint_percent,
         "joint_observed_seconds": joint_seconds},
        ["2D proximity does not confirm contact, licking, or grooming; sniffing or other movements can resemble this cue."]))
    # Accept either reliable front paw; a hidden second paw must not veto an observable cue.
    usable_paws = [s for s in paws if s["data_sufficient"]]
    paw_quality = {"insufficient_data_reasons": [] if usable_paws else ["No front paw has sufficient relative-motion data"]}
    active_paw_pairs = {r["i"] for name in ("front_left_paw", "front_right_paw")
                        if measurements[name]["data_sufficient"] for r in records[name]
                        if r["scaled"] >= config.relative_motion_diagonals_per_second}
    joint_active_seconds = sum(r["dt"] for r in records["body"]
                               if r["i"] in active_paw_pairs
                               and r["scaled"] >= config.body_motion_diagonals_per_second)
    candidates.append(_candidate(
        "play_like_activity_candidate", "Body shows sustained movement and repeated direction changes, with concurrent movement of at least one reliable front paw.",
        [body, paw_quality], body["movement_bout_count"] > 0
        and body["movement_direction_changes"] >= config.play_minimum_direction_changes
        and any(s["movement_bout_count"] > 0 for s in usable_paws)
        and joint_active_seconds + 1e-9 >= config.minimum_bout_seconds,
        {"body": {**_motion_evidence("body", body), "direction_changes": body["movement_direction_changes"]},
         "reliable_front_paw_count": len(usable_paws),
         "minimum_body_direction_changes": config.play_minimum_direction_changes,
         "joint_active_seconds": joint_active_seconds,
         "minimum_joint_active_seconds": config.minimum_bout_seconds,
         "front_paw_movement_bout_counts": [s["movement_bout_count"] for s in usable_paws]},
        ["This exploratory activity pattern also occurs outside play; object, social and video context are absent."]))
    for candidate in candidates:
        candidate["interval_seconds"] = {"start": start, "end": end}
    return {"start_seconds": start, "end_seconds": end, "frames_processed": len(frames),
            "measurements": measurements, "candidates": candidates}


def _round(value):
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return round(value, 6) if isinstance(value, float) else value


def analyze_dataset(dataset, *, config=None, landmark_map=None):
    """Return a JSON-safe summary without AI calls or modifying the input dataset."""
    config = config or MovementConfig()
    if not isinstance(config, MovementConfig):
        raise TypeError("config must be MovementConfig")
    info, frames, available, warnings = _validate(dataset)
    if landmark_map is not None and (not isinstance(landmark_map, dict) or set(landmark_map) - ALIASES.keys()):
        raise ValueError("landmark_map must map known canonical names to source labels or None")
    mapping = {}
    for name, aliases in ALIASES.items():
        source = (landmark_map or {}).get(name, next((a for a in aliases if a in available), None))
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("Landmark sources must be nonempty strings or None")
        mapping[name] = source if source in available else None
    cleaned, smoothed, rejected, spikes = _clean(frames, mapping, info, config)
    features, proximity = _features(smoothed, info, config)
    whole = _window(frames, features, proximity, info, config, info["start"], info["end"])
    span = info["end"] - info["start"]
    interval_size = max(config.segment_seconds, span / config.max_intervals)
    count = min(config.max_intervals, max(1, math.ceil(span / interval_size)))
    groups = [[] for _ in range(count)]
    for i, frame in enumerate(frames):
        index = min(count - 1, int((frame["time"] - info["start"]) / interval_size))
        groups[index].append(i)
    intervals = []
    for i, indices in enumerate(groups):
        intervals.append(_window(
            [frames[j] for j in indices], {name: [values[j] for j in indices] for name, values in features.items()},
            [proximity[j] for j in indices], info, config,
            info["start"] + i * interval_size, min(info["end"], info["start"] + (i + 1) * interval_size)))
    coverage = {name: _percent(sum(p is not None for p in points), len(frames)) for name, points in cleaned.items()}
    gap_count = sum(not _adjacent(a, b, info, config) for a, b in zip(frames, frames[1:]))
    if gap_count:
        warnings.append(f"{gap_count} frame/time gaps excluded; missing movement is not imputed")
    if spikes:
        warnings.append(f"Removed {sum(spikes.values())} isolated high-speed landmark spikes before smoothing")
    sufficient = sum(m["data_sufficient"] for m in whole["measurements"].values())
    feature_count = len(whole["measurements"])
    rating = "good" if sufficient == feature_count else "limited" if sufficient else "poor"
    result = _round({
        "schema_version": SCHEMA_VERSION,
        "video": {"name": dataset["metadata"].get("original_video"), "fps": info["fps"],
                  "width": info["width"], "height": info["height"], "frames_received": len(frames),
                  "analysis_start_seconds": info["start"], "analysis_end_seconds": info["end"],
                  "duration_seconds": span, "assumed_animal_count": 1},
        "data_quality": {"rating": rating, "landmark_coverage_percentage": coverage,
                         "unavailable_landmarks": [n for n, source in mapping.items() if source is None],
                         "rejected_landmark_counts": dict(rejected), "isolated_spikes_removed": dict(spikes),
                         "frame_time_gaps": gap_count, "sufficient_feature_count": sufficient,
                         "feature_count": feature_count,
                         "rating_definition": "All/any/no features pass their own quality gates; consult each feature individually."},
        "measurements": whole["measurements"], "candidates": whole["candidates"], "intervals": intervals,
        "analysis_settings": {**asdict(config), "landmark_sources": mapping, "rule_version": RULE_VERSION,
                              "actual_interval_seconds": interval_size,
                              "smoothing": "Centered coordinate median within consecutive valid runs; missing values remain missing",
                              "body_reference": "Fixed midpoint of front and rear trunk landmarks; both required",
                              "head_and_paw_reference": "Torso-centred, rotated into the torso image-plane axis",
                              "ear_reference": "Ear angle minus head-axis angle",
                              "tail_reference": "Tip minus base for translation; angular oscillation measured relative to torso",
                              "displacement_definition": "Distance between consecutive valid observations, not endpoint displacement",
                              "quality_denominator": "All received frames/pairs in the interval; temporal gaps also reduce observed time"},
        "warnings": warnings, "limitations": LIMITATIONS[:],
    })
    json.dumps(result, allow_nan=False)
    return result


def validate_summary(summary):
    """Validate the summary contract before an AI prompt can consume it."""
    if not isinstance(summary, dict) or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected {SCHEMA_VERSION} summary")
    for field in ("video", "data_quality", "measurements", "analysis_settings"):
        if not isinstance(summary.get(field), dict):
            raise ValueError(f"summary.{field} must be an object")
    for field in ("candidates", "intervals", "warnings", "limitations"):
        if not isinstance(summary.get(field), list):
            raise ValueError(f"summary.{field} must be a list")
    if summary["data_quality"].get("rating") not in {"good", "limited", "poor"}:
        raise ValueError("Unknown data quality rating")
    for name, measurement in summary["measurements"].items():
        if not isinstance(measurement, dict) or type(measurement.get("data_sufficient")) is not bool:
            raise ValueError(f"measurement {name} needs a boolean data_sufficient field")
        if not isinstance(measurement.get("insufficient_data_reasons"), list):
            raise ValueError(f"measurement {name} needs insufficient_data_reasons")
    for interval in [summary, *summary["intervals"]]:
        if not isinstance(interval, dict) or not isinstance(interval.get("candidates"), list):
            raise ValueError("Each interval must have candidates")
        for candidate in interval["candidates"]:
            if not isinstance(candidate, dict) or candidate.get("status") not in {"detected", "not_detected", "insufficient_data"}:
                raise ValueError("Invalid candidate status")
            for field, kind in (("label", str), ("rule", str), ("rule_id", str), ("rule_version", str),
                                ("evidence", dict), ("interval_seconds", dict), ("limitations", list)):
                if not isinstance(candidate.get(field), kind):
                    raise ValueError(f"candidate.{field} has an invalid type")
            start, end = (_number(candidate["interval_seconds"].get(k)) for k in ("start", "end"))
            if start is None or end is None or start < 0 or end < start:
                raise ValueError("Candidate interval must have ordered finite nonnegative endpoints")
    json.dumps(summary, allow_nan=False)
    return summary


def analyze_file(input_path, output_path=None, *, config=None, landmark_map=None):
    """Analyze a saved dataset and save the summary when output_path is provided."""
    source = Path(input_path)
    if output_path is not None and Path(output_path).resolve() == source.resolve():
        raise ValueError("Summary output must differ from the input keypoint file")
    summary = analyze_dataset(json.loads(source.read_text(encoding="utf-8-sig")),
                              config=config, landmark_map=landmark_map)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def to_ai_evidence(summary):
    """Build a prompt-only view with shared rules and no repeated measurement tables.

    The saved movement-summary remains the public, validated input contract.
    This transport view is not a replacement for the saved full summary.
    """
    validate_summary(summary)
    rules = {c["rule_id"]: {"version": c["rule_version"], "definition": c["rule"]}
             for c in summary["candidates"]}
    limitations = {}

    def compact(candidate):
        refs = []
        for text in candidate["limitations"]:
            if text not in limitations:
                limitations[text] = f"L{len(limitations) + 1}"
            refs.append(limitations[text])
        return {"label": candidate["label"], "status": candidate["status"],
                "evidence": candidate["evidence"], "limitation_ids": refs}

    candidates = [compact(c) for c in summary["candidates"]]
    intervals = [{"start_seconds": i["start_seconds"], "end_seconds": i["end_seconds"],
                  "frames_processed": i["frames_processed"],
                  "candidates": [compact(c) for c in i["candidates"]]} for i in summary["intervals"]]
    return {"schema_version": "movement-evidence/1", "source_summary_schema": SCHEMA_VERSION,
            "video": summary["video"], "data_quality": summary["data_quality"],
            "measurements": summary["measurements"], "candidates": candidates, "intervals": intervals,
            "rules_by_label": rules, "candidate_limitations_by_id": {v: k for k, v in limitations.items()},
            "analysis_settings": summary["analysis_settings"], "warnings": summary["warnings"],
            "limitations": summary["limitations"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help=f"Summary file path; defaults to <video>_summary.json in {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--experimental-behaviors", action="store_true",
                        help="Include unvalidated grooming-like and play-like hypotheses")
    args = parser.parse_args(argv)
    output_path = args.output
    if output_path is None:
        video_name = args.input.stem.removesuffix("_keypoints")
        output_path = DEFAULT_OUTPUT_DIR / f"{video_name}_summary.json"
    try:
        analyze_file(args.input, output_path, config=MovementConfig(
            confidence_threshold=args.confidence, segment_seconds=args.segment_seconds,
            smoothing_window=args.smoothing_window,
            experimental_behavior_candidates=args.experimental_behaviors))
        print(f"Summary saved to {output_path}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Movement analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
