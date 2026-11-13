"""Synthetic ground truth for motion, observability, geometry and AI routing."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from logic.movement_summary import (
    MovementConfig, SCHEMA_VERSION, analyze_dataset, analyze_file, validate_summary,
)


def dataset(count=60, fps=20, transform=None):
    locations = {
        "back_base": (400, 400), "back_end": (500, 400),
        "nose": (350, 400), "left_earbase": (390, 385), "right_earbase": (390, 415),
        "left_earend": (380, 365), "right_earend": (380, 435),
        "tail_base": (500, 400), "tail_end": (560, 400),
        "front_left_paw": (420, 460), "front_right_paw": (420, 480),
        "back_left_paw": (490, 460), "back_right_paw": (490, 480),
    }
    frames = []
    for i in range(count):
        positions = dict(locations)
        if transform:
            transform(i, positions)
        points = {name: {"x": xy[0], "y": xy[1], "likelihood": 0.95, "valid": True}
                  for name, xy in positions.items()}
        frames.append({"frame": i, "timestamp_seconds": i / fps, "keypoints": points})
    return {"metadata": {"original_video": "synthetic", "fps": fps, "width": 1000,
                         "height": 800, "duration_seconds": count / fps, "frame_count": count},
            "keypoint_names": list(locations), "frames": frames}


def candidate(summary, label):
    return next(c for c in summary["candidates"] if c["label"] == label)


class MovementSummaryTests(unittest.TestCase):
    def test_stationary_ground_truth_has_zero_speed_and_sustained_stillness(self):
        raw = dataset()
        untouched = deepcopy(raw)
        summary = analyze_dataset(raw)
        stats = summary["measurements"]["body"]
        self.assertEqual(stats["average_speed_pixels_per_second"], 0)
        self.assertEqual(stats["valid_measurement_count"], 59)
        self.assertEqual(stats["observed_pair_seconds"], 2.95)
        self.assertEqual(candidate(summary, "body_stationary")["status"], "detected")
        self.assertEqual(candidate(summary, "body_moving")["status"], "not_detected")
        self.assertEqual(raw, untouched)
        self.assertIs(validate_summary(summary), summary)

    def test_shared_translation_is_body_motion_not_relative_head_or_paw_motion(self):
        def translate(i, p):
            for name, (x, y) in list(p.items()):
                p[name] = (x + 3 * i, y)
        summary = analyze_dataset(dataset(transform=translate), config=MovementConfig(smoothing_window=1))
        self.assertAlmostEqual(summary["measurements"]["body"]["average_speed_pixels_per_second"], 60)
        self.assertEqual(candidate(summary, "body_moving")["status"], "detected")
        self.assertEqual(candidate(summary, "head_relative_motion")["status"], "not_detected")
        self.assertEqual(candidate(summary, "front_left_paw_relative_motion")["status"], "not_detected")

    def test_rotation_does_not_become_independent_ear_or_paw_movement(self):
        def rotate(i, p):
            angle = i / 100
            for name, (x, y) in list(p.items()):
                x, y = x - 450, y - 400
                p[name] = (450 + x * math.cos(angle) - y * math.sin(angle),
                           400 + x * math.sin(angle) + y * math.cos(angle))
        summary = analyze_dataset(dataset(transform=rotate), config=MovementConfig(smoothing_window=1))
        for name in ("left_ear_orientation_relative_to_head", "head_orientation_relative_to_body"):
            self.assertAlmostEqual(summary["measurements"][name]["average_angular_speed_degrees_per_second"], 0)
        self.assertAlmostEqual(summary["measurements"]["front_left_paw"]["average_speed_pixels_per_second"], 0)
        self.assertGreater(summary["measurements"]["head_orientation"]["average_angular_speed_degrees_per_second"], 0)

    def test_head_turn_is_measured_separately_from_body(self):
        def turn(i, p):
            angle = math.pi + 0.5 * math.sin(i / 5)
            p["nose"] = (390 + 40 * math.cos(angle), 400 + 40 * math.sin(angle))
        summary = analyze_dataset(dataset(transform=turn), config=MovementConfig(smoothing_window=1))
        self.assertEqual(candidate(summary, "head_orientation_change_relative_to_body")["status"], "detected")
        self.assertEqual(candidate(summary, "body_stationary")["status"], "detected")

    def test_angle_wrap_does_not_create_an_artificial_360_degree_jump(self):
        def turn(i, p):
            angle = math.radians(170 + 0.5 * i)
            p["nose"] = (390 + 40 * math.cos(angle), 400 + 40 * math.sin(angle))
        summary = analyze_dataset(dataset(transform=turn), config=MovementConfig(smoothing_window=1))
        self.assertAlmostEqual(summary["measurements"]["head_orientation"]["maximum_angular_speed_degrees_per_second"], 10)

    def test_degenerate_trunk_axis_does_not_support_body_relative_claims(self):
        def collapse(i, p):
            p["back_end"] = p["back_base"]
        summary = analyze_dataset(dataset(transform=collapse))
        self.assertEqual(candidate(summary, "body_stationary")["status"], "insufficient_data")
        self.assertEqual(candidate(summary, "front_left_paw_relative_motion")["status"], "insufficient_data")
        self.assertTrue(summary["measurements"]["head_orientation"]["data_sufficient"])

    def test_tail_oscillation_counts_reversals_across_slow_turnaround(self):
        def oscillate(i, p):
            angle = 0.6 * math.sin(i * math.pi / 10)
            p["tail_end"] = (500 + 60 * math.cos(angle), 400 + 60 * math.sin(angle))
        summary = analyze_dataset(dataset(transform=oscillate), config=MovementConfig(smoothing_window=1))
        tail = summary["measurements"]["tail_orientation_relative_to_body"]
        self.assertGreaterEqual(tail["angular_direction_reversals"], 3)
        self.assertEqual(candidate(summary, "tail_oscillation_candidate")["status"], "detected")

    def test_missing_tail_is_unknown_not_stationary(self):
        raw = dataset()
        for frame in raw["frames"]:
            del frame["keypoints"]["tail_end"]
        summary = analyze_dataset(raw)
        self.assertEqual(candidate(summary, "tail_relative_motion")["status"], "insufficient_data")
        self.assertIsNone(summary["measurements"]["tail_relative"]["average_speed_pixels_per_second"])
        self.assertEqual(candidate(summary, "body_stationary")["status"], "detected")

    def test_disconnected_good_frames_do_not_support_tail_rule(self):
        raw = dataset()
        for i, frame in enumerate(raw["frames"]):
            if i % 2:
                frame["keypoints"]["tail_end"]["valid"] = False
        summary = analyze_dataset(raw, config=MovementConfig(minimum_frame_coverage=0.5))
        self.assertEqual(summary["measurements"]["tail_relative"]["valid_measurement_count"], 0)
        self.assertEqual(candidate(summary, "tail_oscillation_candidate")["status"], "insufficient_data")

    def test_exporter_flag_bounds_and_nonfinite_checks_are_independent(self):
        raw = dataset()
        points = [f["keypoints"]["nose"] for f in raw["frames"]]
        points[0]["x"] = 1200  # High confidence cannot rescue an out-of-image point.
        points[1]["valid"] = False
        points[2]["likelihood"] = 0.1
        points[3]["y"] = float("nan")
        summary = analyze_dataset(raw)
        counts = summary["data_quality"]["rejected_landmark_counts"]
        for reason in ("out_of_bounds", "exporter_invalid", "confidence", "nonfinite_or_missing_number"):
            self.assertEqual(counts[reason], 1)
        json.dumps(summary, allow_nan=False)

    def test_large_time_gap_does_not_support_whole_clip_stillness(self):
        raw = dataset()
        for frame in raw["frames"][30:]:
            frame["timestamp_seconds"] += 8
        summary = analyze_dataset(raw)
        self.assertEqual(summary["data_quality"]["frame_time_gaps"], 1)
        self.assertEqual(candidate(summary, "body_stationary")["status"], "insufficient_data")
        self.assertTrue(any(i["frames_processed"] == 0 for i in summary["intervals"]))

    def test_short_burst_is_not_a_sustained_movement_bout(self):
        def blip(i, p):
            if i >= 20:
                for name, (x, y) in list(p.items()):
                    p[name] = (x + 5, y)
        summary = analyze_dataset(dataset(transform=blip), config=MovementConfig(smoothing_window=1))
        self.assertEqual(summary["measurements"]["body"]["movement_bout_count"], 0)
        self.assertEqual(candidate(summary, "body_moving")["status"], "not_detected")
        self.assertGreater(summary["measurements"]["body"]["maximum_speed_pixels_per_second"], 0)

    def test_missing_torso_anchor_does_not_switch_body_definition(self):
        raw = dataset()
        for i, frame in enumerate(raw["frames"]):
            if i % 2:
                del frame["keypoints"]["back_end"]
        summary = analyze_dataset(raw)
        self.assertEqual(summary["measurements"]["body"]["valid_measurement_count"], 0)

    def test_isolated_high_confidence_spike_is_removed(self):
        raw = dataset()
        raw["frames"][20]["keypoints"]["nose"]["x"] = 950
        summary = analyze_dataset(raw)
        self.assertEqual(summary["data_quality"]["isolated_spikes_removed"]["nose"], 1)
        self.assertEqual(candidate(summary, "head_relative_motion")["status"], "not_detected")

    def test_resize_preserves_normalized_speed(self):
        def move(i, p):
            for name, (x, y) in list(p.items()):
                p[name] = (x + 3 * i, y)
        raw = dataset(transform=move)
        resized = deepcopy(raw)
        resized["metadata"]["width"] *= 2
        resized["metadata"]["height"] *= 2
        for frame in resized["frames"]:
            for point in frame["keypoints"].values():
                point["x"] *= 2
                point["y"] *= 2
        a, b = analyze_dataset(raw), analyze_dataset(resized)
        self.assertAlmostEqual(a["measurements"]["body"]["average_speed_diagonals_per_second"],
                               b["measurements"]["body"]["average_speed_diagonals_per_second"])

    def test_all_unobservable_and_short_clips_keep_nulls_and_quality(self):
        raw = dataset(count=2)
        for frame in raw["frames"]:
            frame["keypoints"] = {}
        summary = analyze_dataset(raw)
        self.assertEqual(summary["data_quality"]["rating"], "poor")
        self.assertTrue(all(c["status"] == "insufficient_data" for c in summary["candidates"]))
        self.assertTrue(all(c["limitations"] for c in summary["candidates"]))

    def test_detected_candidate_always_retains_interpretation_limits(self):
        summary = analyze_dataset(dataset())
        self.assertTrue(all(c["limitations"] for c in summary["candidates"]))
        self.assertTrue(all(c["score"] is None for c in summary["candidates"]))

    def test_behavior_hypotheses_require_explicit_opt_in(self):
        summary = analyze_dataset(dataset())
        self.assertNotIn("play_like_activity_candidate", [c["label"] for c in summary["candidates"]])

    def test_play_cue_can_use_one_reliable_front_paw(self):
        def activity(i, p):
            t = i if i < 20 else 40 - i if i < 40 else i - 40
            for name, (x, y) in list(p.items()):
                p[name] = (x + 4 * t, y)
            x, y = p["front_left_paw"]
            p["front_left_paw"] = (x + 15 * math.sin(i * math.pi / 5), y)
            del p["front_right_paw"]
        summary = analyze_dataset(dataset(transform=activity), config=MovementConfig(
            smoothing_window=1, experimental_behavior_candidates=True))
        play = candidate(summary, "play_like_activity_candidate")
        self.assertEqual(play["status"], "detected")
        self.assertEqual(play["evidence"]["reliable_front_paw_count"], 1)
        self.assertTrue(play["limitations"])

    def test_grooming_cue_requires_proximity_and_motion_at_same_time(self):
        def separated(i, p):
            if i < 20:
                p["front_left_paw"] = (350, 400)
            else:
                p["nose"] = (350 + 8 * math.sin(i * math.pi / 8), 400)
        summary = analyze_dataset(dataset(transform=separated), config=MovementConfig(
            smoothing_window=1, experimental_behavior_candidates=True))
        self.assertEqual(candidate(summary, "head_relative_motion")["status"], "detected")
        self.assertEqual(candidate(summary, "grooming_like_motion_candidate")["status"], "not_detected")

    def test_grooming_cue_detects_simultaneous_proximity_and_relative_motion(self):
        def coordinated(i, p):
            p["nose"] = (350 + 8 * math.sin(i * math.pi / 8), 400)
            p["front_left_paw"] = (350, 400)
        summary = analyze_dataset(dataset(transform=coordinated), config=MovementConfig(
            smoothing_window=1, experimental_behavior_candidates=True))
        cue = candidate(summary, "grooming_like_motion_candidate")
        self.assertEqual(cue["status"], "detected")
        self.assertTrue(any("licking" in text for text in cue["limitations"]))

    def test_invalid_dataset_config_and_unknown_schema_are_rejected(self):
        for changes in ({"fps": 0}, {"animal_count": 2}):
            raw = dataset()
            raw["metadata"].update(changes)
            with self.assertRaises(ValueError):
                analyze_dataset(raw)
        raw = dataset()
        raw["frames"][1]["timestamp_seconds"] = 0
        with self.assertRaises(ValueError):
            analyze_dataset(raw)
        with self.assertRaises(ValueError):
            MovementConfig(smoothing_window=2)
        with self.assertRaises(ValueError):
            validate_summary({"schema_version": SCHEMA_VERSION})

    def test_file_export_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.json"
            source.write_text(json.dumps(dataset()), encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze_file(source, source)
            target = Path(temp) / "nested" / "summary.json"
            summary = analyze_file(source, target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), summary)

    def test_ai_accepts_raw_and_summary_without_calling_api(self):
        from ai.ai import build_prompt
        raw = dataset()
        summary = analyze_dataset(raw)
        for data in (raw, summary):
            prompt = build_prompt(data)
            self.assertIn(SCHEMA_VERSION, prompt)
            self.assertIn("not confirmed behaviors", prompt)
            self.assertIn("body_stationary", prompt)
            self.assertIn("candidate_limitations_by_id", prompt)
        from ai.behavior_preprocessor import prepare_dataset
        self.assertIn("behavior-summary/1", build_prompt(prepare_dataset(raw)))
        with self.assertRaisesRegex(ValueError, "Unsupported summary schema"):
            build_prompt({"schema_version": "1.1"})

    def test_existing_sample_clips_produce_finite_valid_summaries(self):
        root = Path(__file__).resolve().parents[1] / "analysis" / "DeepLabCut-Processing" / "outputs"
        for path in sorted(root.glob("*_keypoints.json")):
            with self.subTest(video=path.name):
                summary = analyze_file(path)
                validate_summary(summary)
                self.assertGreater(summary["video"]["frames_received"], 0)
                self.assertLessEqual(len(summary["intervals"]), 60)


if __name__ == "__main__":
    unittest.main()
