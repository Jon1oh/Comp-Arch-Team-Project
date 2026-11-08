"""Run from the project root: python -m unittest discover -s tests -v."""

import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Works in project/tests and alongside the downloadable logic_manager.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import logic_manager as logic


def make_dataset(count=30, fps=10, dx=0):
    points = {
        "nose": (180, 100), "left_eye": (165, 95), "right_eye": (165, 105),
        "left_ear_base": (150, 90), "right_ear_base": (150, 110),
        "left_ear_tip": (150, 70), "right_ear_tip": (150, 130),
        "shoulder": (140, 150), "spine_mid": (110, 150), "hip": (80, 150),
        "tail_base": (80, 150), "tail_mid": (60, 150), "tail_tip": (40, 150),
    }
    return {
        "metadata": {"fps": fps, "width": 640, "height": 480,
                     "frame_count": count, "duration_seconds": count / fps},
        "keypoint_names": list(points),
        "frames": [{
            "frame": index, "timestamp_seconds": index / fps,
            "keypoints": {name: {"x": x + dx * index, "y": y,
                                 "likelihood": 0.95, "valid": True}
                          for name, (x, y) in points.items()},
        } for index in range(count)],
    }


def analyze(dataset, **settings):
    return logic.analyze_dataset(dataset, config=logic.LogicConfig(smoothing_window=1, **settings))


class LogicManagerTests(unittest.TestCase):
    def test_stationary_geometry_and_quality(self):
        result = analyze(make_dataset())
        self.assertEqual(result["data_quality"]["rating"], "good")
        self.assertEqual(result["data_quality"]["reliable_detection_percentage"], 100)
        self.assertEqual(result["observed_features"]["body"]["movement"]["movement"], "low")
        self.assertEqual(result["observed_features"]["body"]["mean_spine_interior_angle_degrees"], 180)
        self.assertEqual(result["observed_features"]["head"]["orientation_in_image"]["mean_angle_degrees"], 0)

    def test_known_displacement_speed_and_normalization(self):
        result = analyze(make_dataset(dx=2))
        nose = result["movement_metrics"]["nose"]
        self.assertAlmostEqual(nose["average_displacement_pixels"], 2)
        self.assertAlmostEqual(nose["average_speed_pixels_per_second"], 20)
        self.assertAlmostEqual(nose["average_speed_diagonals_per_second"], 20 / 800)
        self.assertEqual(nose["valid_measurement_count"], 29)
        self.assertEqual(nose["movement"], "moderate")

    def test_resolution_invariance(self):
        dataset = make_dataset(dx=2)
        original = analyze(dataset)
        dataset["metadata"]["width"] *= 2
        dataset["metadata"]["height"] *= 2
        for frame in dataset["frames"]:
            for point in frame["keypoints"].values():
                point["x"] *= 2
                point["y"] *= 2
        scaled = analyze(dataset)
        self.assertEqual(original["movement_metrics"]["nose"]["average_speed_diagonals_per_second"],
                         scaled["movement_metrics"]["nose"]["average_speed_diagonals_per_second"])

    def test_translation_does_not_become_relative_tail_or_ear_motion(self):
        features = analyze(make_dataset(dx=2))["observed_features"]
        self.assertEqual(features["tail"]["movement"]["average_speed_pixels_per_second"], 0)
        self.assertEqual(features["ears"]["left"]["relative_movement"]["average_speed_pixels_per_second"], 0)
        self.assertGreater(features["body"]["movement"]["average_speed_pixels_per_second"], 0)

    def test_superanimal_names_and_trunk_proxies(self):
        dataset = make_dataset()
        rename = {"left_ear_base": "left_earbase", "right_ear_base": "right_earbase",
                  "left_ear_tip": "left_earend", "right_ear_tip": "right_earend",
                  "tail_tip": "tail_end", "spine_mid": "back_middle",
                  "shoulder": "back_base", "hip": "back_end"}
        dataset["keypoint_names"] = [rename.get(name, name) for name in dataset["keypoint_names"]]
        for frame in dataset["frames"]:
            frame["keypoints"] = {rename.get(name, name): value for name, value in frame["keypoints"].items()}
            frame["keypoints"]["left_antler_end"] = {"x": 0, "y": 0, "likelihood": 0.99, "valid": True}
        result = analyze(dataset)
        self.assertEqual(result["analysis_settings"]["keypoint_mapping"]["tail_tip"], "tail_end")
        self.assertEqual(result["data_quality"]["mapped_keypoint_count"], 13)
        self.assertTrue(any("trunk proxies" in line for line in result["limitations"]))

    def test_standardized_metadata(self):
        dataset = make_dataset()
        dataset["metadata"] = {"video_fps": 10, "video_width": 640, "video_height": 480,
                               "video_duration_seconds": 3, "video_frame_count": 30,
                               "video_name": "cat.mp4", "animal_count": 1}
        self.assertEqual(analyze(dataset)["video_summary"]["video_name"], "cat.mp4")

    def test_invalid_points_are_filtered_without_losing_other_parts(self):
        dataset = make_dataset()
        bad_points = [
            {"likelihood": 0.59}, {"likelihood": -1}, {"likelihood": 1.1},
            {"x": None}, {"x": float("nan")}, {"y": float("inf")},
            {"x": -1}, {"x": 640}, {"valid": False}, {"x": True},
            {"valid": "false"}, {"x": "12"},
        ]
        for frame, changes in zip(dataset["frames"], bad_points):
            frame["keypoints"]["nose"].update(changes)
        result = analyze(dataset)
        self.assertEqual(result["data_quality"]["keypoint_coverage_percentage"]["nose"], 60)
        self.assertEqual(result["data_quality"]["keypoint_coverage_percentage"]["tail_tip"], 100)
        self.assertEqual(result["video_summary"]["frames_processed"], 30)
        json.dumps(result, allow_nan=False)

    def test_source_invalid_flag_is_respected(self):
        dataset = make_dataset()
        for frame in dataset["frames"]:
            frame["keypoints"]["nose"]["valid"] = False
        self.assertEqual(analyze(dataset, confidence_threshold=0.1)["observed_features"]["head"]["movement"], "unknown")

    def test_missing_flag_can_be_recomputed_from_measurements(self):
        dataset = make_dataset()
        for frame in dataset["frames"]:
            del frame["keypoints"]["nose"]["valid"]
        self.assertEqual(analyze(dataset)["data_quality"]["keypoint_coverage_percentage"]["nose"], 100)

    def test_missing_tail_is_unknown_and_excluded_from_quality_denominator(self):
        dataset = make_dataset()
        dataset["keypoint_names"].remove("tail_tip")
        for frame in dataset["frames"]:
            del frame["keypoints"]["tail_tip"]
        result = analyze(dataset)
        self.assertEqual(result["observed_features"]["tail"]["movement"]["movement"], "unknown")
        self.assertEqual(result["data_quality"]["reliable_detection_percentage"], 100)
        self.assertIn("tail_tip", result["data_quality"]["missing_keypoints"])

    def test_all_invalid_points_and_one_frame(self):
        dataset = make_dataset(count=1)
        for point in dataset["frames"][0]["keypoints"].values():
            point["valid"] = False
        result = analyze(dataset)
        self.assertEqual(result["data_quality"]["rating"], "poor")
        self.assertIsNone(result["movement_metrics"]["nose"]["average_displacement_pixels"])
        self.assertEqual(result["observed_features"]["head"]["movement"], "unknown")

    def test_no_known_landmarks(self):
        dataset = make_dataset(count=3)
        dataset["keypoint_names"] = []
        for frame in dataset["frames"]:
            frame["keypoints"] = {}
        result = analyze(dataset)
        self.assertEqual(result["data_quality"]["rating"], "poor")
        self.assertEqual(result["data_quality"]["mapped_keypoint_count"], 0)

    def test_smoothing_cannot_fill_or_cross_a_detection_gap(self):
        dataset = make_dataset(count=7)
        for index, frame in enumerate(dataset["frames"]):
            frame["keypoints"]["nose"]["x"] = 100 if index < 3 else 300
        dataset["frames"][3]["keypoints"]["nose"]["valid"] = False
        result = logic.analyze_dataset(dataset)
        self.assertEqual(result["movement_metrics"]["nose"]["maximum_displacement_pixels"], 0)
        self.assertEqual(result["movement_metrics"]["nose"]["valid_measurement_count"], 4)

    def test_smoothing_reduces_alternating_noise(self):
        dataset = make_dataset()
        for index, frame in enumerate(dataset["frames"]):
            frame["keypoints"]["nose"]["x"] += 5 * (-1) ** index
        raw = analyze(dataset)["movement_metrics"]["nose"]["average_displacement_pixels"]
        smooth = logic.analyze_dataset(dataset)["movement_metrics"]["nose"]["average_displacement_pixels"]
        self.assertLess(smooth, raw)

    def test_frame_and_timestamp_gaps_excluded(self):
        dataset = make_dataset(count=6, dx=2)
        del dataset["frames"][2]
        result = analyze(dataset)
        self.assertEqual(result["movement_metrics"]["nose"]["valid_measurement_count"], 3)
        self.assertTrue(any("frame/time gaps" in warning for warning in result["warnings"]))
        dataset = make_dataset(count=6)
        for frame in dataset["frames"][3:]:
            frame["timestamp_seconds"] += 2
            frame["keypoints"]["nose"]["x"] += 200
        result = logic.analyze_dataset(dataset)
        self.assertEqual(result["movement_metrics"]["nose"]["maximum_displacement_pixels"], 0)

    def test_time_weighted_speed(self):
        dataset = make_dataset(count=4)
        for frame, timestamp in zip(dataset["frames"], [0, 0.05, 0.15, 0.30]):
            frame["timestamp_seconds"] = timestamp
            frame["keypoints"]["nose"]["x"] = 180 + 20 * timestamp
        result = analyze(dataset, maximum_gap_factor=1.6)
        self.assertAlmostEqual(result["movement_metrics"]["nose"]["average_speed_pixels_per_second"], 20)

    def test_circular_angle_wrap_is_small(self):
        dataset = make_dataset(count=3)
        for frame, angle in zip(dataset["frames"], [179, -179, 179]):
            radians = math.radians(angle)
            frame["keypoints"]["tail_tip"].update(x=80 + 40 * math.cos(radians), y=150 + 40 * math.sin(radians))
        result = analyze(dataset)["observed_features"]["tail"]["orientation_in_image"]
        self.assertAlmostEqual(result["average_angular_speed_degrees_per_second"], 20)
        self.assertGreater(abs(result["mean_angle_degrees"]), 170)

    def test_angular_reversal_rate(self):
        dataset = make_dataset(count=5)
        for frame, angle in zip(dataset["frames"], [0, 10, 0, -10, 0]):
            radians = math.radians(angle)
            frame["keypoints"]["tail_tip"].update(x=80 + 40 * math.cos(radians), y=150 + 40 * math.sin(radians))
        metrics = analyze(dataset)["observed_features"]["tail"]["orientation_in_image"]
        self.assertEqual(metrics["angular_direction_changes"], 2)
        self.assertEqual(metrics["angular_direction_changes_per_second"], 5)

    def test_temporal_change_and_partial_last_window(self):
        dataset = make_dataset(count=65)
        for index, frame in enumerate(dataset["frames"]):
            if index >= 30:
                frame["keypoints"]["tail_tip"]["x"] += 10 * (-1) ** index
        result = analyze(dataset)
        windows = result["temporal_patterns"]
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[-1]["end_seconds"], 6.5)
        self.assertIn("tail_activity_increased", windows[1]["changes_from_previous_interval"])
        self.assertEqual(sum(window["frames_processed"] for window in windows), 65)

    def test_custom_mapping(self):
        dataset = make_dataset()
        for frame in dataset["frames"]:
            frame["keypoints"]["snout"] = frame["keypoints"].pop("nose")
        dataset["keypoint_names"].remove("nose")
        result = logic.analyze_dataset(dataset, keypoint_map={"nose": "snout", "tail_mid": None})
        self.assertEqual(result["analysis_settings"]["keypoint_mapping"]["nose"], "snout")
        self.assertIsNone(result["analysis_settings"]["keypoint_mapping"]["tail_mid"])

    def test_input_is_not_mutated(self):
        dataset = make_dataset()
        before = copy.deepcopy(dataset)
        logic.analyze_dataset(dataset)
        self.assertEqual(before, dataset)

    def test_malformed_input_is_rejected(self):
        variants = [None, {}, {"metadata": {}, "frames": []}]
        for field, value in [("fps", 0), ("fps", float("nan")), ("width", -2), ("animal_count", 2)]:
            dataset = make_dataset()
            dataset["metadata"][field] = value
            variants.append(dataset)
        for change in ("duplicate_time", "backward_frame", "bad_points"):
            dataset = make_dataset()
            if change == "duplicate_time":
                dataset["frames"][1]["timestamp_seconds"] = 0
            elif change == "backward_frame":
                dataset["frames"][1]["frame"] = 0
            else:
                dataset["frames"][1]["keypoints"] = []
            variants.append(dataset)
        for dataset in variants:
            with self.subTest(dataset=str(dataset)[:60]), self.assertRaises(ValueError):
                logic.analyze_dataset(dataset)

    def test_invalid_configuration_and_mapping(self):
        for kwargs in ({"smoothing_window": 2}, {"segment_seconds": 0},
                       {"confidence_threshold": 1.1}, {"minimum_coverage": 0},
                       {"maximum_gap_factor": 0.5}, {"low_speed_threshold": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                logic.LogicConfig(**kwargs)
        for mapping in ({"unknown": "nose"}, {"nose": 123}, ["nose"]):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                logic.analyze_dataset(make_dataset(), keypoint_map=mapping)

    def test_json_roundtrip_and_input_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.json", Path(folder) / "summary.json"
            source.write_text(json.dumps(make_dataset()), encoding="utf-8-sig")
            result = logic.analyze_file(source, target)
            self.assertEqual(result, json.loads(target.read_text(encoding="utf-8")))
            with self.assertRaises(ValueError):
                logic.analyze_file(source, source)
            source.write_text("{ broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                logic.analyze_file(source)

    def test_cli_success_and_error(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.json", Path(folder) / "summary.json"
            source.write_text(json.dumps(make_dataset()), encoding="utf-8")
            command = [sys.executable, "-B", str(Path(logic.__file__)), str(source), "--output", str(target)]
            success = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout), json.loads(target.read_text(encoding="utf-8")))
            failure = subprocess.run(command + ["--smoothing-window", "2"], capture_output=True, text=True)
            self.assertEqual(failure.returncode, 1)
            self.assertIn("positive odd", failure.stderr)


if __name__ == "__main__":
    unittest.main()
