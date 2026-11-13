import unittest

from ai.behavior_preprocessor import PreprocessConfig, prepare_dataset


def make_dataset(*, moving=True, include_tail=True):
    frames = []
    for index in range(12):
        offset = index * 2 if moving else 0
        points = {
            "back_base": {"x": 100 + offset, "y": 100, "likelihood": 0.95, "valid": True},
            "back_middle": {"x": 120 + offset, "y": 100, "likelihood": 0.95, "valid": True},
            "back_end": {"x": 140 + offset, "y": 100, "likelihood": 0.95, "valid": True},
            "nose": {"x": 90 + offset, "y": 95, "likelihood": 0.9, "valid": True},
        }
        if include_tail:
            points["tail_base"] = {"x": 140 + offset, "y": 100, "likelihood": 0.9, "valid": True}
            points["tail_end"] = {
                "x": 150 + offset + (index % 2) * 5,
                "y": 100,
                "likelihood": 0.9,
                "valid": True,
            }
        for name, x in (
            ("front_left_paw", 105),
            ("front_right_paw", 110),
            ("back_left_paw", 130),
            ("back_right_paw", 135),
        ):
            points[name] = {
                "x": x + offset + index % 2,
                "y": 120,
                "likelihood": 0.9,
                "valid": True,
            }
        frames.append({
            "frame": index,
            "timestamp_seconds": index / 10,
            "keypoints": points,
        })
    return {
        "metadata": {
            "original_video": "synthetic",
            "fps": 10,
            "duration_seconds": 1.2,
            "width": 320,
            "height": 240,
        },
        "keypoint_names": list(frames[0]["keypoints"]),
        "frames": frames,
    }


class BehaviorPreprocessorTests(unittest.TestCase):
    def test_detects_body_displacement_from_synthetic_motion(self):
        summary = prepare_dataset(make_dataset())
        body_candidate = summary["intervals"][0]["candidates"][0]

        self.assertEqual(body_candidate["label"], "body_displacement")
        self.assertEqual(body_candidate["status"], "detected")
        self.assertGreater(body_candidate["median_speed_image_diagonals_per_second"], 0)

    def test_stationary_body_is_not_detected_as_displacement(self):
        summary = prepare_dataset(make_dataset(moving=False))
        body_candidate = summary["intervals"][0]["candidates"][0]

        self.assertEqual(body_candidate["status"], "not_detected")

    def test_missing_tail_is_reported_as_insufficient_data(self):
        summary = prepare_dataset(make_dataset(include_tail=False))
        tail_candidate = summary["intervals"][0]["candidates"][2]

        self.assertEqual(tail_candidate["status"], "insufficient_data")
        self.assertIn("tail_base", summary["data_quality"]["unavailable_landmarks"])
        self.assertIn("tail_tip", summary["data_quality"]["unavailable_landmarks"])

    def test_low_confidence_points_do_not_count_as_valid(self):
        dataset = make_dataset()
        dataset["frames"][0]["keypoints"]["back_base"]["likelihood"] = 0.1
        summary = prepare_dataset(dataset, config=PreprocessConfig(confidence_threshold=0.6))

        self.assertLess(
            summary["data_quality"]["landmark_coverage_percent"]["torso_front"],
            100,
        )

    def test_invalid_fps_is_rejected(self):
        dataset = make_dataset()
        dataset["metadata"]["fps"] = 0

        with self.assertRaisesRegex(ValueError, "metadata.fps"):
            prepare_dataset(dataset)


if __name__ == "__main__":
    unittest.main()
