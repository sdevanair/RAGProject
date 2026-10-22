import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.motion_detection import (
    box_iou,
    is_duplicate_detection,
    load_yolo_model,
    open_video,
)


class DetectionDeduplicationTests(unittest.TestCase):
    def test_identical_boxes_have_full_overlap(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes_have_no_overlap(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [11, 11, 20, 20]), 0.0)

    def test_same_class_box_within_cooldown_is_duplicate(self):
        recent = [
            {
                "object_class": "person",
                "bounding_box": [10, 10, 40, 60],
                "_video_time": 3.0,
            }
        ]
        self.assertTrue(
            is_duplicate_detection("person", [11, 10, 41, 60], 4.0, recent)
        )

    def test_different_class_or_expired_event_is_not_duplicate(self):
        recent = [
            {
                "object_class": "person",
                "bounding_box": [10, 10, 40, 60],
                "_video_time": 1.0,
            }
        ]
        self.assertFalse(
            is_duplicate_detection("car", [10, 10, 40, 60], 2.0, recent)
        )
        self.assertFalse(
            is_duplicate_detection("person", [10, 10, 40, 60], 7.0, recent)
        )


class InputValidationTests(unittest.TestCase):
    def test_missing_video_raises_clear_error(self):
        with TemporaryDirectory() as directory:
            missing_video = Path(directory) / "missing.mp4"
            with self.assertRaisesRegex(FileNotFoundError, "Video not found"):
                open_video(missing_video)

    def test_unreadable_video_raises_clear_error(self):
        with TemporaryDirectory() as directory:
            empty_video = Path(directory) / "empty.mp4"
            empty_video.touch()
            with self.assertRaisesRegex(RuntimeError, "could not open video|could not read a frame"):
                open_video(empty_video)

    def test_missing_model_weights_raise_clear_error(self):
        with TemporaryDirectory() as directory:
            missing_weights = Path(directory) / "missing.pt"
            with self.assertRaisesRegex(FileNotFoundError, "YOLO weights not found"):
                load_yolo_model(missing_weights)


if __name__ == "__main__":
    unittest.main()
