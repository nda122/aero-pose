import numpy as np
import pytest

from aero_pose.detection.yolo_pose import YOLOPoseDetector, COCO_KEYPOINTS


def test_coco_keypoints_count() -> None:
    assert len(COCO_KEYPOINTS) == 17


@pytest.mark.skip(reason="Requires webcam or video file")
def test_detector_initialization() -> None:
    detector = YOLOPoseDetector(model_size="n", device="cpu")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy)
    assert isinstance(detections, list)
