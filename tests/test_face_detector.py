import numpy as np
import pytest

from neurovector_vision.config import Settings
from neurovector_vision.face_detector import FaceDistanceDetector


class FakeBox:
    def __init__(self, confidence: float = 0.9) -> None:
        self.conf = [np.array(confidence)]


class FakeKeypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class FakeResult:
    orig_shape = (480, 640)

    def __init__(self, keypoints: np.ndarray) -> None:
        self.keypoints = FakeKeypoints(keypoints)
        self.boxes = [FakeBox()]


class FakeModel:
    def __init__(self, keypoints: np.ndarray) -> None:
        self._keypoints = keypoints

    def predict(self, source, **_kwargs):
        return [FakeResult(self._keypoints)]


class FakeDepth:
    last_inference_ms = 4.2

    def estimate(self, source):
        return np.ones((480, 640), dtype=np.float32)

    def median_depth(self, depth_map, box):
        return 0.7


def keypoints() -> np.ndarray:
    data = np.zeros((1, 17, 3), dtype=np.float64)
    data[0, 0] = (131.0, 130.0, 0.95)  # nose
    data[0, 1] = (100.0, 100.0, 0.95)  # left eye
    data[0, 2] = (163.0, 100.0, 0.95)  # right eye
    data[0, 3] = (60.0, 100.0, 0.90)  # left ear
    data[0, 4] = (203.0, 100.0, 0.90)  # right ear
    return data


def detector(*, depth: bool = False) -> FaceDistanceDetector:
    settings = Settings(
        _env_file=None,
        depth_enabled=False,
        camera_focal_length_px=1000.0,
        face_keypoint_confidence=0.5,
        confidence_threshold=0.35,
    )
    instance = FaceDistanceDetector(settings)
    instance._model = FakeModel(keypoints())
    if depth:
        instance._depth = FakeDepth()
    return instance


def test_face_detector_builds_face_box_and_keypoint_distance() -> None:
    measurements = detector().detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.detection.label == "face"
    assert measurement.detection.distance_m == pytest.approx(1.0)
    assert measurement.distance_source == "keypoint"
    assert measurement.eye_distance_px == pytest.approx(63.0)
    assert measurement.detection.box.x1 < 100 < measurement.detection.box.x2
    assert measurement.detection.box.y1 < 100 < measurement.detection.box.y2


def test_face_detector_prefers_depth_model() -> None:
    measurements = detector(depth=True).detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))

    measurement = measurements[0]
    assert measurement.raw_depth_m == pytest.approx(0.7)
    assert measurement.detection.distance_m == pytest.approx(0.7)
    assert measurement.distance_source == "depth"


def test_face_detector_calibrates_ipd_and_depth_scale() -> None:
    instance = detector(depth=True)
    measurement = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))[0]

    result = instance.calibrate(0.7, measurement, image_width=640)

    assert result["ipd_scale_px_m"] == pytest.approx(44.1)
    assert result["depth_scale"] == pytest.approx(1.0)
    assert "NEUROVECTOR_FACE_IPD_SCALE_PX_M=44.10" in result["recommended_env"]
