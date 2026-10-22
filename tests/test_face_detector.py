import numpy as np
import pytest

from neurovector_vision.config import Settings
from neurovector_vision.distance import CameraIntrinsics
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

    def median_depth_around_points(self, depth_map, points, **_kwargs):
        return 0.7


def face_keypoints() -> np.ndarray:
    data = np.zeros((1, 17, 3), dtype=np.float64)
    data[0, 0] = (131.0, 130.0, 0.95)  # nose
    data[0, 1] = (100.0, 100.0, 0.95)  # left eye
    data[0, 2] = (163.0, 100.0, 0.95)  # right eye
    data[0, 3] = (60.0, 100.0, 0.90)  # left ear
    data[0, 4] = (203.0, 100.0, 0.90)  # right ear
    return data


def full_body_keypoints() -> np.ndarray:
    data = face_keypoints()
    data[0, 0] = (320.0, 100.0, 0.95)  # nose
    data[0, 1] = (305.0, 95.0, 0.95)  # left eye
    data[0, 2] = (335.0, 95.0, 0.95)  # right eye
    data[0, 3] = (290.0, 100.0, 0.95)  # left ear
    data[0, 4] = (350.0, 100.0, 0.95)  # right ear
    data[0, 5] = (280.0, 160.0, 0.95)  # left shoulder
    data[0, 6] = (360.0, 160.0, 0.95)  # right shoulder
    data[0, 7] = (250.0, 230.0, 0.90)  # left elbow
    data[0, 8] = (390.0, 230.0, 0.90)  # right elbow
    data[0, 9] = (230.0, 300.0, 0.90)  # left wrist
    data[0, 10] = (410.0, 300.0, 0.90)  # right wrist
    data[0, 11] = (295.0, 280.0, 0.95)  # left hip
    data[0, 12] = (345.0, 280.0, 0.95)  # right hip
    data[0, 13] = (290.0, 370.0, 0.90)  # left knee
    data[0, 14] = (350.0, 370.0, 0.90)  # right knee
    data[0, 15] = (285.0, 450.0, 0.90)  # left ankle
    data[0, 16] = (355.0, 450.0, 0.90)  # right ankle
    return data


def detector(
    *,
    depth: bool = False,
    body_parts: bool = False,
    keypoint_data: np.ndarray | None = None,
) -> FaceDistanceDetector:
    settings = Settings(
        _env_file=None,
        depth_enabled=False,
        body_parts_enabled=body_parts,
        camera_focal_length_px=1000.0,
        face_keypoint_confidence=0.5,
        confidence_threshold=0.35,
    )
    instance = FaceDistanceDetector(settings)
    instance._model = FakeModel(keypoint_data if keypoint_data is not None else face_keypoints())
    if depth:
        instance._depth = FakeDepth()
    return instance


def test_face_detector_builds_face_box_and_keypoint_distance() -> None:
    measurements = detector().detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.detection.label == "face"
    assert measurement.detection.priority_level == 1
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


def test_face_detector_applies_diagonal_ray_correction() -> None:
    instance = detector(depth=True)
    instance.set_camera_intrinsics(CameraIntrinsics.from_image(640, 480, 1000.0))

    measurement = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))[0]

    assert measurement.ray_factor is not None
    assert measurement.ray_factor > 1.0
    assert measurement.keypoint_distance_m is not None
    assert measurement.keypoint_distance_m > measurement.keypoint_z_m
    assert measurement.detection.distance_m == pytest.approx(0.72, abs=0.01)


def test_body_parts_are_detected_with_depth_distance() -> None:
    instance = detector(
        depth=True,
        body_parts=True,
        keypoint_data=full_body_keypoints(),
    )

    measurements = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))
    labels = {measurement.detection.label for measurement in measurements}

    assert labels == {
        "face",
        "body",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "left_leg",
        "right_leg",
        "left_foot",
        "right_foot",
    }
    assert all(measurement.detection.priority_level == 1 for measurement in measurements)

    left_hand = next(
        measurement for measurement in measurements if measurement.detection.label == "left_hand"
    )
    assert left_hand.detection.distance_m == pytest.approx(0.7)
    assert left_hand.distance_source == "depth"
    assert left_hand.detection.box.x1 < 230 < left_hand.detection.box.x2
    assert left_hand.detection.box.y1 < 300 < left_hand.detection.box.y2


def test_body_parts_keep_boxes_when_depth_is_disabled() -> None:
    instance = detector(
        depth=False,
        body_parts=True,
        keypoint_data=full_body_keypoints(),
    )

    measurements = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))
    left_arm = next(
        measurement for measurement in measurements if measurement.detection.label == "left_arm"
    )

    assert left_arm.detection.distance_m is None
    assert left_arm.detection.distance_note == "depth_unavailable"


def test_face_only_mode_skips_body_parts() -> None:
    instance = detector(
        depth=True,
        body_parts=False,
        keypoint_data=full_body_keypoints(),
    )

    measurements = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))

    assert [measurement.detection.label for measurement in measurements] == ["face"]


def test_conflicting_depth_is_rejected_and_exposes_source_metadata() -> None:
    instance = detector(depth=True)
    instance._depth.median_depth = lambda *args: 4.0
    measurement = instance.detect_measurements(np.zeros((480, 640, 3), dtype=np.uint8))[0]
    assert measurement.detection.distance_m is None
    assert measurement.detection.distance_note == "depth_keypoint_mismatch"
    assert measurement.detection.distance_source is None
    assert measurement.detection.distance_confidence == 0.0
    assert measurement.distance_source is None
    # Preserve intermediate values for diagnostics/calibration.
    assert measurement.depth_distance_m == 4.0
    assert measurement.keypoint_distance_m == pytest.approx(1.0)


def test_depth_failure_recovers_after_cooldown(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr("neurovector_vision.face_detector.time.monotonic", lambda: now[0])
    instance = detector(depth=True)

    class RecoveringDepth(FakeDepth):
        calls = 0

        def estimate(self, source):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary depth failure")
            return super().estimate(source)

    depth = RecoveringDepth()
    instance._depth = depth
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert instance.detect(image)[0].distance_source == "keypoint"
    assert instance.last_depth_error is not None
    assert instance.detect(image)[0].distance_source == "keypoint"
    assert depth.calls == 1
    assert instance.depth_enabled
    now[0] += instance._settings.depth_retry_interval_s + 1.0
    assert instance.detect(image)[0].distance_source == "depth"
    assert instance.last_depth_error is None
    assert depth.calls == 2


def test_depth_warmup_failure_does_not_permanently_disable_depth() -> None:
    instance = detector(depth=True)

    def fail():
        raise RuntimeError("temporary model load failure")

    instance._depth._load_model = fail
    instance.warmup()
    assert instance.depth_enabled
    assert instance.last_depth_error is not None
    instance._depth_retry_at = 0.0
    assert instance.detect(np.zeros((480, 640, 3), dtype=np.uint8))[0].distance_source == "depth"


def test_body_confidence_uses_weakest_visible_keypoint_and_marks_approximation() -> None:
    data = full_body_keypoints()
    data[0, 9, 2] = 0.55
    instance = detector(depth=True, body_parts=True, keypoint_data=data)
    detections = instance.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    hand = next(item for item in detections if item.label == "left_hand")
    arm = next(item for item in detections if item.label == "left_arm")
    assert hand.confidence == pytest.approx(0.55)
    assert arm.confidence == pytest.approx(0.55)
    assert hand.region_is_approximate
    assert hand.distance_source == "depth"
    assert hand.distance_confidence is None


def test_warmup_runs_depth_even_without_people() -> None:
    instance = detector(depth=True)
    instance._model = FakeModel(np.zeros((0, 17, 3)))
    calls = []
    instance._depth._load_model = lambda: None
    original_estimate = instance._depth.estimate

    def record(image):
        calls.append(image.shape)
        return original_estimate(image)

    instance._depth.estimate = record
    instance.warmup()
    assert calls == [(640, 640, 3)]
    assert instance.last_depth_error is None


def test_face_detector_reuses_provided_depth_without_inference() -> None:
    instance = detector(depth=True)

    def fail(image):
        raise AssertionError("shared depth must not run again")

    instance._depth.estimate = fail
    detections = instance.detect(
        np.zeros((480, 640, 3)), depth_map=np.ones((480, 640)), estimate_depth=False
    )
    assert detections[0].distance_source == "depth"


def test_face_detector_does_not_retry_missing_shared_depth() -> None:
    instance = detector(depth=True)

    def fail(image):
        raise AssertionError("failed shared depth must not be retried in the same frame")

    instance._depth.estimate = fail
    detections = instance.detect(np.zeros((480, 640, 3)), estimate_depth=False)
    assert detections[0].distance_source == "keypoint"


def test_face_detector_uses_shared_runtime_state() -> None:
    from neurovector_vision.depth import DepthRuntime

    settings = Settings(_env_file=None, depth_enabled=False)
    runtime = DepthRuntime(settings)
    instance = FaceDistanceDetector(settings, depth_runtime=runtime)
    runtime.estimator = FakeDepth()
    runtime.last_error = "previous shared failure"
    assert instance._depth is runtime.estimator
    assert instance.depth_enabled
    assert instance.last_depth_error == "previous shared failure"
    instance._depth_retry_at = 12.0
    assert runtime.retry_at == 12.0


def test_face_warmup_can_skip_shared_depth() -> None:
    instance = detector(depth=True)

    def fail(image):
        raise AssertionError("pipeline warms shared depth separately")

    instance._depth.estimate = fail
    instance.warmup(warm_depth=False)
