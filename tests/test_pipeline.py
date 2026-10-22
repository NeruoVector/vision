import numpy as np
import pytest

from neurovector_vision.camera_calibration import CameraCalibration
from neurovector_vision.config import Settings
from neurovector_vision.pipeline import VisionPipeline


class RecordingDetector:
    def __init__(self):
        self.frames = []
        self.intrinsics = None
        self.focal = 10.0

    @property
    def depth_scale(self):
        return 1.0

    def detect(self, frame, **kwargs):
        self.frames.append(frame)
        return []

    def focal_px(self, width):
        return self.focal

    def set_focal_length_px(self, focal):
        self.focal = focal

    def set_camera_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics


def test_pipeline_shares_scaled_calibration_and_decoded_frame():
    import cv2

    pipeline = VisionPipeline(Settings(_env_file=None))
    pipeline._prepared = True
    pipeline.obstacles = RecordingDetector()
    pipeline.faces = RecordingDetector()
    pipeline._calibration = CameraCalibration(
        camera_matrix=np.array([[100.0, 0.0, 30.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        image_size=(100, 80),
    )
    ok, encoded = cv2.imencode(".png", np.zeros((40, 50, 3), dtype=np.uint8))
    assert ok
    assert pipeline.detect(encoded.tobytes()) == []
    assert pipeline.obstacles.frames[0] is pipeline.faces.frames[0]
    intrinsics = pipeline.obstacles.intrinsics
    assert intrinsics is pipeline.faces.intrinsics
    assert (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) == (50, 100, 15, 10)
    assert pipeline.obstacles.focal == 50


def test_calibrated_vertical_distance_uses_fy():
    from neurovector_vision.distance import CameraIntrinsics, DistanceEstimator
    from neurovector_vision.models import BoundingBox

    estimator = DistanceEstimator(focal_length_px=500)
    estimator.set_camera_intrinsics(CameraIntrinsics(fx=500, fy=1000, cx=100, cy=200))
    estimate = estimator.estimate_detail(
        "person", BoundingBox(x1=75, x2=125, y1=115, y2=285), 400, 500
    )
    assert estimate.distance_m == pytest.approx(10.0)


def test_object_only_pipeline_does_not_create_face_model(tmp_path):
    pipeline = VisionPipeline(
        Settings(
            _env_file=None,
            face_enabled=False,
            depth_enabled=False,
            camera_calibration_path=str(tmp_path / "missing.json"),
        )
    )
    pipeline.obstacles = RecordingDetector()
    assert pipeline.detect(np.zeros((40, 50, 3), dtype=np.uint8)) == []
    assert pipeline.faces is None


def test_malformed_calibration_is_service_unavailable(tmp_path):
    from neurovector_vision.detector import VisionBackendUnavailable

    calibration = tmp_path / "bad.json"
    calibration.write_text('{"camera_matrix": []}')
    pipeline = VisionPipeline(
        Settings(
            _env_file=None,
            face_enabled=False,
            depth_enabled=False,
            camera_calibration_path=str(calibration),
        )
    )
    with pytest.raises(VisionBackendUnavailable, match="보정 파일"):
        pipeline.detect(np.zeros((40, 50, 3), dtype=np.uint8))


def test_depth_distance_for_unknown_class_and_truncated_box():
    from types import SimpleNamespace

    from neurovector_vision.depth import DepthEstimator
    from neurovector_vision.distance import CameraIntrinsics
    from neurovector_vision.models import BoundingBox, Detection
    from neurovector_vision.pipeline import apply_depth_distances

    runtime = SimpleNamespace(estimator=DepthEstimator(Settings(_env_file=None)))
    detection = Detection(
        label="custom_obstacle",
        confidence=0.9,
        box=BoundingBox(x1=0, y1=0, x2=40, y2=40),
        distance_note="unknown_class",
    )
    result = apply_depth_distances(
        [detection],
        np.full((80, 80), 2.0),
        runtime,
        CameraIntrinsics(fx=100, fy=100, cx=20, cy=20),
        1.5,
    )[0]
    assert result.distance_m == 3.0
    assert result.distance_source == "depth"
    assert result.distance_note is None
    assert detection.distance_m is None


def test_shared_depth_runs_once_without_pose_people():
    from types import SimpleNamespace

    from neurovector_vision.depth import DepthEstimator
    from neurovector_vision.models import BoundingBox, Detection

    settings = Settings(_env_file=None)
    pipeline = VisionPipeline(settings)
    pipeline._prepared = True
    pipeline.obstacles = RecordingDetector()
    pipeline.obstacles.detect = lambda frame: [
        Detection(
            label="car",
            confidence=0.9,
            box=BoundingBox(x1=5, y1=5, x2=35, y2=35),
        )
    ]
    calls = []

    def estimate(frame):
        calls.append(frame)
        return np.full(frame.shape[:2], 2.0)

    class Faces(RecordingDetector):
        def detect(self, frame, **kwargs):
            assert kwargs["estimate_depth"] is False
            assert kwargs["depth_map"].shape == frame.shape[:2]
            assert frame is calls[0]
            return []

    pipeline.faces = Faces()
    pipeline.depth = SimpleNamespace(estimator=DepthEstimator(settings), estimate=estimate)
    detections = pipeline.detect(np.zeros((40, 40, 3), dtype=np.uint8))
    assert len(calls) == 1
    assert detections[0].distance_source == "depth"


def test_depth_is_independent_of_face_detection(tmp_path, monkeypatch):
    from neurovector_vision.depth import DepthRuntime
    from neurovector_vision.models import BoundingBox, Detection

    calls = []

    def estimate(self, frame):
        calls.append(frame.shape)
        return np.full(frame.shape[:2], 2.0)

    monkeypatch.setattr(DepthRuntime, "estimate", estimate)
    pipeline = VisionPipeline(
        Settings(
            _env_file=None,
            face_enabled=False,
            camera_calibration_path=str(tmp_path / "missing.json"),
        )
    )
    pipeline.obstacles = RecordingDetector()
    pipeline.obstacles.detect = lambda frame: [
        Detection(
            label="bottle",
            confidence=0.9,
            box=BoundingBox(x1=5, y1=5, x2=35, y2=35),
        )
    ]
    result = pipeline.detect(np.zeros((40, 40, 3), dtype=np.uint8))
    assert pipeline.faces is None
    assert calls == [(40, 40, 3)]
    assert result[0].distance_source == "depth"


def test_invalid_depth_keeps_bbox_fallback_and_radial_correction():
    from types import SimpleNamespace

    from neurovector_vision.depth import DepthEstimator
    from neurovector_vision.distance import CameraIntrinsics
    from neurovector_vision.models import BoundingBox, Detection
    from neurovector_vision.pipeline import apply_depth_distances

    runtime = SimpleNamespace(estimator=DepthEstimator(Settings(_env_file=None)))
    intrinsics = CameraIntrinsics(fx=100, fy=100, cx=0, cy=20)
    detection = Detection(
        label="car",
        confidence=0.9,
        box=BoundingBox(x1=10, y1=10, x2=30, y2=30),
        distance_m=5.0,
        distance_source="bbox",
    )
    result = apply_depth_distances([detection], None, runtime, intrinsics, 1.0)[0]
    assert result.distance_m == 5
    assert result.distance_source == "bbox"
    assert result.distance_note == "depth_unavailable_bbox_fallback"
    result = apply_depth_distances(
        [detection],
        np.full((40, 40), 2.0),
        runtime,
        intrinsics,
        1.0,
    )[0]
    assert result.distance_m == round(2 * np.sqrt(1.04), 2)
