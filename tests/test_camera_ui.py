import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from neurovector_vision.face_detector import FaceMeasurement  # noqa: E402
from neurovector_vision.models import BoundingBox, Detection  # noqa: E402
from neurovector_vision.tracking import TemporalTracker  # noqa: E402


def test_camera_infers_directly_from_frame_and_retains_raw_calibration_measurements() -> None:
    path = Path(__file__).parents[1] / "tools" / "camera_ui.py"
    spec = importlib.util.spec_from_file_location("camera_ui_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        pipeline = module.CameraPipeline.__new__(module.CameraPipeline)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        measurement = FaceMeasurement(
            detection=Detection(
                label="face",
                confidence=0.9,
                box=BoundingBox(x1=1, y1=1, x2=10, y2=10),
                distance_m=2.0,
            )
        )
        detector = Mock()
        detector.detect_measurements.return_value = [measurement]
        detector.focal_px.return_value = 10.0
        detector.ipd_scale_px_m.return_value = 1.0
        detector.depth_scale = 1.0
        detector.depth_enabled = False
        detector.last_depth_error = None
        detector.last_depth_ms = None
        pipeline._detector = detector
        pipeline._depth_runtime = Mock()
        pipeline._depth_runtime.estimate.return_value = None
        pipeline._obstacles = None
        pipeline._config = SimpleNamespace(mirror=False, focal_length_px=None, fov=70.0)
        pipeline._calibration = None
        pipeline._tracker = TemporalTracker()
        pipeline._inference_lock = threading.Lock()
        pipeline._calibration_epoch = 0
        pipeline._tracked_epoch = 0
        pipeline._tracking_frame_shape = None
        pipeline._condition = threading.Condition()
        pipeline._status = {"camera": {}, "frames": 0}
        pipeline._placeholder = lambda _message: b"placeholder"
        pipeline._frame_times = []
        pipeline._running = True
        capture = Mock()
        capture.get.return_value = 16
        capture.isOpened.return_value = True

        def read():
            pipeline._running = False
            return True, frame

        capture.read.side_effect = read
        pipeline._open_camera = lambda: capture
        pipeline._run()

        assert detector.detect_measurements.call_args.args[0] is frame
        assert pipeline._status["ok"] is True
        assert pipeline._status["detections"][0]["track_id"] == 1
        assert pipeline._last_measurements[0] is measurement
        assert measurement.detection.track_id is None
        capture.release.assert_called_once()
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def camera_module():
    path = Path(__file__).parents[1] / "tools" / "camera_ui.py"
    spec = importlib.util.spec_from_file_location("camera_ui_epoch_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_calibration_queues_reset_and_resolution_changes_expire_tracks(camera_module) -> None:
    pipeline = camera_module.CameraPipeline.__new__(camera_module.CameraPipeline)
    pipeline._tracker = TemporalTracker()
    pipeline._inference_lock = threading.Lock()
    pipeline._condition = threading.Condition()
    pipeline._calibration_epoch = 0
    pipeline._tracked_epoch = 0
    pipeline._tracking_frame_shape = None
    pipeline._status = {"camera": {"width": 16}}
    measurement = FaceMeasurement(
        detection=Detection(
            label="face",
            confidence=0.9,
            box=BoundingBox(x1=1, y1=1, x2=10, y2=10),
            distance_m=1.0,
        ),
        eye_distance_px=5.0,
    )
    pipeline._last_measurements = [measurement]
    pipeline._detector = Mock()
    pipeline._depth_runtime = Mock()
    pipeline._depth_runtime.estimate.return_value = None
    pipeline._obstacles = None
    pipeline._detector.detect_measurements.return_value = [measurement]
    pipeline._detector.calibrate.return_value = {"ok": True}
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    _, first = pipeline._infer_frame(frame)
    assert pipeline.calibrate(5.0) == {"ok": True}
    assert pipeline._tracked_epoch == 0
    assert pipeline._calibration_epoch == 1
    measurement.detection.distance_m = 5.0
    _, second = pipeline._infer_frame(frame)
    assert second[0].distance_m == 5.0
    assert second[0].track_id != first[0].track_id
    assert pipeline._tracked_epoch == 1
    _, third = pipeline._infer_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    assert third[0].track_id != second[0].track_id


@pytest.mark.parametrize("arguments", [[], ["--no-depth"]])
def test_camera_cli_honors_depth_disabled_in_environment(camera_module, monkeypatch, arguments):
    monkeypatch.setenv("NEUROVECTOR_DEPTH_ENABLED", "false")
    monkeypatch.setattr(sys, "argv", ["camera_ui.py", *arguments])
    assert camera_module.parse_args().depth_enabled is False


def test_camera_shares_one_depth_map_for_objects_and_pose(camera_module, monkeypatch):
    pipeline = camera_module.CameraPipeline.__new__(camera_module.CameraPipeline)
    pipeline._tracker = None
    pipeline._inference_lock = threading.Lock()
    pipeline._calibration_epoch = 0
    pipeline._tracked_epoch = 0
    pipeline._tracking_frame_shape = None
    pipeline._camera_intrinsics = object()
    pipeline._depth_runtime = Mock()
    depth_map = np.ones((16, 16), dtype=np.float32)
    pipeline._depth_runtime.estimate.return_value = depth_map
    pipeline._detector = Mock()
    pipeline._detector.depth_scale = 1.7
    pipeline._detector.detect_measurements.return_value = []
    pipeline._obstacles = Mock()
    object_detection = Detection(
        label="chair",
        confidence=0.9,
        box=BoundingBox(x1=1, y1=1, x2=10, y2=10),
        distance_m=1.7,
        distance_source="depth",
    )
    pipeline._obstacles.detect.return_value = [object_detection]
    apply_depth = Mock(return_value=[object_detection])
    monkeypatch.setattr(camera_module, "apply_depth_distances", apply_depth)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    measurements, detections = pipeline._infer_frame(frame)
    pipeline._depth_runtime.estimate.assert_called_once_with(frame)
    pipeline._detector.detect_measurements.assert_called_once_with(
        frame, depth_map=depth_map, estimate_depth=False
    )
    pipeline._obstacles.detect.assert_called_once_with(frame)
    apply_depth.assert_called_once_with(
        [object_detection],
        depth_map,
        pipeline._depth_runtime,
        pipeline._camera_intrinsics,
        1.7,
    )
    assert detections == [object_detection]
    assert measurements[0].distance_source == "depth"


def test_camera_cli_selects_general_object_model_and_face_only_disables_objects(
    camera_module, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["camera_ui.py", "--object-model", "yolo26s.pt"])
    config = camera_module.parse_args()
    assert config.object_model_path == "yolo26s.pt"
    assert config.objects_enabled is True
    monkeypatch.setattr(sys, "argv", ["camera_ui.py", "--face-only"])
    config = camera_module.parse_args()
    assert config.objects_enabled is False
    assert config.body_parts_enabled is False


def test_camera_mirror_uses_pixel_index_for_calibration_principal_point(camera_module):
    pipeline = camera_module.CameraPipeline.__new__(camera_module.CameraPipeline)
    pipeline._config = SimpleNamespace(mirror=True)
    pipeline._calibration = SimpleNamespace(
        camera_matrix=np.array([[100.0, 0.0, 3.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    )
    intrinsics = pipeline._make_intrinsics(16, 16, None)
    assert intrinsics.cx == 12.0


def test_camera_calibrated_intrinsics_honor_manual_focal(camera_module):
    pipeline = camera_module.CameraPipeline.__new__(camera_module.CameraPipeline)
    pipeline._config = SimpleNamespace(mirror=False)
    pipeline._calibration = SimpleNamespace(
        camera_matrix=np.array([[100.0, 0.0, 3.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]])
    )
    intrinsics = pipeline._make_intrinsics(16, 16, 200.0)
    assert (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy) == (200, 200, 3, 4)
