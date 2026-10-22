import numpy as np
import pytest

from neurovector_vision.config import Settings
from neurovector_vision.depth import DepthEstimator
from neurovector_vision.models import BoundingBox


def test_median_depth_uses_central_box_region() -> None:
    estimator = DepthEstimator(Settings(_env_file=None, depth_enabled=False))
    depth = np.ones((100, 100), dtype=np.float32) * 10.0
    depth[40:60, 40:60] = 0.8

    value = estimator.median_depth(
        depth,
        BoundingBox(x1=10, y1=10, x2=90, y2=90),
        inset_ratio=0.4,
    )

    assert value == pytest.approx(0.8)


def test_median_depth_returns_none_for_invalid_values() -> None:
    estimator = DepthEstimator(Settings(_env_file=None, depth_enabled=False))
    depth = np.full((20, 20), np.nan, dtype=np.float32)

    assert estimator.median_depth(depth, BoundingBox(x1=0, y1=0, x2=20, y2=20)) is None


def test_median_depth_around_keypoints_uses_body_surface() -> None:
    estimator = DepthEstimator(Settings(_env_file=None, depth_enabled=False))
    depth = np.ones((100, 100), dtype=np.float32) * 5.0
    depth[40:50, 20:30] = 0.8
    depth[20:30, 70:80] = 1.2

    value = estimator.median_depth_around_points(
        depth,
        [(25.0, 45.0), (75.0, 25.0)],
        radius_px=3,
    )

    assert value == pytest.approx(1.0)


def test_median_depth_around_keypoints_returns_none_without_valid_pixels() -> None:
    estimator = DepthEstimator(Settings(_env_file=None, depth_enabled=False))
    depth = np.full((40, 40), np.nan, dtype=np.float32)

    assert estimator.median_depth_around_points(depth, [(20.0, 20.0)]) is None


def test_cuda_failure_uses_cpu_until_cooldown_then_recovers(monkeypatch) -> None:
    from types import SimpleNamespace

    now = [10.0]
    monkeypatch.setattr("neurovector_vision.depth.time.monotonic", lambda: now[0])
    estimator = DepthEstimator(Settings(_env_file=None, device="0"))

    class Model:
        def __init__(self):
            self.devices = []
            self.moves = []

        def to(self, device):
            self.moves.append(device)

        def predict(self, *, device, **kwargs):
            self.devices.append(device)
            if len(self.devices) == 1:
                raise RuntimeError("CUDA out of memory")
            return [SimpleNamespace(depth=SimpleNamespace(data=np.ones((20, 20))))]

    model = Model()
    estimator._model = model
    estimator.estimate(np.zeros((20, 20, 3)))
    estimator.estimate(np.zeros((20, 20, 3)))
    now[0] += estimator._settings.depth_gpu_retry_interval_s + 1.0
    estimator.estimate(np.zeros((20, 20, 3)))
    assert model.devices == ["0", "cpu", "cpu", "0"]
    assert model.moves == ["cpu", "cuda:0"]
    assert estimator.last_device == "0"


@pytest.mark.parametrize("device,message", [("cpu", "CUDA error"), ("0", "invalid tensor shape")])
def test_non_cuda_runtime_failure_is_not_retried(device, message) -> None:
    estimator = DepthEstimator(Settings(_env_file=None, device=device))

    class Model:
        calls = 0

        def predict(self, **kwargs):
            self.calls += 1
            raise RuntimeError(message)

    estimator._model = Model()
    with pytest.raises(RuntimeError, match=message):
        estimator.estimate(np.zeros((20, 20, 3)))
    assert estimator._model.calls == 1


@pytest.mark.parametrize(
    "coordinates",
    [(-100, 5, -20, 20), (25, 5, 40, 20), (5, -100, 20, -20), (5, 25, 20, 40)],
)
def test_depth_sampling_rejects_boxes_outside_image(coordinates) -> None:
    estimator = DepthEstimator(Settings(_env_file=None))
    box = BoundingBox(x1=coordinates[0], y1=coordinates[1], x2=coordinates[2], y2=coordinates[3])
    assert estimator.median_depth(np.ones((20, 20)), box) is None


def test_depth_sampling_rejects_points_outside_image() -> None:
    estimator = DepthEstimator(Settings(_env_file=None))
    assert (
        estimator.median_depth_around_points(
            np.ones((20, 20)), [(-10, 10), (10, -10), (25, 10), (10, 25), (float("nan"), 5)]
        )
        is None
    )


def test_depth_map_must_match_input_coordinates() -> None:
    from types import SimpleNamespace

    from neurovector_vision.depth import DepthModelUnavailable

    estimator = DepthEstimator(Settings(_env_file=None, device="cpu"))
    estimator._model = SimpleNamespace(
        predict=lambda **kwargs: [SimpleNamespace(depth=SimpleNamespace(data=np.ones((10, 10))))]
    )
    with pytest.raises(DepthModelUnavailable, match="크기"):
        estimator.estimate(np.zeros((20, 20, 3)))


def test_shared_depth_runtime_failure_cooldown_and_recovery(monkeypatch) -> None:
    from neurovector_vision.depth import DepthRuntime

    now = [10.0]
    monkeypatch.setattr("neurovector_vision.depth.time.monotonic", lambda: now[0])
    runtime = DepthRuntime(Settings(_env_file=None, depth_retry_interval_s=5.0))

    class Estimator:
        last_inference_ms = 1.0
        calls = 0

        def estimate(self, image):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary depth failure")
            return np.ones((20, 20))

    runtime.estimator = Estimator()
    frame = np.zeros((20, 20, 3))
    assert runtime.estimate(frame) is None
    assert "temporary depth failure" in runtime.last_error
    assert runtime.estimate(frame) is None
    assert runtime.estimator.calls == 1
    now[0] += 6
    assert runtime.estimate(frame).shape == (20, 20)
    assert runtime.last_error is None
    assert runtime.last_inference_ms == 1.0
