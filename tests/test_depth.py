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
