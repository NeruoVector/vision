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
