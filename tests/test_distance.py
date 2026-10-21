import pytest

from neurovector_vision.config import Settings
from neurovector_vision.distance import DistanceEstimator, focal_length_px
from neurovector_vision.models import BoundingBox


def test_focal_length_from_horizontal_fov() -> None:
    # 90도 화각이면 초점 거리는 이미지 너비의 절반이다.
    assert focal_length_px(640, 90.0) == pytest.approx(320.0)


def test_focal_length_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        focal_length_px(0, 70.0)
    with pytest.raises(ValueError):
        focal_length_px(640, 180.0)


def test_estimator_uses_class_dimensions() -> None:
    estimator = DistanceEstimator(horizontal_fov_deg=90.0)
    box = BoundingBox(x1=0, y1=0, x2=180, y2=150)

    assert estimator.estimate("car", box, image_width=640) == pytest.approx(3.2)


def test_estimator_uses_focal_length_override() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    box = BoundingBox(x1=0, y1=0, x2=100, y2=100)

    # 너비 1.8m, 높이 1.5m 추정값 중 더 가까운 1.5m
    assert estimator.estimate("car", box, image_width=1920) == pytest.approx(1.5)


def test_estimator_returns_none_for_unknown_class() -> None:
    estimator = DistanceEstimator(horizontal_fov_deg=90.0)
    box = BoundingBox(x1=0, y1=0, x2=50, y2=50)

    assert estimator.estimate("fine-tuned-unknown", box, image_width=640) is None


def test_estimator_is_conservative_when_axes_disagree() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    # 너비 추정 1.8m, 높이 추정 15m -> 더 가까운 값을 사용한다.
    box = BoundingBox(x1=0, y1=0, x2=100, y2=10)

    assert estimator.estimate("car", box, image_width=640) == pytest.approx(1.8)


def test_custom_object_sizes_merge_with_defaults() -> None:
    settings = Settings(
        _env_file=None,
        object_sizes={"my-obstacle": {"width_m": 2.0, "height_m": 1.0}},
    )
    sizes = settings.resolved_object_sizes

    assert sizes["my-obstacle"].height_m == 1.0
    assert "car" in sizes


def test_object_sizes_from_environment_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "NEUROVECTOR_OBJECT_SIZES",
        '{"fine-tuned-obstacle": {"width_m": 2.5, "height_m": 1.2}}',
    )

    settings = Settings(_env_file=None)

    assert settings.resolved_object_sizes["fine-tuned-obstacle"].width_m == 2.5
