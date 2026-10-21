import pytest

from neurovector_vision.config import Settings
from neurovector_vision.distance import (
    DistanceEstimator,
    focal_length_from_reference,
    focal_length_px,
    horizontal_fov_deg,
)
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

    # 너비 1.8m, 높이 1.5m 추정값이 모두 유효하므로 면적 기반 기하평균을 쓴다.
    assert estimator.estimate("car", box, image_width=1920) == pytest.approx(1.64)


def test_estimator_returns_none_for_unknown_class() -> None:
    estimator = DistanceEstimator(horizontal_fov_deg=90.0)
    box = BoundingBox(x1=0, y1=0, x2=50, y2=50)

    assert estimator.estimate("fine-tuned-unknown", box, image_width=640) is None


def test_estimator_rejects_when_axes_disagree() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    # 너비 추정 1.8m, 높이 추정 15m -> 형상이 크게 어긋나므로 거리를 버린다.
    box = BoundingBox(x1=100, y1=100, x2=200, y2=110)

    result = estimator.estimate_detail("car", box, image_width=640, image_height=480)

    assert result.distance_m is None
    assert result.note == "inconsistent_axes"


def test_estimator_rejects_bbox_truncated_on_both_axes() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    box = BoundingBox(x1=0, y1=0, x2=100, y2=100)

    result = estimator.estimate_detail("person", box, image_width=640, image_height=480)

    assert result.distance_m is None
    assert result.note == "bbox_truncated"


def test_estimator_uses_untruncated_axis() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    # 왼쪽만 잘렸고 높이는 온전한 자동차 -> 높이로만 추정한다.
    box = BoundingBox(x1=0, y1=100, x2=100, y2=250)

    assert estimator.estimate("car", box, image_width=640, image_height=480) == pytest.approx(1.0)


def test_horizontal_fov_is_inverse_of_focal_length() -> None:
    focal = focal_length_px(640, 90.0)

    assert horizontal_fov_deg(640, focal) == pytest.approx(90.0)


def test_focal_length_from_known_reference() -> None:
    # 2m 거리의 1.7m 사람이 200px 높이로 보이면 초점 거리는 235.3px.
    assert focal_length_from_reference(2.0, 200.0, 1.7) == pytest.approx(235.294, rel=1e-3)


def test_estimator_can_apply_runtime_focal_length() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    box = BoundingBox(x1=100, y1=100, x2=200, y2=200)

    estimator.set_focal_length_px(200.0)

    assert estimator.focal_length_override == 200.0
    assert estimator.estimate("car", box, image_width=640) == pytest.approx(3.29)


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
