import pytest

from neurovector_vision.config import Settings
from neurovector_vision.distance import (
    CameraIntrinsics,
    DistanceEstimator,
    FaceDistanceEstimator,
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

    # 높이가 0.5m 이상이므로 너비와 섞지 않고 높이만 사용한다.
    assert estimator.estimate("car", box, image_width=1920) == pytest.approx(1.5)


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


def test_estimator_uses_width_as_approximate_when_primary_axis_is_truncated() -> None:
    estimator = DistanceEstimator(focal_length_px=100.0)
    # 얼굴/상반신이 화면 아래로 잘린 경우: 높이 대신 너비로 근사한다.
    box = BoundingBox(x1=300, y1=100, x2=600, y2=478)

    result = estimator.estimate_detail("person", box, image_width=640, image_height=480)

    assert result.distance_m == pytest.approx(0.17)
    assert result.note == "approximate_bbox_truncated"


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
    assert estimator.estimate("car", box, image_width=640) == pytest.approx(3.0)


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


def test_estimator_prefers_height_without_blending_pose_dependent_width() -> None:
    estimator = DistanceEstimator(focal_length_px=1000.0)
    # 걷는 사람처럼 너비가 부풀어도 높이 1.7m 기준 거리를 유지한다.
    box = BoundingBox(x1=100, y1=100, x2=400, y2=600)

    result = estimator.estimate_detail("person", box, image_width=1280, image_height=720)

    assert result.distance_m == pytest.approx(3.4)
    assert result.note == "approximate_axis_mismatch"


def test_estimator_rejects_head_only_aspect() -> None:
    estimator = DistanceEstimator(focal_length_px=1000.0)
    # 얼굴만 화면 안에 들어온 경우 전체 사람 형상이 아니므로 거리를 버린다.
    box = BoundingBox(x1=100, y1=100, x2=600, y2=250)

    result = estimator.estimate_detail("person", box, image_width=1280, image_height=720)

    assert result.distance_m is None
    assert result.note == "inconsistent_axes"


def test_face_estimator_uses_interpupillary_distance() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0, ipd_m=0.063)
    result = estimator.estimate_eyes((100.0, 100.0, 0.9), (163.0, 100.0, 0.9), 640)

    assert result.distance_m == pytest.approx(1.0)
    assert result.note is None


def test_face_estimator_rejects_missing_eyes() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0)
    result = estimator.estimate_eyes((100.0, 100.0, 0.2), (163.0, 100.0, 0.9), 640)

    assert result.distance_m is None
    assert result.note == "eye_keypoints_missing"


def test_face_estimator_rejects_too_close_eyes() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0, min_eye_distance_px=10.0)
    result = estimator.estimate_eyes((100.0, 100.0, 0.9), (103.0, 100.0, 0.9), 640)

    assert result.distance_m is None
    assert result.note == "eye_keypoints_too_close"


def test_face_estimator_uses_ears_when_eyes_are_missing() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0)
    result = estimator.estimate_ears((50.0, 100.0, 0.9), (200.0, 100.0, 0.9), 640)

    assert result.distance_m == pytest.approx(1.0)
    assert result.note is None


def test_face_estimator_calibrates_from_known_distance() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0)
    scale = estimator.calibrate_from_eyes(0.7, 100.0)

    assert scale == pytest.approx(70.0)
    assert estimator.ipd_scale_px_m(640) == pytest.approx(70.0)
    assert estimator.estimate_eyes((0.0, 0.0, 0.9), (100.0, 0.0, 0.9), 640).distance_m == 0.7


def test_camera_intrinsics_ray_factor_is_one_at_principal_point() -> None:
    intrinsics = CameraIntrinsics.from_image(1280, 720, 1000.0)

    assert intrinsics.ray_factor(640.0, 360.0) == pytest.approx(1.0)
    assert intrinsics.ray_factor(1000.0, 100.0) > 1.0


def test_distance_estimator_applies_diagonal_ray_correction() -> None:
    estimator = DistanceEstimator(focal_length_px=1000.0)
    box = BoundingBox(x1=590, y1=100, x2=890, y2=600)
    image_width, image_height = 1280, 720
    without_correction = estimator.estimate_detail(
        "person",
        box,
        image_width=image_width,
        image_height=image_height,
    )
    estimator.set_camera_intrinsics(CameraIntrinsics.from_image(image_width, image_height, 1000.0))
    with_correction = estimator.estimate_detail(
        "person",
        box,
        image_width=image_width,
        image_height=image_height,
    )

    assert without_correction.distance_m is not None
    assert with_correction.distance_m is not None
    assert with_correction.distance_m > without_correction.distance_m


def test_face_calibration_divides_out_ray_factor() -> None:
    estimator = FaceDistanceEstimator(focal_length_px=1000.0)

    scale = estimator.calibrate_from_eyes(1.0, 63.0, ray_factor=1.1)

    # 보정 시점의 사선 배율을 나눠 두어야 이후 추정값이 직선거리로 유지된다.
    assert scale == pytest.approx(63.0 / 1.1)
