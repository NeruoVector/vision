import numpy as np
import pytest

from neurovector_vision.camera_calibration import CameraCalibration


def test_camera_calibration_round_trip(tmp_path) -> None:
    matrix = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.1, -0.05, 0.0, 0.0, 0.0])
    calibration = CameraCalibration(
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=(1280, 720),
        reprojection_error=0.42,
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)

    loaded = CameraCalibration.load(path)

    assert loaded is not None
    assert loaded.focal_px == pytest.approx(900.0)
    assert loaded.image_size == (1280, 720)
    assert loaded.reprojection_error == pytest.approx(0.42)
    assert np.allclose(loaded.camera_matrix, matrix)


def test_camera_calibration_missing_file_returns_none(tmp_path) -> None:
    assert CameraCalibration.load(tmp_path / "missing.json") is None


def test_camera_calibration_undistorts_frame_shape() -> None:
    matrix = np.array([[500.0, 0.0, 4.0], [0.0, 500.0, 4.0], [0.0, 0.0, 1.0]])
    calibration = CameraCalibration(
        camera_matrix=matrix,
        distortion_coefficients=np.zeros(5),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert calibration.undistort(frame).shape == frame.shape
