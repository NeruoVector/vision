"""체커보드 카메라 캘리브레이션 로드/적용.

``tools/calibrate_camera.py`` 로 만든 JSON을 읽어 렌즈 왜곡을 보정하고,
거리 추정에 실제 초점 거리(``fx``)를 제공한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int] | None = None
    reprojection_error: float | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        if distortion.size < 4:
            raise ValueError("distortion_coefficients must have at least 4 values")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)

    @property
    def focal_px(self) -> float:
        return float(self.camera_matrix[0, 0])

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """렌즈 왜곡을 제거한다. 왜곡 계수가 0이면 원본을 그대로 반환한다."""
        if np.allclose(self.distortion_coefficients, 0.0):
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.distortion_coefficients)

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "image_size": list(self.image_size) if self.image_size else None,
            "reprojection_error": self.reprojection_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraCalibration:
        image_size = data.get("image_size")
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(data["distortion_coefficients"], dtype=np.float64),
            image_size=tuple(int(v) for v in image_size) if image_size else None,
            reprojection_error=(
                float(data["reprojection_error"])
                if data.get("reprojection_error") is not None
                else None
            ),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibration | None:
        file = Path(path)
        if not file.exists():
            return None
        return cls.from_dict(json.loads(file.read_text(encoding="utf-8")))
