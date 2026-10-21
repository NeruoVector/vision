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

    def scaled_for(self, image_width: int, image_height: int) -> CameraCalibration:
        """다른 해상도로 리사이즈된 프레임에 맞게 내부 파라미터를 스케일한다.

        정규화 좌표 기준의 왜곡 계수는 유지하고, 픽셀 단위인 fx/fy/cx/cy 만
        해상도 비율에 맞춰 바꾼다.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_size is None:
            return self
        old_width, old_height = self.image_size
        if old_width <= 0 or old_height <= 0:
            return self
        if (old_width, old_height) == (image_width, image_height):
            return self
        scale_x = image_width / old_width
        scale_y = image_height / old_height
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= scale_x
        matrix[0, 2] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[1, 2] *= scale_y
        return CameraCalibration(
            camera_matrix=matrix,
            distortion_coefficients=self.distortion_coefficients.copy(),
            image_size=(image_width, image_height),
            reprojection_error=self.reprojection_error,
        )

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
