"""Shared YOLO26 object, pose and depth inference for one corrected frame.

A caller must serialize access to each instance. Monocular model distances remain
estimates and should be calibrated against measured distances for the camera.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Settings
from .detector import ObstacleDetector, VisionBackendUnavailable, _decode_image_bytes
from .distance import CameraIntrinsics
from .models import Detection

if TYPE_CHECKING:
    import numpy as np

    from .depth import DepthRuntime


def apply_depth_distances(
    detections: list[Detection],
    depth_map: np.ndarray | None,
    runtime: DepthRuntime | None,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
) -> list[Detection]:
    """Sample object surface Z, apply shared scale and convert to radial distance.

    Object class sizes are only a fallback. A depth map can provide distances for
    classes without known physical dimensions and objects truncated by the frame.
    """
    if runtime is None or runtime.estimator is None:
        return detections
    output = []
    for detection in detections:
        raw_depth = (
            runtime.estimator.median_depth(depth_map, detection.box)
            if depth_map is not None
            else None
        )
        if raw_depth is not None:
            box = detection.box
            ray = intrinsics.ray_factor((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
            output.append(
                detection.model_copy(
                    update={
                        "distance_m": round(raw_depth * depth_scale * ray, 2),
                        "distance_source": "depth",
                        "distance_confidence": None,
                        "distance_note": None,
                    }
                )
            )
        else:
            note = "depth_unavailable" if depth_map is None else "depth_invalid_region"
            if detection.distance_m is not None:
                note += "_bbox_fallback"
            output.append(
                detection.model_copy(
                    update={
                        "distance_note": detection.distance_note or note,
                    }
                )
            )
    return output


class VisionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.obstacles = ObstacleDetector(settings)
        self.faces = None
        self.depth = None
        self._calibration = None
        self._prepared = False

    def _prepare(self) -> None:
        if self._prepared:
            return
        from .camera_calibration import CameraCalibration
        from .depth import DepthRuntime

        try:
            self._calibration = CameraCalibration.load(self.settings.camera_calibration_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VisionBackendUnavailable("카메라 보정 파일을 읽을 수 없습니다.") from exc
        self.depth = DepthRuntime(self.settings)
        if self.settings.face_enabled:
            from .face_detector import FaceDistanceDetector

            self.faces = FaceDistanceDetector(self.settings, depth_runtime=self.depth)
        self._prepared = True

    def warmup(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise VisionBackendUnavailable("numpy is required for vision inference") from exc
        self._prepare()
        if self.faces is not None:
            self.faces.warmup(warm_depth=False)
        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    def detect(self, image) -> list[Detection]:
        frame = _decode_image_bytes(image) if isinstance(image, bytes) else image
        self._prepare()
        height, width = frame.shape[:2]
        calibration = self._calibration
        if calibration is not None:
            calibration = calibration.scaled_for(width, height)
            frame = calibration.undistort(frame)
            matrix = calibration.camera_matrix
            focal = self.settings.camera_focal_length_px
            intrinsics = CameraIntrinsics(
                fx=focal if focal is not None else float(matrix[0, 0]),
                fy=focal if focal is not None else float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
            )
            self.obstacles.set_focal_length_px(intrinsics.fx)
        else:
            intrinsics = CameraIntrinsics.from_image(width, height, self.obstacles.focal_px(width))
        self.obstacles.set_camera_intrinsics(intrinsics)
        depth_map = self.depth.estimate(frame) if self.depth is not None else None
        depth_scale = (
            self.faces.depth_scale if self.faces is not None else self.settings.depth_scale
        )
        detections = apply_depth_distances(
            self.obstacles.detect(frame),
            depth_map,
            self.depth,
            intrinsics,
            depth_scale,
        )
        if self.faces is not None:
            self.faces.set_focal_length_px(intrinsics.fx)
            self.faces.set_camera_intrinsics(intrinsics)
            detections.extend(self.faces.detect(frame, depth_map=depth_map, estimate_depth=False))
        return detections
