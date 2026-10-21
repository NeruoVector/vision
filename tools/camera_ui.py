"""로컬 카메라 + YOLO26 얼굴/신체 부위 감지 실시간 테스트 UI.

브라우저에서 ``http://localhost:8090`` 을 열면 카메라 영상과 얼굴, 몸통,
팔, 손, 다리, 발 박스와 부위별 거리, 추론 시간/FPS 를 볼 수 있다.
운영 API 가 아니라 로컬 카메라/모델/거리 보정을 빠르게 확인하는 개발 도구다.

실행:
    uv run --extra vision python tools/camera_ui.py
    uv run --extra vision python tools/camera_ui.py --camera 0 --port 8090
    uv run --extra vision python tools/camera_ui.py --face-only
"""

from __future__ import annotations

import argparse
import copy
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from neurovector_vision.camera_calibration import CameraCalibration
from neurovector_vision.config import Settings
from neurovector_vision.distance import (
    CameraIntrinsics,
    focal_length_px,
    horizontal_fov_deg,
)
from neurovector_vision.face_detector import FaceDistanceDetector
from neurovector_vision.priority import priority_name

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neurovector_vision.face_detector import FaceMeasurement
    from neurovector_vision.models import Detection

logger = logging.getLogger(__name__)
HTML = Path(__file__).with_name("camera_ui.html").read_text(encoding="utf-8")


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    face_model_path: str = "yolo26n-pose.pt"
    depth_model_path: str = "yolo26n-depth.pt"
    width: int = 1280
    height: int = 720
    port: int = 8090
    host: str = "127.0.0.1"
    fov: float = 70.0
    focal_length_px: float | None = None
    face_ipd_scale_px_m: float | None = None
    calibration_path: str = "camera_calibration.json"
    depth_enabled: bool = True
    body_parts_enabled: bool = True
    device: str = "auto"
    mirror: bool = True


class CalibrationRequest(BaseModel):
    actual_distance_m: float = Field(gt=0.0)
    # 이전 UI/클라이언트와의 호환을 위해 받지만 얼굴 보정에는 사용하지 않는다.
    label: str | None = None
    dimension: str = "height"


class CameraPipeline:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._calibration = CameraCalibration.load(config.calibration_path)
        focal_override = config.focal_length_px
        if focal_override is None and self._calibration is not None:
            focal_override = self._calibration.focal_px
        self._settings = Settings(
            face_model_path=config.face_model_path,
            depth_model_path=config.depth_model_path,
            depth_enabled=config.depth_enabled,
            body_parts_enabled=config.body_parts_enabled,
            device=config.device,
            camera_horizontal_fov_deg=config.fov,
            camera_focal_length_px=focal_override,
            face_ipd_scale_px_m=config.face_ipd_scale_px_m,
        )
        self._detector = FaceDistanceDetector(self._settings)
        self._camera_intrinsics = self._make_intrinsics(
            config.width,
            config.height,
            focal_override,
        )
        self._detector.set_camera_intrinsics(self._camera_intrinsics)
        self._condition = threading.Condition()
        self._latest_jpeg = self._placeholder("Preparing YOLO26 and camera...")
        self._snapshot_jpeg = self._latest_jpeg
        self._last_measurements: list[FaceMeasurement] = []
        self._last_detections: list[Detection] = []
        self._status: dict = {
            "ok": False,
            "error": None,
            "frames": 0,
            "fps": 0.0,
            "inference_ms": None,
            "depth_ms": None,
            "detections": [],
            "nearest_distance_m": None,
            "focal_px": None,
            "fov_deg": None,
            "face_scale_px_m": None,
            "depth_scale": self._detector.depth_scale,
            "depth_enabled": self._detector.depth_enabled,
            "body_parts_enabled": self._detector.body_parts_enabled,
            "depth_error": None,
            "model": Path(config.face_model_path).name,
            "device": self._settings.resolved_device,
            "calibration": {
                "loaded": self._calibration is not None,
                "path": config.calibration_path,
                "reprojection_error": (
                    self._calibration.reprojection_error if self._calibration is not None else None
                ),
            },
            "camera": {"ok": False, "width": 0, "height": 0, "error": None},
            "updated_at": None,
        }
        self._frame_times: list[float] = []
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="camera-pipeline",
            daemon=True,
        )
        self._thread.start()

    def _placeholder(self, message: str) -> bytes:
        image = np.zeros((self._config.height, self._config.width, 3), dtype=np.uint8)
        image[:] = (12, 18, 28)
        width = self._config.width
        height = self._config.height
        cv2.putText(
            image,
            "NeuroVector Vision",
            (width // 2 - 230, height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (220, 235, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            message,
            (width // 2 - 230, height // 2 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (120, 200, 220),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return encoded.tobytes() if ok else b""

    def _open_camera(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._config.camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self._config.camera_index)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _make_intrinsics(
        self,
        image_width: int,
        image_height: int,
        focal_override: float | None,
    ) -> CameraIntrinsics | None:
        if image_width <= 0 or image_height <= 0:
            return None
        if self._calibration is not None:
            matrix = self._calibration.camera_matrix
            cx = float(matrix[0, 2])
            if self._config.mirror:
                # undistort 후 좌우 반전하므로 주점도 반전 좌표계로 옮긴다.
                cx = image_width - cx
            return CameraIntrinsics(
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=cx,
                cy=float(matrix[1, 2]),
            )
        focal = focal_override
        if focal is None:
            focal = focal_length_px(image_width, self._config.fov)
        try:
            return CameraIntrinsics.from_image(image_width, image_height, focal)
        except ValueError:
            return None

    def _intrinsics(self, image_width: int) -> tuple[float | None, float | None]:
        if image_width <= 0:
            return None, None
        try:
            focal = self._detector.focal_px(image_width)
            return round(focal, 2), round(horizontal_fov_deg(image_width, focal), 2)
        except ValueError:
            return None, None

    def _fail(self, message: str) -> None:
        with self._condition:
            self._status["ok"] = False
            self._status["error"] = message
            self._status["camera"]["ok"] = False
            self._latest_jpeg = self._placeholder("Camera pipeline stopped.")
            self._condition.notify_all()

    def _run(self) -> None:
        capture = self._open_camera()
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._calibration is not None and actual_width > 0 and actual_height > 0:
            self._calibration = self._calibration.scaled_for(actual_width, actual_height)
            if self._config.focal_length_px is None:
                self._detector.set_focal_length_px(self._calibration.focal_px)
        self._camera_intrinsics = self._make_intrinsics(
            actual_width,
            actual_height,
            self._config.focal_length_px,
        )
        self._detector.set_camera_intrinsics(self._camera_intrinsics)
        focal_px, fov_deg = self._intrinsics(actual_width)
        with self._condition:
            self._status["camera"] = {
                "ok": capture.isOpened(),
                "width": actual_width,
                "height": actual_height,
                "error": None if capture.isOpened() else "camera open failed",
            }
            self._status["focal_px"] = focal_px
            self._status["fov_deg"] = fov_deg
            self._status["face_scale_px_m"] = (
                round(self._detector.ipd_scale_px_m(actual_width), 2) if actual_width > 0 else None
            )
            self._condition.notify_all()

        if not capture.isOpened():
            self._fail("카메라를 열 수 없습니다. --camera 인덱스와 권한을 확인하세요.")
            return

        try:
            # 첫 요청에서 모델 로딩 지연이 발생하지 않도록 미리 준비한다.
            self._detector.warmup()
            with self._condition:
                self._latest_jpeg = self._placeholder("Waiting for the first frame...")
                self._status["depth_error"] = self._detector.last_depth_error
                self._status["depth_enabled"] = self._detector.depth_enabled
                self._condition.notify_all()

            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                started = time.perf_counter()
                if self._calibration is not None:
                    frame = self._calibration.undistort(frame)
                if self._config.mirror:
                    frame = cv2.flip(frame, 1)
                ok, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    continue
                measurements = self._detector.detect_measurements(raw.tobytes())
                detections = [measurement.detection for measurement in measurements]
                inference_ms = (time.perf_counter() - started) * 1000.0

                annotated = self._draw(frame.copy(), detections)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 84],
                )
                if not ok:
                    continue

                now = time.monotonic()
                self._frame_times.append(now)
                self._frame_times = self._frame_times[-30:]
                fps = 0.0
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    if span > 0:
                        fps = (len(self._frame_times) - 1) / span

                nearest = min(
                    (d.distance_m for d in detections if d.distance_m is not None),
                    default=None,
                )
                payload = [
                    {
                        "label": measurement.detection.label,
                        "confidence": round(measurement.detection.confidence, 4),
                        "distance_m": measurement.detection.distance_m,
                        "distance_note": measurement.detection.distance_note,
                        "priority_level": measurement.detection.priority_level,
                        "priority_name": priority_name(measurement.detection.priority_level),
                        "box": measurement.detection.box.model_dump(),
                        "keypoint_distance_m": measurement.keypoint_distance_m,
                        "keypoint_z_m": measurement.keypoint_z_m,
                        "raw_depth_m": measurement.raw_depth_m,
                        "depth_distance_m": measurement.depth_distance_m,
                        "distance_source": measurement.distance_source,
                        "ray_factor": (
                            round(measurement.ray_factor, 4)
                            if measurement.ray_factor is not None
                            else None
                        ),
                        "eye_distance_px": (
                            round(measurement.eye_distance_px, 1)
                            if measurement.eye_distance_px is not None
                            else None
                        ),
                    }
                    for measurement in measurements
                ]
                focal_px, fov_deg = self._intrinsics(actual_width)
                face_scale = (
                    round(self._detector.ipd_scale_px_m(actual_width), 2)
                    if actual_width > 0
                    else None
                )

                with self._condition:
                    self._latest_jpeg = encoded.tobytes()
                    self._snapshot_jpeg = self._latest_jpeg
                    self._last_measurements = measurements
                    self._last_detections = detections
                    self._status.update(
                        {
                            "ok": True,
                            "error": None,
                            "frames": self._status["frames"] + 1,
                            "fps": round(fps, 2),
                            "inference_ms": round(inference_ms, 1),
                            "depth_ms": (
                                round(self._detector.last_depth_ms, 1)
                                if self._detector.last_depth_ms is not None
                                else None
                            ),
                            "detections": payload,
                            "nearest_distance_m": (
                                round(nearest, 2) if nearest is not None else None
                            ),
                            "focal_px": focal_px,
                            "fov_deg": fov_deg,
                            "face_scale_px_m": face_scale,
                            "depth_scale": round(self._detector.depth_scale, 3),
                            "depth_enabled": self._detector.depth_enabled,
                            "depth_error": self._detector.last_depth_error,
                            "camera": {
                                "ok": True,
                                "width": actual_width,
                                "height": actual_height,
                                "error": None,
                            },
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    self._condition.notify_all()
        except Exception as exc:
            logger.exception("camera pipeline failed")
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            capture.release()

    @staticmethod
    def _draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for detection in detections:
            box = detection.box
            x1, y1, x2, y2 = (int(round(v)) for v in (box.x1, box.y1, box.x2, box.y2))
            distance = detection.distance_m
            color = (
                (113, 113, 251)
                if distance is not None and distance < 2
                else (36, 191, 251)
                if distance is not None and distance < 5
                else (153, 211, 52)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{detection.priority_level} {detection.label} {detection.confidence:.2f}"
            if distance is not None:
                label += f"  {distance:.2f} m"
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            top = max(0, y1 - text_h - baseline - 10)
            cv2.rectangle(
                frame,
                (x1, top),
                (x1 + text_w + 12, top + text_h + baseline + 10),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 6, top + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (8, 14, 22),
                2,
                cv2.LINE_AA,
            )
        return frame

    def calibrate(
        self,
        actual_distance_m: float,
        label: str | None = None,
        dimension: str = "height",
    ) -> dict:
        """현재 가장 큰 얼굴의 실제 거리로 얼굴 IPD/깊이 스케일을 보정한다."""
        del dimension  # 얼굴 보정은 눈 간격만 사용한다.
        with self._condition:
            measurements = copy.deepcopy(self._last_measurements)
            image_width = int(self._status["camera"]["width"])

        if image_width <= 0:
            raise ValueError("카메라 해상도를 알 수 없습니다.")
        # 신체 부위가 함께 감지되어도 보정은 항상 얼굴(눈 간격)로 한다.
        wanted = (label or "face").strip().lower()
        candidates = [
            item for item in measurements if item.detection.label.strip().lower() == wanted
        ]
        if wanted == "face":
            candidates = [item for item in candidates if item.eye_distance_px is not None]
        if not candidates:
            raise ValueError("보정할 얼굴이 감지되지 않았습니다. 얼굴을 정면으로 보여주세요.")

        # 픽셀 오차를 줄이기 위해 가장 큰(가까운) 얼굴을 사용한다.
        measurement = max(
            candidates,
            key=lambda item: (
                (item.detection.box.x2 - item.detection.box.x1)
                * (item.detection.box.y2 - item.detection.box.y1)
            ),
        )
        return self._detector.calibrate(actual_distance_m, measurement, image_width)

    def latest(self) -> bytes:
        with self._condition:
            return self._latest_jpeg

    def snapshot(self) -> bytes:
        with self._condition:
            return self._snapshot_jpeg

    def wait_for_update(self, timeout: float = 1.0) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)

    def status(self) -> dict:
        with self._condition:
            return copy.deepcopy(self._status)

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


def create_app(config: CameraConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = CameraPipeline(config)
        try:
            yield
        finally:
            app.state.pipeline.stop()

    app = FastAPI(title="NeuroVector Camera Test UI", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/health")
    def health(request: Request) -> dict:
        status = request.app.state.pipeline.status()
        return {
            "ok": status["ok"],
            "error": status["error"],
            "camera": status["camera"],
        }

    @app.get("/api/status")
    def status(request: Request) -> dict:
        return request.app.state.pipeline.status()

    @app.post("/api/calibrate")
    def calibrate(request: Request, payload: CalibrationRequest) -> dict:
        try:
            return request.app.state.pipeline.calibrate(
                payload.actual_distance_m,
                payload.label,
                payload.dimension,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/snapshot.jpg")
    def snapshot(request: Request) -> Response:
        jpeg = request.app.state.pipeline.snapshot()
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/stream.mjpg")
    def stream(request: Request) -> StreamingResponse:
        pipeline = request.app.state.pipeline

        def frames() -> AsyncIterator[bytes]:
            while True:
                jpeg = pipeline.latest()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
                pipeline.wait_for_update(timeout=0.05)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    return app


def parse_args() -> CameraConfig:
    settings = Settings()
    parser = argparse.ArgumentParser(description="NeuroVector local camera test UI")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    parser.add_argument(
        "--model",
        default=settings.face_model_path,
        help="YOLO26 pose weights for face keypoints",
    )
    parser.add_argument(
        "--depth-model",
        default=settings.depth_model_path,
        help="YOLO26 depth weights",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default=settings.device, help="'auto', '0', or 'cpu'")
    parser.add_argument(
        "--calibration",
        default=settings.camera_calibration_path,
        help="checkerboard camera calibration JSON",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=settings.camera_horizontal_fov_deg,
        help="horizontal FOV (degrees), used when no calibration/focal is set",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=settings.camera_focal_length_px,
        help="calibrated focal length in pixels (overrides FOV)",
    )
    parser.add_argument(
        "--face-scale",
        type=float,
        default=settings.face_ipd_scale_px_m,
        help="calibrated face IPD scale (m*px)",
    )
    parser.add_argument("--no-depth", action="store_true", help="disable YOLO26-depth")
    parser.add_argument(
        "--face-only",
        action="store_true",
        help="disable body-part boxes and show face only",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie mirror")
    args = parser.parse_args()
    return CameraConfig(
        camera_index=args.camera,
        face_model_path=args.model,
        depth_model_path=args.depth_model,
        width=args.width,
        height=args.height,
        port=args.port,
        host=args.host,
        fov=args.fov,
        focal_length_px=args.focal,
        face_ipd_scale_px_m=args.face_scale,
        calibration_path=args.calibration,
        depth_enabled=not args.no_depth,
        body_parts_enabled=settings.body_parts_enabled and not args.face_only,
        device=args.device,
        mirror=not args.no_mirror,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_args()
    print(f"camera UI: http://{config.host}:{config.port}")
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
