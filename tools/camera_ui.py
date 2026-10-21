"""로컬 카메라 + YOLO 실시간 프리뷰 테스트 UI.

브라우저에서 ``http://localhost:8090`` 을 열면 카메라 영상과 감지 박스,
클래스별 거리, 추론 시간/FPS 를 볼 수 있다. 운영 API 가 아니라 로컬
카메라/모델/거리 추정을 빠르게 확인하기 위한 개발 도구다.

실행:
    uv run --extra vision python tools/camera_ui.py
    uv run --extra vision python tools/camera_ui.py --camera 0 --port 8090
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

from neurovector_vision.config import Settings
from neurovector_vision.detector import ObstacleDetector
from neurovector_vision.distance import (
    focal_length_from_reference,
    horizontal_fov_deg,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from neurovector_vision.models import Detection

logger = logging.getLogger(__name__)
HTML = Path(__file__).with_name("camera_ui.html").read_text(encoding="utf-8")


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    model_path: str = "yolov8n.pt"
    width: int = 1280
    height: int = 720
    port: int = 8090
    host: str = "127.0.0.1"
    fov: float = 70.0
    focal_length_px: float | None = None
    mirror: bool = True


class CalibrationRequest(BaseModel):
    actual_distance_m: float = Field(gt=0.0)
    label: str | None = None
    dimension: str = "height"


class CameraPipeline:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._settings = Settings(
            _env_file=None,
            model_path=config.model_path,
            camera_horizontal_fov_deg=config.fov,
            camera_focal_length_px=config.focal_length_px,
        )
        self._detector = ObstacleDetector(self._settings)
        self._condition = threading.Condition()
        self._latest_jpeg = self._placeholder("Preparing camera and model...")
        self._snapshot_jpeg = self._latest_jpeg
        self._last_detections: list[Detection] = []
        self._status: dict = {
            "ok": False,
            "error": None,
            "frames": 0,
            "fps": 0.0,
            "inference_ms": None,
            "detections": [],
            "nearest_distance_m": None,
            "focal_px": None,
            "fov_deg": None,
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
            self._condition.notify_all()

        if not capture.isOpened():
            self._fail("카메라를 열 수 없습니다. --camera 인덱스와 권한을 확인하세요.")
            return

        try:
            # 첫 요청에서 모델 로딩 지연이 발생하지 않도록 미리 준비한다.
            self._detector._load_model()
            with self._condition:
                self._latest_jpeg = self._placeholder("Waiting for the first frame...")
                self._condition.notify_all()

            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                started = time.perf_counter()
                if self._config.mirror:
                    frame = cv2.flip(frame, 1)
                ok, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    continue
                detections = self._detector.detect(raw.tobytes())
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
                        "label": detection.label,
                        "confidence": round(detection.confidence, 4),
                        "distance_m": detection.distance_m,
                        "distance_note": detection.distance_note,
                        "box": detection.box.model_dump(),
                    }
                    for detection in detections
                ]
                focal_px, fov_deg = self._intrinsics(actual_width)

                with self._condition:
                    self._latest_jpeg = encoded.tobytes()
                    self._snapshot_jpeg = self._latest_jpeg
                    self._last_detections = detections
                    self._status.update(
                        {
                            "ok": True,
                            "error": None,
                            "frames": self._status["frames"] + 1,
                            "fps": round(fps, 2),
                            "inference_ms": round(inference_ms, 1),
                            "detections": payload,
                            "nearest_distance_m": (
                                round(nearest, 2) if nearest is not None else None
                            ),
                            "focal_px": focal_px,
                            "fov_deg": fov_deg,
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
            label = f"{detection.label} {detection.confidence:.2f}"
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
        """현재 감지된 기준 객체로 초점 거리를 보정한다."""
        if dimension not in {"width", "height"}:
            raise ValueError("dimension must be 'width' or 'height'")
        if actual_distance_m <= 0.0:
            raise ValueError("actual_distance_m must be positive")

        with self._condition:
            detections = copy.deepcopy(self._last_detections)
            image_width = int(self._status["camera"]["width"])
            image_height = int(self._status["camera"]["height"])

        if image_width <= 0 or image_height <= 0:
            raise ValueError("카메라 해상도를 알 수 없습니다.")

        candidates = [detection for detection in detections if detection.distance_m is not None]
        if label:
            wanted = label.strip().lower()
            candidates = [
                detection for detection in candidates if detection.label.strip().lower() == wanted
            ]
        if not candidates:
            raise ValueError("보정할 수 있는 신뢰 가능한 감지가 없습니다.")

        # 픽셀 오차를 줄이기 위해 가장 큰(가까운) 바운딩 박스를 사용한다.
        detection = max(
            candidates,
            key=lambda item: (item.box.x2 - item.box.x1) * (item.box.y2 - item.box.y1),
        )
        size = self._settings.resolved_object_sizes.get(detection.label.strip().lower())
        if size is None:
            raise ValueError(f"'{detection.label}' 클래스의 실제 크기 정보가 없습니다.")

        margin_x = max(2.0, image_width * self._settings.distance_edge_margin_ratio)
        margin_y = max(2.0, image_height * self._settings.distance_edge_margin_ratio)
        if dimension == "width":
            if detection.box.x1 <= margin_x or detection.box.x2 >= image_width - margin_x:
                raise ValueError("객체 너비가 화면 경계에서 잘려 보정할 수 없습니다.")
            bbox_dimension = detection.box.x2 - detection.box.x1
            real_dimension = size.width_m
        else:
            if detection.box.y1 <= margin_y or detection.box.y2 >= image_height - margin_y:
                raise ValueError("객체 높이가 화면 경계에서 잘려 보정할 수 없습니다.")
            bbox_dimension = detection.box.y2 - detection.box.y1
            real_dimension = size.height_m

        previous_focal = self._detector.focal_px(image_width)
        focal = focal_length_from_reference(
            actual_distance_m,
            bbox_dimension,
            real_dimension,
        )
        self._detector.set_focal_length_px(focal)
        with self._condition:
            self._condition.notify_all()

        return {
            "label": detection.label,
            "dimension": dimension,
            "actual_distance_m": round(actual_distance_m, 2),
            "previous_focal_px": round(previous_focal, 2),
            "focal_px": round(focal, 2),
            "fov_deg": round(horizontal_fov_deg(image_width, focal), 2),
            "recommended_env": f"NEUROVECTOR_CAMERA_FOCAL_LENGTH_PX={focal:.2f}",
        }

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
    settings = Settings(_env_file=None)
    parser = argparse.ArgumentParser(description="NeuroVector local camera test UI")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 camera index")
    parser.add_argument("--model", default=settings.model_path, help="YOLO weights path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fov",
        type=float,
        default=settings.camera_horizontal_fov_deg,
        help="horizontal FOV (degrees)",
    )
    parser.add_argument(
        "--focal",
        type=float,
        default=settings.camera_focal_length_px,
        help="calibrated focal length in pixels (overrides --fov)",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie mirror")
    args = parser.parse_args()
    return CameraConfig(
        camera_index=args.camera,
        model_path=args.model,
        width=args.width,
        height=args.height,
        port=args.port,
        host=args.host,
        fov=args.fov,
        focal_length_px=args.focal,
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
