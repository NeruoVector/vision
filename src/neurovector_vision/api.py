"""FastAPI 앱: 장애물 감지 API."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, UploadFile

from . import __version__
from .backend import BackendClient
from .config import Settings, get_settings
from .detector import InvalidImage, ObstacleDetector, VisionBackendUnavailable
from .models import DetectResponse

app = FastAPI(
    title="NeuroVector Vision",
    description="드론 영상에서 장애물을 감지하고 백엔드로 전달하는 서비스",
    version=__version__,
)


def _settings() -> Settings:
    return get_settings()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile) -> DetectResponse:
    """업로드된 이미지에서 장애물을 감지하고 백엔드로 전송한다."""
    settings = _settings()
    detector = ObstacleDetector(settings)
    client = BackendClient(settings)

    started = time.perf_counter()
    payload = await file.read()

    try:
        detections = detector.detect(payload)
    except VisionBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    forwarded = await client.send_detections(settings.drone_id, detections)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return DetectResponse(
        drone_id=settings.drone_id,
        detections=detections,
        forwarded=forwarded,
        elapsed_ms=round(elapsed_ms, 2),
    )
