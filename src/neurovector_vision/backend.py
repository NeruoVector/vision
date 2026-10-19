"""백엔드 API 로 감지 결과를 전송하는 클라이언트."""

from __future__ import annotations

import httpx

from .config import Settings
from .models import Detection, DetectionRequest


class BackendClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._url = settings.backend_detections_url

    async def send_detections(
        self,
        drone_id: str,
        detections: list[Detection],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        """감지 결과를 백엔드로 전송하고 성공 건수를 반환한다."""
        if not detections:
            return 0

        sent = 0
        async with httpx.AsyncClient(timeout=self._settings.backend_timeout_seconds) as client:
            for detection in detections:
                payload = DetectionRequest(
                    drone_id=drone_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    latitude=latitude,
                    longitude=longitude,
                )
                response = await client.post(self._url, json=payload.model_dump(mode="json"))
                response.raise_for_status()
                sent += 1

        return sent
