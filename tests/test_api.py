import pytest
from fastapi.testclient import TestClient

from neurovector_vision.api import app
from neurovector_vision.config import Settings
from neurovector_vision.models import BoundingBox, Detection


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.port == 8082
    assert settings.backend_detections_url.endswith("/api/v1/detections")


def test_detection_model_validation() -> None:
    detection = Detection(
        label="tree",
        confidence=0.9,
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        distance_m=4.2,
    )
    assert detection.label == "tree"
    assert detection.distance_m == 4.2
    assert detection.priority_level == 2


def test_detect_returns_distance_and_forwards_it(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, warmup_on_startup=False)
    forwarded: list[Detection] = []

    class FakeDetector:
        def __init__(self, _settings: Settings) -> None:
            pass

        def detect(self, _payload: bytes) -> list[Detection]:
            return [
                Detection(
                    label="tree",
                    confidence=0.9,
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=200),
                    distance_m=6.4,
                    priority_level=1,
                )
            ]

    async def fake_send_detections(
        _self: object,
        _drone_id: str,
        detections: list[Detection],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        forwarded.extend(detections)
        return len(detections)

    monkeypatch.setattr("neurovector_vision.api._settings", lambda: settings)
    monkeypatch.setattr("neurovector_vision.api.VisionPipeline", FakeDetector)
    monkeypatch.setattr(
        "neurovector_vision.api.BackendClient.send_detections",
        fake_send_detections,
    )

    with client:
        response = client.post(
            "/detect",
            files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["detections"][0]["distance_m"] == 6.4
    assert response.json()["detections"][0]["priority_level"] == 1
    assert forwarded[0].distance_m == 6.4
    assert forwarded[0].priority_level == 1


def test_api_reuses_worker_and_reports_backend_failure(monkeypatch):
    import httpx

    instances = []

    class FakePipeline:
        def __init__(self, settings):
            instances.append(self)
            self.calls = 0

        def warmup(self):
            pass

        def detect(self, payload):
            self.calls += 1
            return []

    async def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("neurovector_vision.api.VisionPipeline", FakePipeline)
    monkeypatch.setattr("neurovector_vision.api._settings", lambda: Settings(_env_file=None))
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post("/detect", files={"file": ("x.jpg", b"x")}).status_code == 200
        assert len(instances) == 1
        assert instances[0].calls == 2
        monkeypatch.setattr("neurovector_vision.api.BackendClient.send_detections", fail)
        assert client.post("/detect", files={"file": ("x.jpg", b"x")}).status_code == 502


def test_cancelled_request_retains_worker_capacity(monkeypatch):
    import asyncio
    import threading

    from fastapi import HTTPException

    from neurovector_vision.api import InferenceRuntime

    started = threading.Event()
    finish = threading.Event()

    class BlockingPipeline:
        def __init__(self, settings):
            pass

        def detect(self, payload):
            started.set()
            assert finish.wait(timeout=5)
            return []

    monkeypatch.setattr("neurovector_vision.api.VisionPipeline", BlockingPipeline)

    async def scenario():
        runtime = InferenceRuntime(
            Settings(
                _env_file=None,
                inference_queue_size=1,
                inference_queue_timeout_s=0.02,
            )
        )
        try:
            task = asyncio.create_task(runtime.infer(b"x"))
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(HTTPException) as error:
                await runtime.infer(b"x")
            assert error.value.status_code == 503
        finally:
            finish.set()
            await asyncio.to_thread(runtime.executor.shutdown, wait=True)

    asyncio.run(scenario())


def test_invalid_upload_does_not_degrade_health(monkeypatch):
    from neurovector_vision.detector import InvalidImage

    class InvalidPipeline:
        def __init__(self, settings):
            pass

        def detect(self, payload):
            raise InvalidImage("invalid image")

    monkeypatch.setattr("neurovector_vision.api.VisionPipeline", InvalidPipeline)
    monkeypatch.setattr(
        "neurovector_vision.api._settings",
        lambda: Settings(_env_file=None, warmup_on_startup=False),
    )
    with TestClient(app) as client:
        response = client.post("/detect", files={"file": ("x.jpg", b"invalid")})
        assert response.status_code == 400
        assert client.get("/health").json()["status"] == "ok"
