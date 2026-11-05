"""Forwarding preserves payloads and uses its caller-owned HTTP connection pool."""

import anyio
import httpx
import pytest

from neurovector_vision.backend import BackendClient
from neurovector_vision.config import Settings
from neurovector_vision.models import BoundingBox, Detection, DetectionRequest


def detection() -> Detection:
    return Detection(
        label="person",
        confidence=0.9,
        priority_level=1,
        distance_m=4.2,
        box=BoundingBox(x1=1, y1=2, x2=30, y2=40),
    )


def test_multiple_batches_share_client_and_preserve_payload() -> None:
    observed: list[DetectionRequest] = []

    def receive(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://backend.test/api/v1/detections"
        observed.append(DetectionRequest.model_validate_json(request.content))
        return httpx.Response(201, json={})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as http:
            backend = BackendClient(
                Settings(_env_file=None, backend_base_url="http://backend.test/"), http
            )
            assert (
                await backend.send_detections("school", [detection(), detection()], 37.0, 127.0)
                == 2
            )
            assert await backend.send_detections("school", [detection()]) == 1
            assert not http.is_closed
        assert http.is_closed

    anyio.run(scenario)
    assert len(observed) == 3
    assert observed[0].drone_id == "school"
    assert observed[0].latitude == 37.0 and observed[0].longitude == 127.0
    assert observed[2].latitude is None and observed[2].longitude is None
    assert observed[0].distance_m == 4.2 and observed[0].priority_level == 1
    assert observed[0].detected_at.tzinfo is not None


def test_empty_batch_does_not_send_a_request() -> None:
    calls: list[httpx.Request] = []

    def receive(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(201)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as http:
            backend = BackendClient(Settings(_env_file=None), http)
            assert await backend.send_detections("school", []) == 0

    anyio.run(scenario)
    assert not calls


def test_first_backend_error_stops_the_batch() -> None:
    calls: list[httpx.Request] = []

    def receive(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as http:
            backend = BackendClient(Settings(_env_file=None), http)
            with pytest.raises(httpx.HTTPStatusError):
                await backend.send_detections("school", [detection(), detection()])

    anyio.run(scenario)
    assert len(calls) == 1
