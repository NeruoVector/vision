"""FastAPI app with a bounded, serialized inference worker per process."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile

from . import __version__
from .backend import BackendClient
from .config import Settings, get_settings
from .detector import InvalidImage, VisionBackendUnavailable
from .models import DetectResponse
from .pipeline import VisionPipeline

logger = logging.getLogger(__name__)


def _settings() -> Settings:
    return get_settings()


class InferenceRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = VisionPipeline(settings)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
        self.slots = asyncio.Semaphore(settings.inference_queue_size)
        self.error: str | None = None

    async def infer(self, payload: bytes):
        try:
            await asyncio.wait_for(
                self.slots.acquire(), timeout=self.settings.inference_queue_timeout_s
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail="추론 대기열이 가득 찼습니다.") from exc
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self.executor, self.pipeline.detect, payload)
        except BaseException:
            self.slots.release()
            raise

        # Cancellation of an HTTP request must not release capacity while its worker still runs.
        def completed(future):
            self.slots.release()
            if future.cancelled():
                return
            # Consume exceptions even when the HTTP request has already been cancelled.
            error = future.exception()
            if error is None:
                self.error = None
            elif not isinstance(error, InvalidImage):
                self.error = f"{type(error).__name__}: {error}"

        future.add_done_callback(completed)
        return await asyncio.shield(future)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = InferenceRuntime(_settings())
    app.state.runtime = runtime
    try:
        if runtime.settings.warmup_on_startup:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    runtime.executor, runtime.pipeline.warmup
                )
            except Exception as exc:
                runtime.error = f"{type(exc).__name__}: {exc}"
                logger.exception("Vision warmup failed; inference will retry on the next request")
        yield
    finally:
        await asyncio.to_thread(runtime.executor.shutdown, wait=True, cancel_futures=True)
        del app.state.runtime


app = FastAPI(
    title="NeuroVector Vision",
    description="드론 영상에서 장애물을 감지하고 백엔드로 전달하는 서비스",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict:
    runtime = getattr(request.app.state, "runtime", None)
    depth = getattr(runtime.pipeline, "depth", None) if runtime is not None else None
    depth_error = getattr(depth, "last_error", None)
    return {
        "status": "degraded" if runtime is not None and (runtime.error or depth_error) else "ok",
        "version": __version__,
        "inference_error": runtime.error if runtime is not None else None,
        "depth_error": depth_error,
        "models": {
            "object": runtime.settings.model_path,
            "pose": runtime.settings.face_model_path if runtime.settings.face_enabled else None,
            "depth": runtime.settings.depth_model_path if runtime.settings.depth_enabled else None,
        }
        if runtime is not None
        else None,
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request, file: UploadFile) -> DetectResponse:
    runtime: InferenceRuntime = request.app.state.runtime
    settings = runtime.settings
    started = time.perf_counter()
    payload = await file.read()
    try:
        detections = await runtime.infer(payload)
    except VisionBackendUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logger.exception("Vision inference failed")
        raise HTTPException(status_code=503, detail="AI 추론을 사용할 수 없습니다.") from exc

    try:
        forwarded = await BackendClient(settings).send_detections(settings.drone_id, detections)
    except httpx.HTTPError as exc:
        logger.warning("Backend forwarding failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="감지 결과를 백엔드로 전송하지 못했습니다."
        ) from exc

    return DetectResponse(
        drone_id=settings.drone_id,
        detections=detections,
        forwarded=forwarded,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
