"""FastAPI entrypoint for real-time Ultralytics YOLO detection over RTSP."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from api.config import get_settings
from api.detection_service import (
    CameraConnectionError,
    DetectionService,
    DetectionServiceError,
    EncodeFrameError,
    FrameCaptureError,
    InferenceError,
    ModelLoadError,
)
from api.models import DetectionResponse, ErrorResponse, HealthResponse

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

detection_service = DetectionService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.preload_model:
        try:
            # Loading yolo11n.pt here triggers the official Ultralytics auto-download if needed.
            await asyncio.to_thread(detection_service.load_model)
        except ModelLoadError:
            logger.exception("YOLO model preload failed; /detect and /stream will retry on demand.")
    yield


app = FastAPI(
    title="Ultralytics YOLO RTSP Detection API",
    description="FastAPI server for object detection from an IP camera RTSP stream.",
    version="1.0.0",
    lifespan=lifespan,
)


def _service_error_to_http(exc: DetectionServiceError) -> HTTPException:
    if isinstance(exc, CameraConnectionError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if isinstance(exc, FrameCaptureError):
        return HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc))
    if isinstance(exc, (ModelLoadError, InferenceError, EncodeFrameError)):
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected detection service error.")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return detection_service.health()


@app.get(
    "/detect",
    response_model=DetectionResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def detect() -> DetectionResponse:
    try:
        return await asyncio.to_thread(detection_service.detect_current_frame)
    except DetectionServiceError as exc:
        logger.exception("Detection request failed.")
        raise _service_error_to_http(exc) from exc


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        detection_service.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

