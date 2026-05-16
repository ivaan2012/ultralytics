"""Runtime configuration for the RTSP detection API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


DEFAULT_RTSP_URL = "rtsp://admin:DaMi90%23@192.168.0.64:554/stream1"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    rtsp_url: str
    model_path: str
    confidence_threshold: float
    image_size: int
    rtsp_open_timeout_ms: int
    rtsp_read_timeout_ms: int
    detect_read_timeout_seconds: float
    stream_frame_delay_seconds: float
    stream_reconnect_delay_seconds: float
    stream_max_read_failures: int
    jpeg_quality: int
    preload_model: bool
    log_level: str


@lru_cache
def get_settings() -> Settings:
    """Read settings once so env vars can override defaults without adding extra dependencies."""

    return Settings(
        rtsp_url=os.getenv("YOLO_RTSP_URL", DEFAULT_RTSP_URL),
        model_path=os.getenv("YOLO_MODEL_PATH", "yolo11n.pt"),
        confidence_threshold=_env_float("YOLO_CONFIDENCE_THRESHOLD", 0.25),
        image_size=_env_int("YOLO_IMAGE_SIZE", 640),
        rtsp_open_timeout_ms=_env_int("YOLO_RTSP_OPEN_TIMEOUT_MS", 5000),
        rtsp_read_timeout_ms=_env_int("YOLO_RTSP_READ_TIMEOUT_MS", 5000),
        detect_read_timeout_seconds=_env_float("YOLO_DETECT_READ_TIMEOUT_SECONDS", 6.0),
        stream_frame_delay_seconds=_env_float("YOLO_STREAM_FRAME_DELAY_SECONDS", 0.03),
        stream_reconnect_delay_seconds=_env_float("YOLO_STREAM_RECONNECT_DELAY_SECONDS", 2.0),
        stream_max_read_failures=_env_int("YOLO_STREAM_MAX_READ_FAILURES", 30),
        jpeg_quality=_env_int("YOLO_JPEG_QUALITY", 85),
        preload_model=os.getenv("YOLO_PRELOAD_MODEL", "true").lower() in {"1", "true", "yes", "on"},
        log_level=os.getenv("YOLO_LOG_LEVEL", "INFO"),
    )

