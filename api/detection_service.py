"""YOLO inference and RTSP frame handling for the FastAPI app."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
from ultralytics import YOLO

from api.config import Settings
from api.models import BoundingBox, Detection, DetectionResponse, HealthResponse

logger = logging.getLogger(__name__)


class DetectionServiceError(RuntimeError):
    """Base exception for expected detection service failures."""


class ModelLoadError(DetectionServiceError):
    """Raised when the YOLO model cannot be loaded or downloaded."""


class CameraConnectionError(DetectionServiceError):
    """Raised when OpenCV cannot open the RTSP stream."""


class FrameCaptureError(DetectionServiceError):
    """Raised when a valid frame cannot be read from the RTSP stream."""


class InferenceError(DetectionServiceError):
    """Raised when YOLO inference fails."""


class EncodeFrameError(DetectionServiceError):
    """Raised when an annotated frame cannot be encoded as JPEG."""


class DetectionService:
    """Coordinates RTSP capture, YOLO inference, and MJPEG frame rendering."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def safe_rtsp_url(self) -> str:
        """Return the RTSP URL with credentials redacted for logs and health checks."""

        parsed = urlsplit(self.settings.rtsp_url)
        if not parsed.username:
            return self.settings.rtsp_url

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        redacted = f"{parsed.username}:***@{host}"
        return urlunsplit((parsed.scheme, redacted, parsed.path, parsed.query, parsed.fragment))

    def load_model(self) -> Any:
        """Load YOLO once.

        Passing "yolo11n.pt" to the official Ultralytics YOLO API downloads the
        official weight file automatically when it is not present locally.
        """

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                logger.info("Loading YOLO model: %s", self.settings.model_path)
                self._model = YOLO(self.settings.model_path)
                logger.info("YOLO model ready: %s", self.settings.model_path)
                return self._model
            except FileNotFoundError as exc:
                raise ModelLoadError(
                    f"YOLO model '{self.settings.model_path}' was not found and could not be downloaded."
                ) from exc
            except Exception as exc:  # Ultralytics can raise several dependency/network errors here.
                raise ModelLoadError(f"Could not load YOLO model '{self.settings.model_path}': {exc}") from exc

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=self.settings.model_path,
            model_loaded=self.model_loaded,
            rtsp_url=self.safe_rtsp_url,
        )

    def _open_capture(self) -> cv2.VideoCapture:
        """Open a new RTSP capture with OpenCV timeouts where the backend supports them."""

        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.settings.rtsp_open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.settings.rtsp_read_timeout_ms,
        ]

        try:
            cap = cv2.VideoCapture(self.settings.rtsp_url, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            cap = cv2.VideoCapture(self.settings.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.settings.rtsp_open_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.settings.rtsp_read_timeout_ms)

        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(
                f"Could not open RTSP stream {self.safe_rtsp_url}. Check camera power, network, credentials, and path."
            )

        return cap

    def _read_valid_frame(self, cap: cv2.VideoCapture, timeout_seconds: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame
            time.sleep(0.05)

        raise FrameCaptureError("Timed out while waiting for a valid RTSP frame.")

    def capture_frame(self) -> np.ndarray:
        cap = self._open_capture()
        try:
            return self._read_valid_frame(cap, self.settings.detect_read_timeout_seconds)
        finally:
            cap.release()

    def _predict(self, frame: np.ndarray) -> Any:
        model = self.load_model()

        try:
            # Ultralytics model calls are synchronous and can use shared state, so serialize inference.
            with self._inference_lock:
                results = model.predict(
                    source=frame,
                    conf=self.settings.confidence_threshold,
                    imgsz=self.settings.image_size,
                    verbose=False,
                )
        except Exception as exc:
            raise InferenceError(f"YOLO inference failed: {exc}") from exc

        if not results:
            raise InferenceError("YOLO returned no result objects.")
        return results[0]

    def _label_for_class(self, names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _parse_detections(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or getattr(self.load_model(), "names", {})
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            x1, y1, x2, y2 = [max(0, int(round(value))) for value in coords]
            detections.append(
                Detection(
                    label=self._label_for_class(names, int(class_id)),
                    confidence=round(float(confidence), 4),
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    def detect_current_frame(self) -> DetectionResponse:
        frame = self.capture_frame()
        result = self._predict(frame)
        return DetectionResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            detections=self._parse_detections(result),
        )

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self._predict(frame)
        try:
            return result.plot()
        except Exception:
            logger.exception("Ultralytics plot() failed; falling back to manual annotation.")
            annotated = frame.copy()
            for detection in self._parse_detections(result):
                box = detection.bounding_box
                label = f"{detection.label} {detection.confidence:.2f}"
                cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (box.x1, max(20, box.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            return annotated

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality])
        if not ok:
            raise EncodeFrameError("Could not encode annotated frame as JPEG.")
        return buffer.tobytes()

    def _mjpeg_payload(self, frame: np.ndarray) -> bytes:
        annotated = self._annotate_frame(frame)
        jpeg = self._encode_jpeg(annotated)
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
            + jpeg
            + b"\r\n"
        )

    def _status_payload(self, message: str) -> bytes:
        """Render an MJPEG error/status frame so browsers show useful feedback."""

        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(frame, "Ultralytics RTSP stream", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, message[:90], (40, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        jpeg = self._encode_jpeg(frame)
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
            + jpeg
            + b"\r\n"
        )

    async def mjpeg_stream(self) -> AsyncIterator[bytes]:
        """Yield annotated MJPEG frames without blocking FastAPI's event loop."""

        logger.info("Starting MJPEG stream from %s", self.safe_rtsp_url)
        while True:
            cap: cv2.VideoCapture | None = None
            try:
                cap = await asyncio.to_thread(self._open_capture)
                failed_reads = 0

                while True:
                    ok, frame = await asyncio.to_thread(cap.read)
                    if not ok or frame is None or frame.size == 0:
                        failed_reads += 1
                        if failed_reads >= self.settings.stream_max_read_failures:
                            raise FrameCaptureError("Too many invalid RTSP frames while streaming.")
                        await asyncio.sleep(0.05)
                        continue

                    failed_reads = 0
                    try:
                        yield await asyncio.to_thread(self._mjpeg_payload, frame)
                    except DetectionServiceError as exc:
                        logger.exception("Could not render annotated stream frame.")
                        yield await asyncio.to_thread(self._status_payload, str(exc))

                    await asyncio.sleep(self.settings.stream_frame_delay_seconds)

            except asyncio.CancelledError:
                logger.info("MJPEG stream client disconnected.")
                raise
            except DetectionServiceError as exc:
                logger.warning("MJPEG stream error: %s", exc)
                yield await asyncio.to_thread(self._status_payload, str(exc))
                await asyncio.sleep(self.settings.stream_reconnect_delay_seconds)
            except Exception as exc:
                logger.exception("Unexpected MJPEG stream error.")
                yield await asyncio.to_thread(self._status_payload, f"Unexpected stream error: {exc}")
                await asyncio.sleep(self.settings.stream_reconnect_delay_seconds)
            finally:
                if cap is not None:
                    await asyncio.to_thread(cap.release)

