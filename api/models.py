"""Pydantic schemas returned by the detection API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)
    x2: int = Field(..., ge=0)
    y2: int = Field(..., ge=0)


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox


class DetectionResponse(BaseModel):
    timestamp: str
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    rtsp_url: str


class ErrorResponse(BaseModel):
    detail: str

