"""Pydantic response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkPrediction(BaseModel):
    index: int
    start: float = Field(..., description="Chunk start time in seconds")
    end: float = Field(..., description="Chunk end time in seconds")
    genre: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    genre: str = Field(..., examples=["metal"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.51])
    probabilities: dict[str, float]
    chunks: list[ChunkPrediction]
    n_chunks: int
    duration: float
    latency_ms: float
    cached: bool
    spectrogram: str = Field(..., description="data:image/png;base64,...")


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_path: str
    cache_backend: str


class ModelInfoResponse(BaseModel):
    version: str
    model_type: str
    classes: list[str]
    metrics: dict[str, float]
    git_sha: str | None = None
