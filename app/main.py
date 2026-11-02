"""FastAPI service exposing the music genre classifier."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

# src/ modules are imported as top-level names, so put src/ on the path.
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cache import get_cache  # noqa: E402
from config import METADATA_PATH, METRICS_PATH, resolve_model_path  # noqa: E402
from predictor import predict_detailed  # noqa: E402
from settings import get_settings  # noqa: E402
from spectrogram import spectrogram_png  # noqa: E402

from .observability import (  # noqa: E402
    PREDICTIONS,
    RequestContextMiddleware,
    configure_logging,
)
from .schemas import (  # noqa: E402
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version="2.0.0")
app.add_middleware(RequestContextMiddleware)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

v1 = APIRouter(prefix="/api/v1", tags=["v1"])


def _validate_upload(file: UploadFile, data: bytes) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extension_set:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix or 'unknown'}'. "
                f"Allowed: {sorted(settings.allowed_extension_set)}"
            ),
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )


async def _classify(file: UploadFile) -> PredictResponse:
    data = await file.read()
    _validate_upload(file, data)

    cache = get_cache()
    cache_key = hashlib.sha256(data).hexdigest() + ":" + str(resolve_model_path())

    suffix = Path(file.filename or "audio").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()

        started = time.perf_counter()
        inference = cache.get(cache_key)
        cached = inference is not None
        if inference is None:
            # CPU-bound: keep the event loop free.
            inference = await run_in_threadpool(predict_detailed, tmp.name)
            cache.set(cache_key, inference)
        spectrogram = await run_in_threadpool(spectrogram_png, tmp.name)
        latency_ms = (time.perf_counter() - started) * 1000
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    PREDICTIONS.labels("hit" if cached else "miss").inc()
    probabilities = inference["probabilities"]
    top_genre = next(iter(probabilities))
    return PredictResponse(
        genre=top_genre,
        confidence=probabilities[top_genre],
        probabilities=probabilities,
        chunks=inference["chunks"],
        n_chunks=inference["n_chunks"],
        duration=inference["duration"],
        latency_ms=round(latency_ms, 1),
        cached=cached,
        spectrogram="data:image/png;base64," + base64.b64encode(spectrogram).decode("ascii"),
    )


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the single-page dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@v1.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe that also reports model availability and cache backend."""
    model_path = resolve_model_path()
    return HealthResponse(
        status="ok",
        model_available=os.path.exists(model_path),
        model_path=str(model_path),
        cache_backend=get_cache().backend,
    )


@v1.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> JSONResponse:
    """Return the production model manifest (version, metrics, classes)."""
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No model metadata found. Train one with `python -m eval.train`.",
        )
    return JSONResponse(json.loads(METADATA_PATH.read_text()))


@v1.get("/metrics/summary")
def metrics_summary() -> JSONResponse:
    """Return the cross-validated model comparison if it has been generated."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run `python -m eval.evaluate`.",
        )
    return JSONResponse(json.loads(METRICS_PATH.read_text()))


@v1.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded audio file and return genre probabilities."""
    return await _classify(file)


# --- Backwards-compatible unversioned aliases -------------------------------
app.include_router(v1)
app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
app.add_api_route("/model/info", model_info, methods=["GET"])
app.add_api_route("/metrics/summary", metrics_summary, methods=["GET"])
app.add_api_route("/predict", predict_endpoint, methods=["POST"], response_model=PredictResponse)
