"""Reusable music-genre prediction.

Loads the pickled scikit-learn classifier produced by ``eval/train.py`` and
returns a probability for every known genre. Long tracks are split into
overlapping chunks whose per-chunk probabilities are averaged, which is more
informative than a single majority-vote label.
"""

from __future__ import annotations

import functools
import os
import pickle
import sys

import librosa
import numpy as np
import pandas as pd

# feature_extraction.py / config.py live next to this file but are imported as
# top-level modules throughout this project, so make sure src/ is importable.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import (  # noqa: E402
    CHUNK_SIZE,
    OVERLAP,
    SAMPLE_RATE,
    resolve_model_path,
)
from feature_extraction import feature_extraction  # noqa: E402

MIN_CHUNK_SAMPLES = 2048  # chroma STFT needs at least this many samples


@functools.lru_cache(maxsize=4)
def load_model(model_path: str | None = None):
    """Load and cache the pickled classifier.

    When ``model_path`` is omitted the production model is resolved via
    ``config.resolve_model_path()`` (env ``MODEL_PATH`` → ``models/model.pkl``
    → legacy ``model.pkl``).
    """
    path = str(model_path) if model_path else str(resolve_model_path())
    with open(path, "rb") as f:
        return pickle.load(f)


def _chunk_probabilities(model, y_chunk: np.ndarray, sr: int):
    features = feature_extraction(y_chunk, sr)
    if features is None:
        return None
    frame = pd.DataFrame([features])
    return model.predict_proba(frame)[0]


def _iter_chunks(y: np.ndarray, sr: int):
    """Yield ``(start_sample, chunk)``; short clips are evaluated whole."""
    chunk_samples = CHUNK_SIZE * sr
    step_samples = (CHUNK_SIZE - OVERLAP) * sr

    if len(y) <= chunk_samples + sr:
        yield 0, y
        return

    for start in range(0, len(y), step_samples):
        chunk = y[start : start + chunk_samples]
        if len(chunk) < MIN_CHUNK_SAMPLES:
            continue
        yield start, chunk


def _analyze(
    audio_path: str, model_path: str | None = None
) -> tuple[dict[str, float], list[dict], float]:
    model = load_model(model_path)
    try:
        y, sr_float = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as exc:
        raise ValueError(f"Could not read audio file '{audio_path}': {exc}") from exc
    if y is None or len(y) == 0:
        raise ValueError(f"Could not read audio data from '{audio_path}'")
    sr = int(sr_float)

    prob_sum = np.zeros(len(model.classes_), dtype=float)
    chunks: list[dict] = []
    for start, chunk in _iter_chunks(y, sr):
        probs = _chunk_probabilities(model, chunk, sr)
        if probs is None:
            continue
        prob_sum += probs
        chunks.append(
            {
                "index": len(chunks),
                "start": round(start / sr, 3),
                "end": round((start + len(chunk)) / sr, 3),
                "genre": str(model.classes_[int(np.argmax(probs))]),
                "confidence": float(np.max(probs)),
            }
        )

    if not chunks:
        raise ValueError("No valid predictions could be made for this audio file")

    mean_probs = prob_sum / len(chunks)
    probabilities = dict(
        sorted(
            ((str(genre), float(p)) for genre, p in zip(model.classes_, mean_probs, strict=True)),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    return probabilities, chunks, round(len(y) / sr, 3)


def predict(audio_path: str, model_path: str | None = None) -> dict[str, float]:
    """Return ``{genre: probability}`` for an audio file, highest first."""
    probabilities, _, _ = _analyze(audio_path, model_path)
    return probabilities


def predict_detailed(audio_path: str, model_path: str | None = None) -> dict:
    """Return probabilities plus the per-chunk breakdown used by the dashboard."""
    probabilities, chunks, duration = _analyze(audio_path, model_path)
    return {
        "probabilities": probabilities,
        "chunks": chunks,
        "n_chunks": len(chunks),
        "duration": duration,
    }
