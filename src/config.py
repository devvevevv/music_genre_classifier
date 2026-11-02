"""Central configuration shared by training, evaluation, and inference.

All audio is loaded at ``SAMPLE_RATE`` in both the training and inference
paths so the model sees the same feature distribution at fit and predict time.
"""

from __future__ import annotations

import os
from pathlib import Path

# Audio is standardised on a single sample rate across training and inference.
SAMPLE_RATE = 22050
CHUNK_SIZE = 30  # seconds
OVERLAP = 5  # seconds

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(os.environ.get("DATASET_DIR", ROOT_DIR / "dataset" / "genres_original"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT_DIR / "models"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", MODELS_DIR / "artifacts"))
FEATURES_CACHE = Path(os.environ.get("FEATURES_CACHE", MODELS_DIR / "features.csv"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "model.pkl"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", MODELS_DIR / "metadata.json"))
METRICS_PATH = Path(os.environ.get("METRICS_PATH", MODELS_DIR / "metrics.json"))
RANDOM_STATE = 42


def resolve_model_path() -> Path:
    """Return the production model path, falling back to the legacy root file."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    if MODEL_PATH.exists():
        return MODEL_PATH
    # Legacy location from before the models/ registry existed.
    return ROOT_DIR / "model.pkl"
