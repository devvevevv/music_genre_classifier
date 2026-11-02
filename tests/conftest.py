"""Shared pytest fixtures.

Tests run without the real dataset or model: a synthetic tone is generated and
a tiny throwaway classifier is fitted on its features, then pointed to via the
``MODEL_PATH`` env var.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _path in (str(ROOT_DIR), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import SAMPLE_RATE  # noqa: E402

LABELS = ["blues", "metal", "pop"]


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory) -> Path:
    """A 6-second synthetic stereo-free tone, long enough for chunking."""
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    t = np.linspace(0, 6, int(SAMPLE_RATE * 6), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    sf.write(path, y, SAMPLE_RATE)
    return path


@pytest.fixture(scope="session")
def tiny_model(sample_wav: Path, tmp_path_factory) -> Path:
    """Fit a small RandomForest on noised copies of the sample's features."""
    import librosa
    from sklearn.ensemble import RandomForestClassifier

    from feature_extraction import feature_extraction

    y, sr = librosa.load(sample_wav, sr=SAMPLE_RATE)
    base = feature_extraction(y, sr)
    assert base is not None
    columns = list(base)

    rng = np.random.default_rng(0)
    rows = [
        {k: base[k] + rng.normal(0, abs(base[k]) * 0.1 + 1e-6) for k in columns} for _ in range(60)
    ]
    X = pd.DataFrame(rows, columns=columns)
    labels = [LABELS[i % len(LABELS)] for i in range(len(rows))]

    model = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, labels)
    path = tmp_path_factory.mktemp("model") / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    return path


@pytest.fixture(scope="session", autouse=True)
def _use_tiny_model(tiny_model: Path):
    os.environ["MODEL_PATH"] = str(tiny_model)
    import predictor

    predictor.load_model.cache_clear()
    yield str(tiny_model)
    predictor.load_model.cache_clear()
    os.environ.pop("MODEL_PATH", None)
