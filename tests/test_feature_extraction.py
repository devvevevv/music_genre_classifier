"""Feature extraction contract tests."""

from __future__ import annotations

import librosa
import numpy as np

from config import SAMPLE_RATE
from feature_extraction import feature_extraction

EXPECTED_FEATURES = 90


def test_feature_vector_length_and_finiteness(sample_wav):
    y, sr = librosa.load(sample_wav, sr=SAMPLE_RATE)
    features = feature_extraction(y, sr)

    assert features is not None
    assert len(features) == EXPECTED_FEATURES
    assert all(np.isfinite(value) for value in features.values())


def test_returns_none_on_empty_audio():
    assert feature_extraction(np.array([]), SAMPLE_RATE) is None
