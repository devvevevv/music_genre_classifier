"""Predictor behaviour tests."""

from __future__ import annotations

import pytest

from predictor import predict, predict_detailed


def test_predict_returns_normalised_probabilities(sample_wav):
    probabilities = predict(str(sample_wav))

    assert probabilities
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    # highest probability first
    values = list(probabilities.values())
    assert values == sorted(values, reverse=True)


def test_predict_detailed_reports_chunks(sample_wav):
    result = predict_detailed(str(sample_wav))

    assert result["n_chunks"] >= 1
    assert len(result["chunks"]) == result["n_chunks"]
    assert result["duration"] > 0
    for chunk in result["chunks"]:
        assert chunk["end"] >= chunk["start"]
        assert 0.0 <= chunk["confidence"] <= 1.0


def test_predict_rejects_unreadable_audio(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"definitely not audio")

    with pytest.raises(ValueError):
        predict(str(bad))
