"""API contract tests (no real dataset/model required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def upload(path, name="tone.wav", content_type="audio/wav"):
    with open(path, "rb") as fh:
        return client.post("/api/v1/predict", files={"file": (name, fh, content_type)})


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is True


def test_predict_contract(sample_wav):
    response = upload(sample_wav)
    assert response.status_code == 200
    body = response.json()

    assert set(body) >= {
        "genre",
        "confidence",
        "probabilities",
        "chunks",
        "n_chunks",
        "duration",
        "latency_ms",
        "cached",
        "spectrogram",
    }
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert body["spectrogram"].startswith("data:image/png;base64,")


def test_second_request_is_cached(sample_wav):
    upload(sample_wav)
    response = upload(sample_wav)
    assert response.json()["cached"] is True


def test_unversioned_alias(sample_wav):
    with open(sample_wav, "rb") as fh:
        response = client.post("/predict", files={"file": ("tone.wav", fh, "audio/wav")})
    assert response.status_code == 200


def test_unsupported_extension():
    response = client.post("/api/v1/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_corrupt_audio_returns_400(tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio data")
    response = upload(bad, name="broken.mp3", content_type="audio/mpeg")
    assert response.status_code == 400


def test_model_info():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    assert "version" in response.json()


def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mgc_requests_total" in response.text
