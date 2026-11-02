"""Spectrogram rendering for the dashboard / API."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless rendering (server / Docker)

import os  # noqa: E402
import sys  # noqa: E402

import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SAMPLE_RATE  # noqa: E402


def spectrogram_png(audio_path: str, max_seconds: float = 30.0) -> bytes:
    """Render a log-frequency spectrogram of the first ``max_seconds`` as PNG."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=max_seconds)
    except Exception as exc:
        raise ValueError(f"Could not read audio file '{audio_path}': {exc}") from exc
    if y is None or len(y) == 0:
        raise ValueError(f"Could not read audio data from '{audio_path}'")

    db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    image = librosa.display.specshow(db, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma")
    ax.set_title("Spectrogram")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
