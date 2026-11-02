"""Train, calibrate, and register the production genre classifier.

Picks the best candidate from ``eval/evaluate.py``, wraps it in
``CalibratedClassifierCV`` so the probabilities exposed by the API are
trustworthy, and writes the model plus a metadata manifest to ``models/``.

Run from the repository root:

    python -m eval.train
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, log_loss  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _path in (str(ROOT_DIR), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import (  # noqa: E402
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SAMPLE_RATE,
)
from eval.evaluate import candidate_models, run_evaluation  # noqa: E402
from eval.features import load_features, split_xy  # noqa: E402

MODEL_VERSION = "1.0.0"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - not a git checkout / git missing
        return "unknown"


def _multiclass_brier(y_true, probabilities, classes) -> float:
    onehot = np.zeros_like(probabilities)
    index = {label: i for i, label in enumerate(classes)}
    for row, label in enumerate(y_true):
        onehot[row, index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def _reliability_curve(y_true, probabilities, classes, path: Path, bins: int = 10) -> None:
    """Top-label reliability diagram (confidence vs empirical accuracy)."""
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    centres, observed, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        observed.append(correct[mask].mean())
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(centres, observed, "o-", label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Top-label reliability diagram")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(model_name: str | None = None, cv: int = 5) -> dict:
    """Fit the calibrated production model and write model + metadata."""
    df = load_features()
    X, y = split_xy(df)

    if model_name is None:
        if METRICS_PATH.exists():
            summary = json.loads(METRICS_PATH.read_text())
            model_name = summary["best_model"]
        else:
            summary = run_evaluation(cv=cv)
            model_name = summary["best_model"]

    print(f"Training calibrated '{model_name}' on {len(df)} samples...")
    base = candidate_models()[model_name]
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=cv)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(calibrated, X, y, cv=skf, method="predict_proba")
    classes = sorted(y.unique().tolist())
    predicted = np.asarray(classes)[np.argmax(probs, axis=1)]

    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "brier": _multiclass_brier(y, probs, classes),
        "log_loss": float(log_loss(y, probs, labels=classes)),
    }
    print(f"Calibrated CV metrics: {metrics}")

    calibrated.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    _reliability_curve(y, probs, classes, MODELS_DIR / "artifacts" / "reliability_curve.png")

    metadata = {
        "version": MODEL_VERSION,
        "model_type": f"{model_name}+sigmoid_calibrated",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "sample_rate": SAMPLE_RATE,
        "n_features": int(X.shape[1]),
        "features": list(X.columns),
        "classes": classes,
        "metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    train(model_name=name)
