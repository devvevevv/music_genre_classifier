"""Cross-validated comparison of candidate models.

Produces ``models/metrics.json`` plus comparison and confusion-matrix plots.
Run from the repository root:

    python -m eval.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _path in (str(ROOT_DIR), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import METRICS_PATH, MODELS_DIR, RANDOM_STATE  # noqa: E402
from eval.features import load_features, split_xy  # noqa: E402

ARTIFACTS_DIR = MODELS_DIR / "artifacts"
SELECTION_METRIC = "macro_f1"


def candidate_models() -> dict:
    """Return the models compared in the experiment table."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def _plot_comparison(results: dict, path: Path) -> None:
    names = list(results)
    accuracy = [results[n]["accuracy"] for n in names]
    macro_f1 = [results[n]["macro_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy")
    ax.bar(x + width / 2, macro_f1, width, label="Macro F1")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (5-fold cross-validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_evaluation(cv: int = 5, features_path: Path | None = None) -> dict:
    """Run k-fold CV for every candidate and write metrics + plots."""
    df = load_features() if features_path is None else load_features(cache_path=features_path)
    X, y = split_xy(df)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for name, model in candidate_models().items():
        print(f"Evaluating {name}...")
        preds = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)
        results[name] = {
            "accuracy": float(accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro")),
            "report": classification_report(y, preds, output_dict=True),
        }
        if name != "dummy_most_frequent":
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                y, preds, normalize="true", xticks_rotation=45, ax=ax, colorbar=False
            )
            ax.set_title(f"Confusion matrix - {name}")
            fig.tight_layout()
            fig.savefig(ARTIFACTS_DIR / f"confusion_matrix_{name}.png", dpi=120)
            plt.close(fig)

    best_model = max(results, key=lambda n: results[n][SELECTION_METRIC])
    _plot_comparison(results, ARTIFACTS_DIR / "model_comparison.png")

    summary = {
        "dataset": {
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "classes": sorted(y.unique().tolist()),
        },
        "cv_folds": cv,
        "selection_metric": SELECTION_METRIC,
        "best_model": best_model,
        "results": results,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nBest model by {SELECTION_METRIC}: {best_model}")
    for name, metrics in results.items():
        print(f"  {name:24s} acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}")
    print(f"Wrote {METRICS_PATH}")
    return summary


if __name__ == "__main__":
    run_evaluation()
