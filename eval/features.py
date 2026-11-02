"""Build and cache the engineered feature matrix for the whole dataset.

Feature extraction is the slow part of this project (~3 s/track), so it is
done once in parallel and cached to ``models/features.csv``. Every downstream
experiment reuses the cache.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for _path in (str(ROOT_DIR), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import DATASET_DIR, FEATURES_CACHE  # noqa: E402


def iter_audio_files(dataset_dir: Path = DATASET_DIR):
    """Yield ``(genre, path)`` for every audio file, genre = directory name."""
    for genre_dir in sorted(Path(dataset_dir).iterdir()):
        if not genre_dir.is_dir() or genre_dir.name.startswith("."):
            continue
        for audio in sorted(genre_dir.iterdir()):
            if audio.name.startswith(".") or not audio.is_file():
                continue
            yield genre_dir.name, audio


def _extract_one(job):
    """Worker entry point: load one file and return its feature dict."""
    genre, path = job
    import librosa  # imported in the worker so spawn pickles only the path

    from config import SAMPLE_RATE
    from feature_extraction import feature_extraction

    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        features = feature_extraction(y, sr)
    except Exception as exc:  # noqa: BLE001 - skip anything unreadable
        print(f"  skipping {path}: {exc}")
        return None

    if features is None:
        return None

    features["genre"] = genre
    features["file path"] = str(path)
    return features


def build_features(
    dataset_dir: Path = DATASET_DIR,
    cache_path: Path = FEATURES_CACHE,
    workers: int | None = None,
) -> pd.DataFrame:
    """Extract features for all tracks in parallel and write the cache."""
    jobs = list(iter_audio_files(dataset_dir))
    if not jobs:
        raise FileNotFoundError(f"No audio files found under {dataset_dir}")

    workers = workers or max(1, (os.cpu_count() or 1) - 2)
    print(f"Extracting features for {len(jobs)} files with {workers} workers...")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, features in enumerate(pool.map(_extract_one, jobs, chunksize=4), 1):
            if features is not None:
                rows.append(features)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} processed")

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Wrote {df.shape[0]} rows x {df.shape[1]} cols -> {cache_path}")
    return df


def load_features(
    cache_path: Path = FEATURES_CACHE,
    dataset_dir: Path = DATASET_DIR,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load the feature cache, building it first if necessary."""
    if cache_path.exists() and not rebuild:
        return pd.read_csv(cache_path)
    return build_features(dataset_dir=dataset_dir, cache_path=cache_path)


def split_xy(df: pd.DataFrame):
    """Split a feature frame into ``X`` (features) and ``y`` (genre labels)."""
    X = df.drop(columns=["genre", "file path"])
    y = df["genre"]
    return X, y


if __name__ == "__main__":
    build_features()
