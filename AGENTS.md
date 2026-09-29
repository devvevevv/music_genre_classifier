# AGENTS.md

Music-genre classifier: extracts librosa features and predicts genre with a scikit-learn Random Forest. Small flat Python project — modules live in `src/` with no package or `__init__.py`, and there is no test suite, linter, typechecker, or CI config.

## Commands

- Use the checked-in virtualenv, not system Python: `./venv/bin/python` (Python 3.14).
- Run from the **repo root**, e.g. `./venv/bin/python src/main.py` and `./venv/bin/python src/train.py`. Do not `cd src`.
  - `src/` modules import each other as top-level names (`from chunks import *`, `from process_data import *`); running `python src/<file>.py` from the root puts `src/` on `sys.path`.
  - `main.py` reads `model.pkl` and `train.py` writes it relative to the current directory, so the cwd must be the repo root or the model won't be found/saved where expected.
- Install deps: `./venv/bin/pip install -r requirements.txt`. The README lists `pickle`, but it's stdlib and not a real dependency.

## Data and model

- Dataset is GTZAN under `dataset/genres_original/<genre>/*.wav`. The **subdirectory name is the training label** and must be exactly: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock` (note `hiphop`, not `hip-hop`).
- `train.py` hardcodes the absolute dataset path `/Users/devvk/music_genre_classifier/dataset/genres_original` in its `__main__` block; update it if the repo moves.
- Training runs `GridSearchCV` (5-fold) over ~1000 tracks and is slow. `model.pkl` (~21 MB) already exists at the repo root, so retraining isn't needed for predictions.

## Gotchas

- `chunks.py` loads the full file once with `librosa.load(..., sr=None)` and slices it in memory specifically to avoid `psf_fseek` errors on MP3s. Keep this behavior; don't switch to default resampling/streaming.
- `feature_extraction.py` catches all exceptions and returns `None`; `process_data.py` silently skips unreadable/corrupt files. Silent skipping is expected, not a crash.
- Chunks shorter than 2048 samples are skipped because chroma extraction requires it.
- `test/sample/blackSabbathParanoid.mp3` is a manual sample only — there is no automated test command.
