# AGENTS.md

Music-genre classifier: extracts librosa features and predicts genre with a scikit-learn classifier. `src/` holds flat, top-level modules (no package); `eval/` is an experiment/production-training package; `app/` is a FastAPI service + dashboard; `tests/` is the pytest suite.

## Commands

Always run from the **repo root** with the checked-in virtualenv (`./venv/bin/python`, Python 3.14). Do not `cd src`.

- Install deps: `./venv/bin/pip install -r requirements.txt -r requirements-dev.txt`
- CLI predict: `./venv/bin/python src/main.py` (prompts for a file path)
- API + dashboard: `./venv/bin/uvicorn app.main:app --reload --port 8000` (dashboard `/`, docs `/docs`, metrics `/metrics`)
- Tests: `./venv/bin/python -m pytest` (no dataset/model needed — synthetic fixtures)
- Lint / format / types: `./venv/bin/ruff check .`, `./venv/bin/ruff format .`, `./venv/bin/mypy src app`
- Build/refresh feature cache (slow, ~6 min parallel): `./venv/bin/python -m eval.features`
- Compare models (needs cache): `./venv/bin/python -m eval.evaluate`
- Train + calibrate production model (needs cache): `./venv/bin/python -m eval.train`
- API + Redis locally: `docker compose up --build`

`eval` modules must run as `python -m eval.<module>` (they import `eval.*` and `src` as top-level names via a `sys.path` bootstrap), and `ruff format` is enforced by CI, so format before committing.

## Architecture

- `src/config.py` — ML constants (`SAMPLE_RATE` 22050, chunking, paths, `RANDOM_STATE`) and `resolve_model_path()` (env `MODEL_PATH` → `models/model.pkl` → legacy `model.pkl`).
- `src/settings.py` — service settings from `.env` (upload limits, allowed extensions, `REDIS_URL`, logging).
- `src/predictor.py` — the only inference path: `predict()` returns `{genre: probability}`; `predict_detailed()` also returns per-chunk predictions. Averages `predict_proba` over chunks, caches the model via `lru_cache`.
- `src/cache.py` — prediction cache keyed by audio SHA-256 + model path; Redis when reachable, else an in-process LRU.
- `src/feature_extraction.py` — the 90-feature dict; **column order must stay stable** (training and inference build a one-row DataFrame from it).
- `app/main.py` — FastAPI routes under `/api/v1` (`/health`, `/model/info`, `/metrics/summary`, `/predict`) plus unversioned aliases and `/metrics` (Prometheus). Blocking work is offloaded with `run_in_threadpool`. `app/observability.py` adds JSON logs, request IDs, and metrics; `app/schemas.py` holds response models.
- `eval/features.py` caches features to `models/features.csv`; `eval/evaluate.py` writes `models/metrics.json` + `models/artifacts/`; `eval/train.py` writes `models/model.pkl` + `models/metadata.json`.

## Data and model

- Dataset is GTZAN under `dataset/genres_original/<genre>/*.wav`. The **subdirectory name is the training label**: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock` (note `hiphop`, not `hip-hop`). `jazz.00054.wav` is corrupt and always skipped.
- Audio is loaded at `SAMPLE_RATE` (22050) in training and inference.
- `models/*.pkl`, `models/features.csv`, and `models/artifacts/` are gitignored; `models/metadata.json` and `models/metrics.json` are committed.

## Gotchas

- Re-run `python -m eval.features` whenever `feature_extraction.py` changes, or the cached feature matrix goes stale.
- Inference loads each file once with `librosa.load(..., sr=SAMPLE_RATE)` and slices in memory to avoid `psf_fseek` errors on MP3s. Keep this.
- Tests point `MODEL_PATH` at a tiny fixture model and call `predictor.load_model.cache_clear()`; don't assume the real model exists during tests.
- `feature_extraction.py` returns `None` on failure; data builders silently skip unreadable files.
- Chunks shorter than 2048 samples are skipped because chroma extraction requires it.
- `test/sample/blackSabbathParanoid.mp3` is a manual sample; automated coverage lives in `tests/`.
- `venv/`, `dataset/`, and `model.pkl` are gitignored.
