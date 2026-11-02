"""Prediction cache keyed by audio content hash.

Uses Redis when ``REDIS_URL`` is set and reachable, otherwise an in-process
LRU. Both store JSON-encoded inference results, so callers get plain dicts.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from settings import get_settings


class PredictionCache:
    def __init__(self, max_entries: int = 256, redis_url: str = "") -> None:
        self._max = max_entries
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._redis = self._connect(redis_url)

    @staticmethod
    def _connect(redis_url: str):
        if not redis_url:
            return None
        try:
            import redis  # local import: optional dependency at runtime

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:  # noqa: BLE001 - degrade to in-process cache
            return None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def get(self, key: str) -> dict | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(f"mgc:{key}")
                return json.loads(raw) if raw else None
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        return None

    def set(self, key: str, value: dict) -> None:
        settings = get_settings()
        if self._redis is not None:
            try:
                self._redis.set(f"mgc:{key}", json.dumps(value), ex=settings.cache_ttl_seconds)
                return
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)


@lru_cache
def get_cache() -> PredictionCache:
    settings = get_settings()
    return PredictionCache(max_entries=settings.cache_max_entries, redis_url=settings.redis_url)
