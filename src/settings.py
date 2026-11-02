"""Service-level settings loaded from the environment / ``.env``."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Music Genre Classifier"
    log_level: str = "INFO"

    # Upload limits
    max_upload_mb: float = 25.0
    allowed_extensions: str = ".wav,.mp3,.flac,.ogg,.m4a"

    # Optional Redis cache; falls back to an in-process LRU when empty/unavailable.
    redis_url: str = ""
    cache_max_entries: int = 256
    cache_ttl_seconds: int = 86400

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def allowed_extension_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
