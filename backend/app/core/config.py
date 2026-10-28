# filepath: backend/app/core/config.py
"""
PlaceMate — Application Settings
==================================

Uses pydantic-settings to load configuration from environment variables
and ``.env`` files.  ``get_settings()`` returns a cached singleton.

SECURITY
────────
• ``judge0_rapidapi_key`` is held as ``SecretStr``.  Its value is
  accessed only inside ``Judge0Sandbox`` at HTTP-send time.
• A validator fails fast if ``sandbox_provider == "judge0"`` and the
  key is missing, with a clear, secret-free message.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All PlaceMate configuration, loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Sandbox provider ────────────────────────────────────────────
    sandbox_provider: str = "fake"  # "fake" or "judge0"

    # ── Judge0 / RapidAPI ───────────────────────────────────────────
    judge0_base_url: str = "https://judge0-ce.p.rapidapi.com"
    judge0_rapidapi_host: str = "judge0-ce.p.rapidapi.com"
    judge0_rapidapi_key: SecretStr | None = None
    judge0_python_language_id: int = 71  # Python 3.8.1 on Judge0 CE

    # ── HTTP timeouts ───────────────────────────────────────────────
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 15.0

    # ── Polling ─────────────────────────────────────────────────────
    poll_interval_initial_s: float = 1.0
    poll_interval_max_s: float = 3.0
    poll_backoff: float = 1.5
    poll_max_attempts: int = 12
    poll_deadline_s: float = 25.0

    # ── Retries and mode ────────────────────────────────────────────
    create_max_retries: int = 2
    judge0_use_wait: bool = False

    # ── Execution limits ────────────────────────────────────────────
    max_cpu_time_s: float = 15.0
    max_wall_time_s: float = 30.0
    default_memory_kb: int = 256_000
    max_source_bytes: int = 20_000

    @model_validator(mode="after")
    def _check_judge0_key(self) -> Settings:
        if self.sandbox_provider == "judge0" and not self.judge0_rapidapi_key:
            raise ValueError(
                "JUDGE0_RAPIDAPI_KEY must be set when SANDBOX_PROVIDER=judge0. "
                "Get a free key at https://rapidapi.com/judge0-official/api/judge0-ce"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
