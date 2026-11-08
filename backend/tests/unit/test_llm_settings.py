# filepath: backend/tests/unit/test_llm_settings.py
"""
LLM settings and factory tests (Step 4.1).

Offline: no SDK, no network, no real key.  The key below is a sentinel
used only to prove it never appears in messages or reprs.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.tools import get_llm_provider
from app.tools.fake_llm import FakeLLM
from app.tools.llm_client import LLMConfigurationError, LLMError, LLMInterface

FAKE_KEY = "sk-sentinel-not-a-real-key"


# ── Defaults ─────────────────────────────────────────────────────────

def test_llm_defaults_are_offline_and_safe() -> None:
    settings = Settings()

    assert settings.llm_provider == "fake"
    assert settings.llm_model
    assert settings.llm_connect_timeout_s > 0
    assert settings.llm_read_timeout_s > 0
    assert settings.llm_max_output_tokens > 0
    assert settings.llm_api_key is None


# ── SecretStr handling ───────────────────────────────────────────────

def test_api_key_is_a_secret_and_hidden_from_repr() -> None:
    settings = Settings(llm_api_key=SecretStr(FAKE_KEY))

    assert isinstance(settings.llm_api_key, SecretStr)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == FAKE_KEY
    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)


# ── Fail-fast validator ──────────────────────────────────────────────

def test_non_fake_provider_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(llm_provider="openai", llm_api_key=None)

    assert "LLM_API_KEY" in str(excinfo.value)


def test_non_fake_provider_with_key_constructs() -> None:
    settings = Settings(llm_provider="openai", llm_api_key=SecretStr(FAKE_KEY))

    assert settings.llm_provider == "openai"


# ── Factory behaviour (as implemented today) ─────────────────────────

def test_factory_returns_fake_llm_for_fake_provider() -> None:
    llm = get_llm_provider(Settings(llm_provider="fake"))

    assert isinstance(llm, FakeLLM)
    assert isinstance(llm, LLMInterface)


def test_factory_raises_configuration_error_for_unimplemented_provider() -> None:
    settings = Settings(llm_provider="openai", llm_api_key=SecretStr(FAKE_KEY))

    with pytest.raises(LLMConfigurationError) as excinfo:
        get_llm_provider(settings)

    message = str(excinfo.value)
    assert "openai" in message
    assert "not implemented" in message
    assert FAKE_KEY not in message
    assert isinstance(excinfo.value, LLMError)


def test_factory_configuration_error_works_without_a_key() -> None:
    """Bypassing the validator must still produce the clear factory error."""
    settings = Settings.model_construct(llm_provider="openai", llm_api_key=None)

    with pytest.raises(LLMConfigurationError):
        get_llm_provider(settings)
