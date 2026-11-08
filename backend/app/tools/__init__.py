# filepath: backend/app/tools/__init__.py
"""
PlaceMate — Tools Package
===========================

Provides two factories:

• ``get_sandbox_provider(settings)`` → ``SandboxInterface``
• ``get_llm_provider(settings)``     → ``LLMInterface``

Provider-specific modules are imported **lazily** so that optional SDKs
(``httpx`` for Judge0, an LLM SDK in Step 4.4) are only loaded when the
corresponding provider is actually configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.tools.code_sandbox import SandboxInterface
from app.tools.llm_client import LLMConfigurationError, LLMInterface

if TYPE_CHECKING:
    from app.core.config import Settings


def get_sandbox_provider(settings: Any) -> SandboxInterface:
    """Factory: return the sandbox provider based on settings.

    - ``"fake"`` → ``FakeSandbox`` (offline, no deps)
    - ``"judge0"`` → ``Judge0Sandbox`` (lazy import of httpx)
    """
    if settings.sandbox_provider == "judge0":
        # Lazy import so httpx is only loaded when needed
        from app.tools.judge0 import Judge0Sandbox
        return Judge0Sandbox(settings=settings)
    else:
        from app.tools.fake_sandbox import FakeSandbox
        return FakeSandbox()


def get_llm_provider(settings: Any) -> LLMInterface:
    """Factory: return the LLM provider based on settings.

    - ``"fake"`` → ``FakeLLM`` (offline, no deps)

    No real adapter exists yet (Step 4.4).  Any other provider raises a
    clear ``LLMConfigurationError`` naming the provider and nothing else.
    """
    if settings.llm_provider == "fake":
        from app.tools.fake_llm import FakeLLM
        return FakeLLM()

    raise LLMConfigurationError(
        f"LLM provider '{settings.llm_provider}' is not implemented yet. "
        "Only 'fake' is available; a real adapter lands in Step 4.4."
    )
