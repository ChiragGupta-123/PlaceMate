# filepath: backend/app/tools/__init__.py
"""
PlaceMate — Tools Package
===========================

Provides ``get_sandbox_provider(settings)`` which returns the appropriate
``SandboxInterface`` implementation based on configuration.

``judge0.py`` is imported **lazily** so that ``httpx`` is only loaded
when ``sandbox_provider == "judge0"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.tools.code_sandbox import SandboxInterface

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
