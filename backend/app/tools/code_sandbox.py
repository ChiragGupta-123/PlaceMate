# filepath: backend/app/tools/code_sandbox.py
"""
PlaceMate — Sandbox Interface (Provider-Agnostic)
==================================================

Defines the ``SandboxInterface`` protocol, typed request/response models
and the error hierarchy.  This module is the ONLY contract that
``ExecutionService`` and ``FakeSandbox`` depend on.

IMPORT RULES
────────────
This module imports ONLY from app.schemas, app.core and the stdlib.
It must never import httpx, Judge0 specifics or anything from app.graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


# ── Request / Response types ────────────────────────────────────────

@dataclass(frozen=True)
class SandboxLimits:
    """Resource limits sent to the sandbox."""
    cpu_time_s: float = 10.0
    wall_time_s: float = 20.0
    memory_kb: int = 256_000


@dataclass(frozen=True)
class SandboxRequest:
    """What we send to the sandbox."""
    source_code: str
    language: str = "python"
    stdin: str = ""
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    tags: dict[str, str] = field(default_factory=dict)  # for logging only


class SandboxOutcome(str, Enum):
    """Coarse outcome of a sandbox run."""
    ok = "ok"
    compile_error = "compile_error"
    runtime_error = "runtime_error"
    timeout = "timeout"
    killed_by_signal = "killed_by_signal"
    internal_error = "internal_error"


@dataclass
class SandboxResponse:
    """What the sandbox gives back."""
    outcome: SandboxOutcome
    stdout: str = ""
    stderr: str = ""
    compile_output: str = ""
    message: str = ""
    exit_code: int | None = None
    signal: int | None = None
    cpu_time_ms: float = 0.0
    wall_time_ms: float = 0.0
    memory_kb: float = 0.0
    raw_status: str = ""
    provider: str = ""
    requests_made: int = 0
    latency_ms: float = 0.0


# ── Error hierarchy ─────────────────────────────────────────────────

class SandboxError(Exception):
    """Base for all sandbox errors.

    Contract: SandboxInterface.run() raises ONLY SandboxError subclasses
    and never leaks credentials.
    """


class SandboxUnavailable(SandboxError):
    """The sandbox API is unreachable (connection refused, DNS failure, etc.)."""


class SandboxTimeout(SandboxError):
    """Our poll deadline expired (not the candidate's TLE)."""


class SandboxQuotaExceeded(SandboxError):
    """The provider's rate/quota limit was hit."""


class SandboxAuthError(SandboxError):
    """401/403 from the provider — bad or missing API key."""


class SandboxBadRequest(SandboxError):
    """422 or other 4xx — our request was rejected."""


class SandboxBadResponse(SandboxError):
    """Malformed JSON, missing token/status in the response."""


class UnsupportedLanguage(SandboxError):
    """The requested language is not supported."""


# ── Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class SandboxInterface(Protocol):
    """Provider-agnostic sandbox contract.

    Implementors:
        • Judge0Sandbox  (app/tools/judge0.py)   — real HTTP calls
        • FakeSandbox    (app/tools/fake_sandbox.py)  — offline testing

    Contract:
        • ``run()`` returns a ``SandboxResponse`` or raises only
          ``SandboxError`` subclasses.
        • Credentials must never appear in exceptions, logs or responses.
    """

    def run(self, request: SandboxRequest) -> SandboxResponse:
        ...
