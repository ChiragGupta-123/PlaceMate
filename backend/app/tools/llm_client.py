# filepath: backend/app/tools/llm_client.py
"""
PlaceMate — LLM Interface (Provider-Agnostic)
=============================================

Defines the ``LLMInterface`` protocol, typed request/response models and
the error hierarchy.  This module is the ONLY contract that the
``QuestionService`` (Step 4.3) and ``FakeLLM`` depend on.

IMPORT RULES
────────────
This module imports ONLY stdlib (and, if it ever needs them, ``app.core``
/ ``pydantic``).  It must never import an LLM SDK, ``httpx`` or anything
from ``app.graph``.

SEMANTICS
─────────
``generate_structured()`` returns the model's raw structured text plus a
best-effort parsed ``data`` dict.  Schema validation is the CALLER's job
(Pydantic, in Step 4.2/4.3): a syntactically valid but schema-invalid
answer is a normal ``LLMResponse`` with ``data`` set, while unparseable
text is a normal ``LLMResponse`` with ``data=None`` — never an exception.

``LLMBadOutput`` is reserved for adapter-level unusable payloads (empty
content, provider refusal envelope), not for schema mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


# ── Request / Response types ────────────────────────────────────────

@dataclass(frozen=True)
class LLMRequest:
    """What we send to the LLM.

    ``tags`` is for logging/tracing only.  It must never contain
    credentials or candidate secrets.
    """
    system_prompt: str
    user_prompt: str
    response_schema: Mapping[str, Any] | None = None  # JSON schema, optional
    temperature: float = 0.4
    max_output_tokens: int = 1024
    timeout_s: float = 30.0
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """What the LLM gives back.

    ``text`` is the raw model output.  ``data`` is the parsed JSON object
    when the text is a JSON object, otherwise ``None``.
    """
    text: str
    data: dict[str, Any] | None = None
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    requests_made: int = 0


# ── Error hierarchy ─────────────────────────────────────────────────

class LLMError(Exception):
    """Base for all LLM errors.

    Contract: ``LLMInterface.generate_structured()`` raises ONLY
    ``LLMError`` subclasses and never leaks credentials or prompt text.
    """


class LLMUnavailable(LLMError):
    """The LLM API is unreachable (connection refused, DNS failure, etc.)."""


class LLMTimeout(LLMError):
    """Our request deadline expired."""


class LLMQuotaExceeded(LLMError):
    """The provider's rate/quota limit was hit."""


class LLMAuthError(LLMError):
    """401/403 from the provider — bad or missing API key."""


class LLMBadOutput(LLMError):
    """The provider returned an unusable payload (empty/refusal envelope)."""


class LLMConfigurationError(LLMError):
    """The configured provider cannot be built (missing/unknown adapter)."""


# ── Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class LLMInterface(Protocol):
    """Provider-agnostic LLM contract.

    Implementors:
        • FakeLLM  (app/tools/fake_llm.py)  — offline testing
        • real adapters (Step 4.4, app/tools/llm_<provider>.py)

    Contract:
        • ``generate_structured()`` returns an ``LLMResponse`` or raises
          only ``LLMError`` subclasses.
        • Credentials must never appear in exceptions, logs or responses.
    """

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        ...
