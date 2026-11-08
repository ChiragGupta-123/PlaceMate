# filepath: backend/app/tools/fake_llm.py
"""
PlaceMate — Fake LLM (Offline Testing)
=======================================

``FakeLLM`` implements ``LLMInterface`` for offline development and
testing.  It never touches the network and never needs an API key.

Two scripting modes (mirroring ``FakeSandbox``):

1. **Scenarios** — a queue of named scenarios, consumed in order.
   Supported: ``"ok"``, ``"bad_json"``, ``"timeout"``, ``"quota"``,
   ``"unavailable"``, ``"auth"``, ``"bad_output"``.
2. **Scripted responses** — a queue of ``str`` / ``dict`` /
   ``LLMResponse`` values, returned in order.

When both queues are empty the fake answers ``"ok"``.

Every request is appended to ``self.calls`` so tests can assert on the
prompts sent and on call counts (Step 4.3 call-cap tests).
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Deque, Dict, List, Sequence

from app.tools.llm_client import (
    LLMAuthError,
    LLMBadOutput,
    LLMError,
    LLMQuotaExceeded,
    LLMRequest,
    LLMResponse,
    LLMTimeout,
    LLMUnavailable,
)

# Scenarios that raise exceptions instead of returning a response
_EXCEPTION_SCENARIOS: Dict[str, type[LLMError]] = {
    "quota": LLMQuotaExceeded,
    "auth": LLMAuthError,
    "unavailable": LLMUnavailable,
    "timeout": LLMTimeout,
    "bad_output": LLMBadOutput,
}

# The default "ok" payload.  Deliberately generic: Step 4.2 owns the real
# output schema, so this fake only proves the plumbing.
_DEFAULT_PAYLOAD: Dict[str, Any] = {
    "question": "Fake question",
    "question_type": "technical",
    "topic": "fake",
    "difficulty": "medium",
}

ScriptedItem = str | Dict[str, Any] | LLMResponse


class FakeLLM:
    """Offline LLM for testing and demo scripts.

    Parameters
    ----------
    scenarios : sequence of scenario names, consumed in order.
    responses : sequence of scripted responses, consumed before scenarios.
    """

    provider = "fake"
    model = "fake-llm"

    def __init__(
        self,
        scenarios: Sequence[str] | None = None,
        responses: Sequence[ScriptedItem] | None = None,
    ) -> None:
        self._scenarios: Deque[str] = deque(scenarios or [])
        self._responses: Deque[ScriptedItem] = deque(responses or [])
        self.calls: List[LLMRequest] = []

    def generate_structured(self, request: LLMRequest) -> LLMResponse:
        """Answer from the scripted queues, or with the default ``ok``."""
        self.calls.append(request)

        if self._responses:
            return self._build(self._responses.popleft())

        scenario = self._scenarios.popleft() if self._scenarios else "ok"
        return self._scripted(scenario)

    # ── Scripted responses ──────────────────────────────────────────

    def _build(self, item: ScriptedItem) -> LLMResponse:
        """Turn a scripted item into an ``LLMResponse``."""
        if isinstance(item, LLMResponse):
            return item

        if isinstance(item, dict):
            return LLMResponse(
                text=json.dumps(item, sort_keys=True),
                data=dict(item),
                model=self.model,
                provider=self.provider,
                requests_made=1,
            )

        data: dict[str, Any] | None = None
        try:
            parsed = json.loads(item)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            data = parsed

        return LLMResponse(
            text=item,
            data=data,
            model=self.model,
            provider=self.provider,
            requests_made=1,
        )

    # ── Named scenarios ─────────────────────────────────────────────

    def _scripted(self, scenario: str) -> LLMResponse:
        """Generate a response for a named scenario."""
        if scenario in _EXCEPTION_SCENARIOS:
            raise _EXCEPTION_SCENARIOS[scenario](f"Fake {scenario}")

        if scenario == "bad_json":
            return LLMResponse(
                text="{not json",
                data=None,
                model=self.model,
                provider=self.provider,
                requests_made=1,
            )

        # Default: "ok" (also the fallback for unknown scenario names)
        return LLMResponse(
            text=json.dumps(_DEFAULT_PAYLOAD, sort_keys=True),
            data=dict(_DEFAULT_PAYLOAD),
            model=self.model,
            provider=self.provider,
            requests_made=1,
        )
