# filepath: backend/app/tools/fake_sandbox.py
"""
PlaceMate — Fake Sandbox (Offline Testing)
============================================

``FakeSandbox`` implements ``SandboxInterface`` for offline development
and testing.  Two modes:

1. **Scripted** — a queue of pre-defined scenarios.
2. **Heuristic** (default) — reproduces Phase 1 mock behavior by
   inspecting ``PM_SOURCE_B64`` from the harness header.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections import deque
from typing import Any, Deque, Dict, List, Sequence

from app.tools.code_sandbox import (
    SandboxAuthError,
    SandboxOutcome,
    SandboxQuotaExceeded,
    SandboxRequest,
    SandboxResponse,
    SandboxTimeout,
    SandboxUnavailable,
)

logger = logging.getLogger(__name__)

# Scenarios that raise exceptions instead of returning a response
_EXCEPTION_SCENARIOS = {
    "quota": SandboxQuotaExceeded,
    "auth": SandboxAuthError,
    "unavailable": SandboxUnavailable,
    "timeout": SandboxTimeout,
}


class FakeSandbox:
    """Offline sandbox for testing and demo scripts.

    Parameters
    ----------
    scenarios : sequence of scenario names, consumed in order.
        Supported: "pass", "wrong_answer", "compile_error",
        "runtime_error", "timeout_partial", "slow", "internal_error",
        "quota", "auth", "unavailable", "timeout".
        When the queue is empty, falls back to heuristic mode.
    """

    def __init__(self, scenarios: Sequence[str] | None = None) -> None:
        self._queue: Deque[str] = deque(scenarios or [])

    def run(self, request: SandboxRequest) -> SandboxResponse:
        """Execute against the fake sandbox."""
        nonce, layout = self._extract_header(request.source_code)

        if self._queue:
            scenario = self._queue.popleft()
            return self._scripted(scenario, nonce, layout)

        return self._heuristic(request.source_code, nonce, layout)

    # ── Header extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_header(source: str) -> tuple[str, List[Dict[str, Any]]]:
        """Extract PM_NONCE and PM_LAYOUT from the harness source."""
        nonce = ""
        layout: List[Dict[str, Any]] = []
        for line in source.split("\n"):
            m = re.match(r"^PM_NONCE\s*=\s*['\"](.+?)['\"]", line)
            if m:
                nonce = m.group(1)
            m = re.match(r"^PM_LAYOUT\s*=\s*'(.+?)'", line)
            if m:
                try:
                    layout = json.loads(m.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
        return nonce, layout

    @staticmethod
    def _extract_source_b64(source: str) -> str:
        """Decode the candidate's source from PM_SOURCE_B64."""
        for line in source.split("\n"):
            m = re.match(r"^PM_SOURCE_B64\s*=\s*['\"](.+?)['\"]", line)
            if m:
                try:
                    return base64.b64decode(m.group(1)).decode("utf-8", errors="replace")
                except Exception:
                    return ""
        return ""

    # ── Scripted mode ───────────────────────────────────────────────

    def _scripted(
        self,
        scenario: str,
        nonce: str,
        layout: List[Dict[str, Any]],
    ) -> SandboxResponse:
        """Generate a response for a named scenario."""
        # Scenarios that raise
        if scenario in _EXCEPTION_SCENARIOS:
            raise _EXCEPTION_SCENARIOS[scenario](f"Fake {scenario}")

        marker = f"@@PM:{nonce}@@"

        if scenario == "internal_error":
            return SandboxResponse(
                outcome=SandboxOutcome.internal_error,
                provider="fake",
                requests_made=1,
            )

        if scenario == "compile_error":
            stdout = marker + json.dumps({
                "type": "compile_error",
                "message": "SyntaxError: invalid syntax",
                "line": 1,
            }) + "\n"
            return SandboxResponse(
                outcome=SandboxOutcome.ok,
                stdout=stdout,
                provider="fake",
                requests_made=1,
            )

        if scenario == "runtime_error":
            lines = []
            for i, tc in enumerate(layout):
                lines.append(marker + json.dumps({
                    "type": "test", "index": i, "kind": tc["kind"],
                    "passed": False, "outcome": "exception",
                    "error": "NameError: name 'x' is not defined",
                    "time_ms": 1.0,
                }) + "\n")
            lines.append(marker + json.dumps({"type": "done"}) + "\n")
            return SandboxResponse(
                outcome=SandboxOutcome.ok,
                stdout="".join(lines),
                provider="fake",
                requests_made=1,
            )

        if scenario == "timeout_partial":
            # Only first test completes, rest are killed
            lines = []
            if layout:
                lines.append(marker + json.dumps({
                    "type": "test", "index": 0, "kind": layout[0]["kind"],
                    "passed": False, "outcome": "timeout",
                    "time_ms": 5000.0,
                }) + "\n")
            return SandboxResponse(
                outcome=SandboxOutcome.timeout,
                stdout="".join(lines),
                provider="fake",
                requests_made=1,
            )

        if scenario == "wrong_answer":
            lines = []
            for i, tc in enumerate(layout):
                lines.append(marker + json.dumps({
                    "type": "test", "index": i, "kind": tc["kind"],
                    "passed": False, "outcome": "wrong_answer",
                    "time_ms": 5.0,
                    **({"input_repr": "[test]", "expected_repr": "[expected]",
                        "actual_repr": "[wrong]"} if tc["kind"] == "public" else {}),
                }) + "\n")
            lines.append(marker + json.dumps({"type": "done"}) + "\n")
            return SandboxResponse(
                outcome=SandboxOutcome.ok,
                stdout="".join(lines),
                provider="fake",
                requests_made=1,
            )

        if scenario == "slow":
            # All correctness tests pass, perf test times out
            lines = []
            for i, tc in enumerate(layout):
                if tc["kind"] == "perf":
                    lines.append(marker + json.dumps({
                        "type": "test", "index": i, "kind": "perf",
                        "passed": False, "outcome": "timeout",
                        "time_ms": 10000.0,
                    }) + "\n")
                else:
                    lines.append(marker + json.dumps({
                        "type": "test", "index": i, "kind": tc["kind"],
                        "passed": True, "outcome": "pass",
                        "time_ms": 5.0,
                    }) + "\n")
            lines.append(marker + json.dumps({"type": "done"}) + "\n")
            return SandboxResponse(
                outcome=SandboxOutcome.ok,
                stdout="".join(lines),
                provider="fake",
                requests_made=1,
            )

        # Default: "pass" — all tests pass including perf
        lines = []
        for i, tc in enumerate(layout):
            lines.append(marker + json.dumps({
                "type": "test", "index": i, "kind": tc["kind"],
                "passed": True, "outcome": "pass",
                "time_ms": 5.0,
            }) + "\n")
        lines.append(marker + json.dumps({"type": "done"}) + "\n")
        return SandboxResponse(
            outcome=SandboxOutcome.ok,
            stdout="".join(lines),
            provider="fake",
            requests_made=1,
        )

    # ── Heuristic mode (Phase 1 backward compat) ────────────────────

    def _heuristic(
        self,
        harness_source: str,
        nonce: str,
        layout: List[Dict[str, Any]],
    ) -> SandboxResponse:
        """Reproduce old Phase 1 mock_execution behavior."""
        candidate = self._extract_source_b64(harness_source)

        if not candidate.strip() or "def " not in candidate:
            return self._scripted("compile_error", nonce, layout)

        # Two or more for-loops → correct but slow
        if candidate.count("for ") >= 2:
            return self._scripted("slow", nonce, layout)

        # Default: all pass
        return self._scripted("pass", nonce, layout)
