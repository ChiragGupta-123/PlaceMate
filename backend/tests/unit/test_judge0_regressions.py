# filepath: backend/tests/unit/test_judge0_regressions.py
"""
Regression tests for Step 3.5 (B12 a-e): Judge0 poll error handling.

Fully offline: ``httpx.MockTransport`` plus injected ``client``, ``sleep``
and ``monotonic``.  No network calls and no real waiting.

Covered items
─────────────
a) 401/403 on a poll → SandboxAuthError immediately, never retried.
b) 2xx with X-RateLimit-Requests-Remaining=0 still returns its result.
c) Only network errors, HTTP 5xx and malformed JSON are transient.
d) No placeholder-less f-string in judge0.py.
e) The API key never reaches an exception message, exception chain or log.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.tools.code_sandbox import (
    SandboxAuthError,
    SandboxBadRequest,
    SandboxOutcome,
    SandboxQuotaExceeded,
    SandboxRequest,
    SandboxUnavailable,
)
from app.tools.judge0 import Judge0Sandbox

BASE_URL = "https://judge0.test.local"
API_KEY = "test-key"
JUDGE0_SOURCE = Path(__file__).resolve().parents[2] / "app" / "tools" / "judge0.py"

Handler = Callable[[httpx.Request], httpx.Response]
PollStep = Callable[[], httpx.Response] | Exception
MakeSandbox = Callable[..., Tuple[Judge0Sandbox, List[float]]]


# ── Helpers ──────────────────────────────────────────────────────────

def _settings(**overrides: object) -> Settings:
    """Judge0 settings with a fake key and tiny poll intervals."""
    return Settings(
        sandbox_provider="judge0",
        judge0_rapidapi_key=SecretStr(API_KEY),
        judge0_base_url=BASE_URL,
        poll_interval_initial_s=0.01,
        poll_interval_max_s=0.05,
        poll_deadline_s=25.0,
        poll_max_attempts=12,
        create_max_retries=2,
        **overrides,  # type: ignore[arg-type]
    )


def _request() -> SandboxRequest:
    """A submission that is only ever sent through MockTransport."""
    return SandboxRequest(
        source_code="def two_sum(a, b):\n    return [0, 1]\n",
        language="python",
    )


def _payload(status_id: int) -> dict[str, object]:
    return {
        "status": {"id": status_id},
        "stdout": None,
        "stderr": None,
        "compile_output": None,
        "message": None,
        "exit_code": 0,
        "exit_signal": 0,
        "time": "0.01",
        "wall_time": "0.02",
        "memory": 1024,
    }


def _result(status_id: int = 3, headers: dict[str, str] | None = None) -> PollStep:
    def make() -> httpx.Response:
        return httpx.Response(200, json=_payload(status_id), headers=headers)

    return make


def _status(code: int, headers: dict[str, str] | None = None) -> PollStep:
    def make() -> httpx.Response:
        return httpx.Response(code, json={"error": "provider error"}, headers=headers)

    return make


def _malformed_json() -> PollStep:
    def make() -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    return make


class _Script:
    """MockTransport handler: a fixed submit reply plus a poll script.

    The final script entry repeats once exhausted, and ``poll_count`` records
    how many polls actually reached the transport.
    """

    def __init__(self, polls: list[PollStep]) -> None:
        assert polls, "poll script must not be empty"
        self._polls = polls
        self.poll_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"token": "tok-1"})
        assert request.method == "GET"
        step = self._polls[min(self.poll_count, len(self._polls) - 1)]
        self.poll_count += 1
        if isinstance(step, Exception):
            raise step
        return step()


def _exception_text(exc: BaseException) -> str:
    """Everything a traceback could print: message, args, cause and context."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        parts.append(repr(current.args))
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


@pytest.fixture
def make_sandbox() -> Iterator[MakeSandbox]:
    """Factory building a Judge0Sandbox over MockTransport (no network/waiting)."""
    clients: list[httpx.Client] = []

    def _make(handler: Handler, **overrides: object) -> Tuple[Judge0Sandbox, List[float]]:
        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE_URL,
        )
        clients.append(client)
        sleeps: list[float] = []
        sandbox = Judge0Sandbox(
            settings=_settings(**overrides),
            client=client,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )
        return sandbox, sleeps

    yield _make
    for client in clients:
        client.close()


# ── a) 401/403 on poll → SandboxAuthError, no retry ──────────────────

@pytest.mark.parametrize("status_code", [401, 403])
def test_a_poll_auth_error_raises_immediately_without_retry(
    make_sandbox: MakeSandbox, status_code: int,
) -> None:
    """B12a: an auth failure during polling is not transient."""
    script = _Script([_status(status_code)])
    sandbox, _sleeps = make_sandbox(script)

    with pytest.raises(SandboxAuthError):
        sandbox.run(_request())

    assert script.poll_count == 1, (
        f"auth failure (HTTP {status_code}) was retried "
        f"{script.poll_count} times (B12a bug)"
    )


# ── b) 2xx + remaining == 0 → still return the result ────────────────

def test_b_zero_remaining_on_successful_poll_returns_result(
    make_sandbox: MakeSandbox, caplog: pytest.LogCaptureFixture,
) -> None:
    """B12b: a paid-for 2xx result must not be discarded for remaining=0."""
    script = _Script([
        _result(3, headers={"X-RateLimit-Requests-Remaining": "0"}),
    ])
    sandbox, _sleeps = make_sandbox(script)

    with caplog.at_level(logging.WARNING, logger="app.tools.judge0"):
        response = sandbox.run(_request())

    assert response.outcome is SandboxOutcome.ok, (
        f"expected ok but got {response.outcome} (B12b bug)"
    )
    assert "quota_exhausted" in caplog.text, (
        "remaining=0 should be logged as a warning, not dropped silently"
    )


# ── c) only network / 5xx / malformed JSON are transient ─────────────

def test_c_poll_4xx_other_than_429_raises_bad_request(
    make_sandbox: MakeSandbox,
) -> None:
    """B12c: 4xx other than 429 propagates immediately as SandboxBadRequest."""
    script = _Script([_status(400)])
    sandbox, _sleeps = make_sandbox(script)

    with pytest.raises(SandboxBadRequest):
        sandbox.run(_request())

    assert script.poll_count == 1, (
        f"HTTP 400 was retried {script.poll_count} times (B12c bug)"
    )


def test_c_poll_429_raises_quota_exceeded(make_sandbox: MakeSandbox) -> None:
    """B12c: 429 propagates as SandboxQuotaExceeded, never retried."""
    script = _Script([_status(429)])
    sandbox, _sleeps = make_sandbox(script)

    with pytest.raises(SandboxQuotaExceeded):
        sandbox.run(_request())

    assert script.poll_count == 1


def test_c_malformed_json_is_transient_then_unavailable(
    make_sandbox: MakeSandbox,
) -> None:
    """B12c: malformed JSON stays transient → 3 strikes → SandboxUnavailable."""
    script = _Script([_malformed_json()])
    sandbox, _sleeps = make_sandbox(script)

    with pytest.raises(SandboxUnavailable):
        sandbox.run(_request())

    assert script.poll_count == 3


# ── d) no placeholder-less f-string ──────────────────────────────────

def test_d_no_fstring_without_placeholders_in_judge0() -> None:
    """B12d: ``f"3 consecutive poll failures"`` had no placeholders."""
    tree = ast.parse(JUDGE0_SOURCE.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and not any(isinstance(v, ast.FormattedValue) for v in node.values)
    ]
    assert not offenders, (
        f"placeholder-less f-string(s) at line(s) {offenders} (B12d bug)"
    )


# ── e) the API key never leaks ───────────────────────────────────────

def test_e_api_key_never_appears_in_exceptions_or_logs(
    make_sandbox: MakeSandbox, caplog: pytest.LogCaptureFixture,
) -> None:
    """B12e: not in any exception message, exception chain or log line."""
    seen_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_keys.append(request.headers.get("X-RapidAPI-Key", ""))
        if request.method == "POST":
            return httpx.Response(200, json={"token": "tok-1"})
        # A hostile low-level message that embeds the credential.
        raise httpx.ConnectError(f"cannot connect with key {API_KEY}")

    sandbox, _sleeps = make_sandbox(handler)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SandboxUnavailable) as excinfo:
            sandbox.run(_request())

    # Sanity: the key really was sent in a header (so it could leak).
    assert seen_keys and all(value == API_KEY for value in seen_keys)

    assert API_KEY not in _exception_text(excinfo.value), (
        "API key leaked into the exception chain (B12e bug)"
    )
    assert API_KEY not in caplog.text, "API key leaked into the logs (B12e bug)"


# ── Required extra cases ─────────────────────────────────────────────

def test_poll_5xx_three_times_raises_unavailable(make_sandbox: MakeSandbox) -> None:
    """Three consecutive 5xx polls exhaust the transient budget."""
    script = _Script([_status(503)])
    sandbox, _sleeps = make_sandbox(script)

    with pytest.raises(SandboxUnavailable):
        sandbox.run(_request())

    assert script.poll_count == 3


def test_poll_recovers_after_one_5xx(make_sandbox: MakeSandbox) -> None:
    """A single transient 5xx resets on the next successful poll."""
    script = _Script([_status(500), _result(3)])
    sandbox, _sleeps = make_sandbox(script)

    response = sandbox.run(_request())

    assert response.outcome is SandboxOutcome.ok
    assert script.poll_count == 2


def test_pending_statuses_keep_polling_until_final(make_sandbox: MakeSandbox) -> None:
    """status_id 1 (in queue) and 2 (processing) are not final."""
    script = _Script([_result(1), _result(2), _result(3)])
    sandbox, _sleeps = make_sandbox(script)

    response = sandbox.run(_request())

    assert response.outcome is SandboxOutcome.ok
    assert script.poll_count == 3
