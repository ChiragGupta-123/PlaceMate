# filepath: backend/tests/unit/test_sandbox_contract.py
"""
Provider-agnostic contract tests for ``SandboxInterface`` (Step 3.6a).

Every provider must:
  • satisfy the ``SandboxInterface`` protocol,
  • return a ``SandboxResponse``,
  • raise only ``SandboxError`` subclasses,
  • never leak the API key in an exception.

Also covers the ``app.tools.get_sandbox_provider`` factory exactly as it
behaves today: ``"judge0"`` → lazy ``Judge0Sandbox`` (key required by
``Settings``, but the sandbox itself tolerates a missing key), everything
else → ``FakeSandbox``.

Fully offline: MockTransport for Judge0, no network, no real waiting.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Tuple

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.tools import get_sandbox_provider
from app.tools.code_sandbox import (
    SandboxError,
    SandboxInterface,
    SandboxOutcome,
    SandboxRequest,
    SandboxResponse,
)
from app.tools.fake_sandbox import FakeSandbox
from app.tools.judge0 import Judge0Sandbox

BASE_URL = "https://judge0.test.local"
API_KEY = "test-key"

Handler = Callable[[httpx.Request], httpx.Response]
MakeJudge0 = Callable[..., Tuple[Judge0Sandbox, List[float]]]


# ── Helpers ──────────────────────────────────────────────────────────

def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "sandbox_provider": "judge0",
        "judge0_rapidapi_key": SecretStr(API_KEY),
        "judge0_base_url": BASE_URL,
        "poll_interval_initial_s": 0.01,
        "poll_interval_max_s": 0.05,
        "create_max_retries": 2,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _request() -> SandboxRequest:
    """A submission that is only ever sent through MockTransport."""
    return SandboxRequest(
        source_code="def two_sum(a, b):\n    return [0, 1]\n",
        language="python",
    )


def _payload(status_id: int = 3) -> dict[str, object]:
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


def _happy_handler(request: httpx.Request) -> httpx.Response:
    """Submit → token, poll → final status 3."""
    if request.method == "POST":
        return httpx.Response(200, json={"token": "tok-1"})
    return httpx.Response(200, json=_payload(3))


def _status_handler(status_code: int, detail: str = "provider error") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": detail})

    return handler


def _malformed_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="not json at all")


def _connect_error_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused")


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


@pytest.fixture(params=["fake", "judge0"])
def provider(request: pytest.FixtureRequest) -> Iterator[Tuple[str, SandboxInterface]]:
    """Both real providers, one at a time."""
    if request.param == "fake":
        yield "fake", FakeSandbox()
        return

    client = httpx.Client(
        transport=httpx.MockTransport(_happy_handler),
        base_url=BASE_URL,
    )
    try:
        yield "judge0", Judge0Sandbox(
            settings=_settings(),
            client=client,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
        )
    finally:
        client.close()


@pytest.fixture
def make_judge0() -> Iterator[MakeJudge0]:
    """Factory: Judge0Sandbox over an arbitrary MockTransport handler."""
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


# ── Protocol + response shape ────────────────────────────────────────

def test_provider_satisfies_sandbox_interface(
    provider: Tuple[str, SandboxInterface],
) -> None:
    _name, sandbox = provider
    assert isinstance(sandbox, SandboxInterface)


def test_run_returns_sandbox_response(
    provider: Tuple[str, SandboxInterface],
) -> None:
    name, sandbox = provider

    response = sandbox.run(_request())

    assert isinstance(response, SandboxResponse)
    assert isinstance(response.outcome, SandboxOutcome)
    assert response.provider == name
    assert response.requests_made >= 1


# ── "Raises only SandboxError subclasses" ────────────────────────────

@pytest.mark.parametrize("scenario", ["quota", "auth", "unavailable", "timeout"])
def test_fake_sandbox_failures_are_sandbox_errors(scenario: str) -> None:
    with pytest.raises(SandboxError):
        FakeSandbox(scenarios=[scenario]).run(_request())


@pytest.mark.parametrize(
    "handler",
    [
        _status_handler(401, "unauthorized"),
        _status_handler(429, "rate limited"),
        _status_handler(500, "server error"),
        _malformed_handler,
        _connect_error_handler,
    ],
    ids=["401", "429", "500", "malformed-json", "connect-error"],
)
def test_judge0_failures_are_sandbox_errors(
    make_judge0: MakeJudge0, handler: Handler,
) -> None:
    sandbox, _sleeps = make_judge0(handler)

    with pytest.raises(SandboxError):
        sandbox.run(_request())


# ── "Never include the API key in an exception" ──────────────────────

@pytest.mark.parametrize(
    "handler",
    [
        _status_handler(401, "unauthorized"),
        _status_handler(429, "rate limited"),
        _status_handler(500, "server error"),
        _malformed_handler,
        _connect_error_handler,
    ],
    ids=["401", "429", "500", "malformed-json", "connect-error"],
)
def test_judge0_failures_never_contain_the_api_key(
    make_judge0: MakeJudge0, handler: Handler,
) -> None:
    sent: list[str] = []

    def recording(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("X-RapidAPI-Key", ""))
        return handler(request)

    sandbox, _sleeps = make_judge0(recording)

    with pytest.raises(SandboxError) as excinfo:
        sandbox.run(_request())

    assert sent, "the sandbox never sent a request"
    assert all(value == API_KEY for value in sent)
    assert API_KEY not in _exception_text(excinfo.value)


# ── Factory behaviour (as implemented today) ─────────────────────────

def test_factory_returns_fake_sandbox_for_fake_provider() -> None:
    sandbox = get_sandbox_provider(_settings(sandbox_provider="fake"))

    assert isinstance(sandbox, FakeSandbox)
    assert isinstance(sandbox, SandboxInterface)


def test_settings_reject_judge0_without_key() -> None:
    """Fail-fast validator: judge0 cannot be configured without a key."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(sandbox_provider="judge0", judge0_rapidapi_key=None)

    assert "JUDGE0_RAPIDAPI_KEY" in str(excinfo.value)


def test_factory_judge0_without_key_builds_lazily() -> None:
    """Bypassing the validator, the factory still builds a keyless sandbox.

    ``Judge0Sandbox`` reads the key only at send time, so construction must
    succeed and simply omit the ``X-RapidAPI-Key`` header.
    """
    settings = Settings.model_construct(
        sandbox_provider="judge0",
        judge0_rapidapi_key=None,
        judge0_base_url=BASE_URL,
    )

    sandbox = get_sandbox_provider(settings)
    try:
        assert isinstance(sandbox, Judge0Sandbox)
        assert isinstance(sandbox, SandboxInterface)
        assert "X-RapidAPI-Key" not in sandbox._headers()
        assert sandbox._headers()["X-RapidAPI-Host"] == settings.judge0_rapidapi_host
    finally:
        sandbox.close()
