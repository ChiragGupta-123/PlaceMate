# filepath: backend/tests/unit/test_judge0_sandbox.py
"""
Offline tests for the Judge0Sandbox transport layer (Step 3.6a).

Covered here: the submit request shape, submit error handling, poll
deadlines, status-ID mapping and output decoding.

NOT repeated here (already covered by tests/unit/test_judge0_regressions.py):
poll 401/403, poll 4xx-other-than-429, poll 429, poll 5xx ×3, poll 5xx
recovery, poll malformed JSON, pending statuses, remaining=0 on 2xx,
the placeholder-less f-string and the poll key-leak test.

Fully offline: ``httpx.MockTransport`` with an injected client, ``sleep``
and ``monotonic``.  No network calls, no real waiting.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, Iterator, List

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.tools.code_sandbox import (
    SandboxAuthError,
    SandboxBadRequest,
    SandboxBadResponse,
    SandboxLimits,
    SandboxOutcome,
    SandboxQuotaExceeded,
    SandboxRequest,
    SandboxTimeout,
    SandboxUnavailable,
    UnsupportedLanguage,
)
from app.tools.judge0 import Judge0Sandbox

BASE_URL = "https://judge0.test.local"
API_KEY = "test-key"

MakeStep = Callable[[], httpx.Response]
MakeSandbox = Callable[..., "_Harness"]


# ── Helpers ──────────────────────────────────────────────────────────

def _settings(**overrides: object) -> Settings:
    """Judge0 settings with a fake key and tiny poll intervals."""
    values: dict[str, object] = {
        "sandbox_provider": "judge0",
        "judge0_rapidapi_key": SecretStr(API_KEY),
        "judge0_base_url": BASE_URL,
        "poll_interval_initial_s": 0.01,
        "poll_interval_max_s": 0.05,
        "poll_deadline_s": 25.0,
        "poll_max_attempts": 12,
        "create_max_retries": 2,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _request(
    source: str = "def two_sum(a, b):\n    return [0, 1]\n",
    stdin: str = "",
    language: str = "python",
    limits: SandboxLimits | None = None,
) -> SandboxRequest:
    """A submission that only ever travels through MockTransport."""
    return SandboxRequest(
        source_code=source,
        language=language,
        stdin=stdin,
        limits=limits or SandboxLimits(),
    )


def _b64(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _payload(status_id: int = 3, **overrides: Any) -> dict[str, Any]:
    """A Judge0 poll body with every field ``_build_response`` reads."""
    body: dict[str, Any] = {
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
    body.update(overrides)
    return body


def _json(
    code: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> MakeStep:
    def make() -> httpx.Response:
        return httpx.Response(code, json=body, headers=headers)

    return make


def _text(code: int, text: str) -> MakeStep:
    def make() -> httpx.Response:
        return httpx.Response(code, text=text)

    return make


def _submit_token() -> MakeStep:
    """200 with a token but no status → the poll path is used."""
    return _json(200, {"token": "tok-1"})


class _Replay:
    """Replays queued responses/exceptions; the last entry repeats forever."""

    def __init__(self, steps: List[MakeStep | Exception]) -> None:
        assert steps, "replay script must not be empty"
        self._steps = steps
        self.count = 0

    def __call__(self) -> httpx.Response:
        step = self._steps[min(self.count, len(self._steps) - 1)]
        self.count += 1
        if isinstance(step, Exception):
            raise step
        return step()


class _Transport:
    """MockTransport handler with separate submit (POST) and poll (GET) scripts."""

    def __init__(
        self,
        submit: "_Replay | None" = None,
        polls: "_Replay | None" = None,
    ) -> None:
        self._submit = submit or _Replay([_submit_token()])
        self._polls = polls or _Replay([_json(200, _payload(3))])
        self.post_count = 0
        self.get_count = 0
        self.last_post: httpx.Request | None = None
        self.last_get: httpx.Request | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.post_count += 1
            self.last_post = request
            return self._submit()
        assert request.method == "GET"
        self.get_count += 1
        self.last_get = request
        return self._polls()


class _Clock:
    """Fake monotonic clock: replays values, then repeats the last one."""

    def __init__(self, values: List[float]) -> None:
        assert values, "clock needs at least one value"
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


class _Harness:
    """A Judge0Sandbox over MockTransport plus its recorded fakes."""

    def __init__(
        self,
        sandbox: Judge0Sandbox,
        transport: _Transport,
        sleeps: List[float],
    ) -> None:
        self.sandbox = sandbox
        self.transport = transport
        self.sleeps = sleeps


@pytest.fixture
def make_sandbox() -> Iterator[MakeSandbox]:
    """Factory: Judge0Sandbox over MockTransport (no network, no waiting)."""
    clients: list[httpx.Client] = []

    def _make(
        transport: _Transport,
        clock: Callable[[], float] | None = None,
        **overrides: object,
    ) -> _Harness:
        client = httpx.Client(
            transport=httpx.MockTransport(transport),
            base_url=BASE_URL,
        )
        clients.append(client)
        sleeps: list[float] = []
        sandbox = Judge0Sandbox(
            settings=_settings(**overrides),
            client=client,
            sleep=sleeps.append,
            monotonic=clock or (lambda: 0.0),
        )
        return _Harness(sandbox, transport, sleeps)

    yield _make
    for client in clients:
        client.close()


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


# ── 1. Submit request shape ──────────────────────────────────────────

def test_submit_request_is_base64_encoded_with_request_limits(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport()
    harness = make_sandbox(transport, judge0_python_language_id=92)
    limits = SandboxLimits(cpu_time_s=2.5, wall_time_s=5.0, memory_kb=123_456)
    source = "def two_sum(a, b):\n    return [0, 1]\n"

    harness.sandbox.run(_request(source=source, limits=limits))

    post = transport.last_post
    assert post is not None
    assert post.url.path == "/submissions"
    assert post.url.params["base64_encoded"] == "true"
    assert post.url.params["wait"] == "false"

    body = _body(post)
    assert base64.b64decode(body["source_code"]).decode("utf-8") == source
    assert body["language_id"] == 92, "language_id must come from settings"
    assert body["cpu_time_limit"] == 2.5, "limits must come from request.limits"
    assert body["wall_time_limit"] == 5.0
    assert body["memory_limit"] == 123_456
    assert "stdin" not in body, "empty stdin must not be sent"


def test_submit_headers_carry_rapidapi_key_and_host(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport()
    harness = make_sandbox(transport)

    harness.sandbox.run(_request(stdin="2 7\n9\n"))

    post, poll = transport.last_post, transport.last_get
    assert post is not None and poll is not None
    for request in (post, poll):
        assert request.headers["X-RapidAPI-Key"] == API_KEY
        assert request.headers["X-RapidAPI-Host"] == "judge0-ce.p.rapidapi.com"


def test_submit_sends_base64_stdin_only_when_present(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport()
    harness = make_sandbox(transport)
    stdin = "2 7\n9\n"

    harness.sandbox.run(_request(stdin=stdin))

    assert transport.last_post is not None
    body = _body(transport.last_post)
    assert base64.b64decode(body["stdin"]).decode("utf-8") == stdin


def test_wait_true_final_status_returns_without_polling(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport(submit=_Replay([_json(200, _payload(3))]))
    harness = make_sandbox(transport, judge0_use_wait=True)

    response = harness.sandbox.run(_request())

    assert transport.last_post is not None
    assert transport.last_post.url.params["wait"] == "true"
    assert transport.get_count == 0, "a final wait=true response needs no poll"
    assert response.requests_made == 1
    assert response.outcome is SandboxOutcome.ok


def test_wait_true_pending_status_still_polls(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(
        submit=_Replay([_json(200, {"token": "tok-1", **_payload(1)})]),
        polls=_Replay([_json(200, _payload(3))]),
    )
    harness = make_sandbox(transport, judge0_use_wait=True)

    response = harness.sandbox.run(_request())

    assert transport.get_count == 1
    assert response.outcome is SandboxOutcome.ok


# ── 2. Submit errors ─────────────────────────────────────────────────

@pytest.mark.parametrize("status_code", [401, 403])
def test_submit_auth_error_is_not_retried(
    make_sandbox: MakeSandbox, status_code: int,
) -> None:
    transport = _Transport(submit=_Replay([_json(status_code, {"error": "nope"})]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxAuthError):
        harness.sandbox.run(_request())

    assert transport.post_count == 1
    assert harness.sleeps == [], "auth failures must not back off and retry"


def test_submit_422_raises_bad_request(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(submit=_Replay([_json(422, {"error": "limits"})]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxBadRequest) as excinfo:
        harness.sandbox.run(_request())

    assert "limits" in str(excinfo.value).lower()
    assert transport.post_count == 1


@pytest.mark.parametrize("status_code", [400, 409])
def test_submit_other_4xx_raises_bad_request(
    make_sandbox: MakeSandbox, status_code: int,
) -> None:
    transport = _Transport(submit=_Replay([_json(status_code, {"error": "bad"})]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxBadRequest) as excinfo:
        harness.sandbox.run(_request())

    assert str(status_code) in str(excinfo.value)
    assert transport.post_count == 1


def test_submit_429_raises_quota_exceeded(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(submit=_Replay([_json(429, {"error": "slow down"})]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxQuotaExceeded):
        harness.sandbox.run(_request())

    assert transport.post_count == 1


def test_submit_5xx_retried_with_injected_sleep_then_unavailable(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport(submit=_Replay([_json(503, {"error": "down"})]))
    harness = make_sandbox(transport, create_max_retries=2)

    with pytest.raises(SandboxUnavailable):
        harness.sandbox.run(_request())

    assert transport.post_count == 3, "1 attempt + create_max_retries (2)"
    assert len(harness.sleeps) == 2, "backoff must use the injected sleep"
    assert all(wait > 0 for wait in harness.sleeps)


def test_submit_5xx_then_success_recovers(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(
        submit=_Replay([_json(500, {"error": "blip"}), _submit_token()]),
        polls=_Replay([_json(200, _payload(3))]),
    )
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.outcome is SandboxOutcome.ok
    assert transport.post_count == 2
    # sleeps[0] is the 5xx backoff, sleeps[1] the single poll interval
    assert harness.sleeps[0] >= 0.5, "5xx backoff must use the injected sleep"


def test_submit_connection_error_retried_then_unavailable(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport(
        submit=_Replay([httpx.ConnectError("connection refused")]),
    )
    harness = make_sandbox(transport, create_max_retries=2)

    with pytest.raises(SandboxUnavailable):
        harness.sandbox.run(_request())

    assert transport.post_count == 3
    assert len(harness.sleeps) == 2


def test_submit_malformed_json_raises_bad_response(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport(submit=_Replay([_text(200, "{not json")]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxBadResponse):
        harness.sandbox.run(_request())

    assert transport.post_count == 1


def test_submit_missing_token_raises_bad_response(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport(submit=_Replay([_json(200, {"status": {"id": 3}})]))
    harness = make_sandbox(transport)

    with pytest.raises(SandboxBadResponse):
        harness.sandbox.run(_request())


# ── 3. Poll deadlines ────────────────────────────────────────────────

def test_poll_deadline_expiry_raises_timeout(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(1))]))
    harness = make_sandbox(transport, clock=_Clock([0.0, 100.0]))

    with pytest.raises(SandboxTimeout) as excinfo:
        harness.sandbox.run(_request())

    assert "deadline" in str(excinfo.value).lower()
    assert transport.get_count == 0, "deadline must win before the first poll"


def test_poll_max_attempts_raises_timeout(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(1))]))
    harness = make_sandbox(
        transport,
        clock=_Clock([0.0]),
        poll_max_attempts=3,
        poll_deadline_s=1_000.0,
    )

    with pytest.raises(SandboxTimeout) as excinfo:
        harness.sandbox.run(_request())

    assert "max poll attempts" in str(excinfo.value).lower()
    assert transport.get_count == 3


# ── 4. Status mapping and output decoding ────────────────────────────

@pytest.mark.parametrize(
    ("status_id", "expected"),
    [
        (3, SandboxOutcome.ok),
        (4, SandboxOutcome.ok),
        (5, SandboxOutcome.timeout),
        (6, SandboxOutcome.compile_error),
        (7, SandboxOutcome.runtime_error),
        (8, SandboxOutcome.runtime_error),
        (9, SandboxOutcome.runtime_error),
        (10, SandboxOutcome.runtime_error),
        (11, SandboxOutcome.runtime_error),
        (12, SandboxOutcome.runtime_error),
        (13, SandboxOutcome.internal_error),
        (14, SandboxOutcome.internal_error),
        (999, SandboxOutcome.internal_error),
    ],
)
def test_status_id_maps_to_outcome(
    make_sandbox: MakeSandbox, status_id: int, expected: SandboxOutcome,
) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(status_id))]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.outcome is expected
    assert response.raw_status == str(status_id)
    assert response.provider == "judge0"


def test_base64_outputs_are_decoded(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(
        3,
        stdout=_b64("hello\n"),
        stderr=_b64("Traceback\n"),
        compile_output=_b64("error: x"),
        message="provider message (not base64)",
    ))]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.stdout == "hello\n"
    assert response.stderr == "Traceback\n"
    assert response.compile_output == "error: x"
    assert response.message == "provider message (not base64)"


def test_non_utf8_output_is_replaced_not_raised(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(3, stdout=_b64(b"\xff\xfe")))]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.stdout == "\ufffd\ufffd"


def test_exit_code_and_signal_pass_through(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(
        7, exit_code=2, exit_signal=11,
    ))]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.exit_code == 2
    assert response.signal == 11


def test_time_wall_time_and_memory_are_parsed(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(polls=_Replay([_json(200, _payload(
        3, time="0.123", wall_time="0.25", memory=4096,
    ))]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.cpu_time_ms == pytest.approx(123.0)
    assert response.wall_time_ms == pytest.approx(250.0)
    assert response.memory_kb == pytest.approx(4096.0)


def test_missing_and_malformed_metrics_default_to_zero(
    make_sandbox: MakeSandbox,
) -> None:
    body: dict[str, Any] = {"status": {"id": 3}, "stdout": None, "time": "n/a"}
    transport = _Transport(polls=_Replay([_json(200, body)]))
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert response.stdout == ""
    assert response.compile_output == ""
    assert response.message == ""
    assert response.cpu_time_ms == 0.0
    assert response.wall_time_ms == 0.0
    assert response.memory_kb == 0.0


# ── 5. Guards and counters ───────────────────────────────────────────

def test_unsupported_language_raises_before_any_http_request(
    make_sandbox: MakeSandbox,
) -> None:
    transport = _Transport()
    harness = make_sandbox(transport)

    with pytest.raises(UnsupportedLanguage):
        harness.sandbox.run(_request(language="java"))

    assert transport.post_count == 0
    assert transport.get_count == 0


def test_requests_made_counts_submit_and_polls(make_sandbox: MakeSandbox) -> None:
    transport = _Transport(
        polls=_Replay([
            _json(200, _payload(1)),
            _json(200, _payload(2)),
            _json(200, _payload(3)),
        ]),
    )
    harness = make_sandbox(transport)

    response = harness.sandbox.run(_request())

    assert transport.get_count == 3
    assert response.requests_made == 4, "1 submit + 3 polls"
    assert response.latency_ms >= 0.0
