# filepath: backend/app/tools/judge0.py
"""
PlaceMate — Judge0 CE Sandbox Provider (RapidAPI)
===================================================

The ONLY file that knows about Judge0, RapidAPI, and httpx.

Implements ``SandboxInterface`` via a submit → poll loop with
configurable timeouts, retries, and jittered exponential backoff.

IMPORT RULES
────────────
This module is imported ONLY by the factory in ``app/tools/__init__.py``,
and lazily (inside the function body) so that ``httpx`` is loaded only
when ``sandbox_provider == "judge0"``.
"""

from __future__ import annotations

import base64
import logging
import random
import time
from typing import Any, Callable, Dict

import httpx

from app.core.config import Settings
from app.tools.code_sandbox import (
    SandboxAuthError,
    SandboxBadRequest,
    SandboxBadResponse,
    SandboxOutcome,
    SandboxQuotaExceeded,
    SandboxRequest,
    SandboxResponse,
    SandboxTimeout,
    SandboxUnavailable,
    UnsupportedLanguage,
)

logger = logging.getLogger(__name__)

# ── Judge0 status-ID → SandboxOutcome mapping ──────────────────────
_STATUS_MAP: Dict[int, SandboxOutcome] = {
    3: SandboxOutcome.ok,             # Accepted
    4: SandboxOutcome.ok,             # Wrong Answer (we grade ourselves)
    5: SandboxOutcome.timeout,        # Time Limit Exceeded
    6: SandboxOutcome.compile_error,  # Compilation Error
    7: SandboxOutcome.runtime_error,  # SIGSEGV
    8: SandboxOutcome.runtime_error,  # SIGXFSZ
    9: SandboxOutcome.runtime_error,  # SIGFPE
    10: SandboxOutcome.runtime_error,  # SIGABRT
    11: SandboxOutcome.runtime_error,  # NZEC
    12: SandboxOutcome.runtime_error,  # Other
    13: SandboxOutcome.internal_error,  # Internal Error
    14: SandboxOutcome.internal_error,  # Exec Format Error
}

# Status IDs that mean "still running"
_PENDING_STATUSES = {1, 2}  # In Queue, Processing

# Fields to request on poll to minimize response size
_POLL_FIELDS = (
    "status_id,stdout,stderr,compile_output,message,"
    "exit_code,exit_signal,time,wall_time,memory"
)


def _b64_encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _b64_decode(val: str | None) -> str:
    if not val:
        return ""
    try:
        return base64.b64decode(val).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_time_ms(val: Any) -> float:
    """Judge0 returns time as seconds in a string, convert to ms."""
    if val is None:
        return 0.0
    try:
        return float(val) * 1000.0
    except (TypeError, ValueError):
        return 0.0


class Judge0Sandbox:
    """Judge0 CE sandbox via RapidAPI.

    Parameters
    ----------
    settings : application Settings
    client : optional pre-built httpx.Client (for testing with MockTransport)
    sleep : injectable sleep function (for testing without real waits)
    monotonic : injectable clock function (for testing without real time)
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._monotonic = monotonic
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=settings.judge0_base_url,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout_s,
                read=settings.read_timeout_s,
                write=settings.read_timeout_s,
                pool=settings.read_timeout_s,
            ),
        )
        # Build language map from config
        self._lang_map: Dict[str, int] = {
            "python": settings.judge0_python_language_id,
        }

    def __enter__(self) -> Judge0Sandbox:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def run(self, request: SandboxRequest) -> SandboxResponse:
        """Submit code, poll for result, return SandboxResponse.

        Raises only SandboxError subclasses.  Never leaks credentials.
        """
        lang_id = self._lang_map.get(request.language)
        if lang_id is None:
            raise UnsupportedLanguage(
                f"Language '{request.language}' is not supported"
            )

        requests_made = 0
        t0 = self._monotonic()

        # ── Submit ──────────────────────────────────────────────────
        use_wait = self._settings.judge0_use_wait
        payload: Dict[str, Any] = {
            "source_code": _b64_encode(request.source_code),
            "language_id": lang_id,
            "cpu_time_limit": request.limits.cpu_time_s,
            "wall_time_limit": request.limits.wall_time_s,
            "memory_limit": request.limits.memory_kb,
        }
        if request.stdin:
            payload["stdin"] = _b64_encode(request.stdin)

        wait_param = "true" if use_wait else "false"
        url = f"/submissions?base64_encoded=true&wait={wait_param}"

        resp = self._submit_with_retries(url, payload)
        requests_made += 1

        try:
            data = resp.json()
        except Exception:
            raise SandboxBadResponse("Malformed JSON in submission response")

        # If wait=true and we got a final status, return immediately
        if use_wait and self._is_final(data):
            return self._build_response(data, requests_made, t0)

        token = data.get("token")
        if not token:
            raise SandboxBadResponse("No token in submission response")

        # ── Poll ────────────────────────────────────────────────────
        deadline = t0 + self._settings.poll_deadline_s
        interval = self._settings.poll_interval_initial_s
        consecutive_errors = 0
        poll_count = 0

        while True:
            if self._monotonic() >= deadline:
                raise SandboxTimeout(
                    f"Poll deadline of {self._settings.poll_deadline_s}s expired"
                )
            if poll_count >= self._settings.poll_max_attempts:
                raise SandboxTimeout(
                    f"Max poll attempts ({self._settings.poll_max_attempts}) reached"
                )

            # Jittered sleep
            jitter = interval * random.uniform(-0.1, 0.1)
            self._sleep(max(0.01, interval + jitter))

            poll_url = (
                f"/submissions/{token}"
                f"?base64_encoded=true&fields={_POLL_FIELDS}"
            )
            try:
                poll_resp = self._client.get(
                    poll_url,
                    headers=self._headers(),
                )
                requests_made += 1
                poll_count += 1
                self._check_rate_limit(poll_resp)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                consecutive_errors = 0
            except SandboxQuotaExceeded:
                raise
            except SandboxAuthError:
                raise
            except (httpx.HTTPStatusError, httpx.RequestError, Exception) as exc:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise SandboxUnavailable(
                        f"3 consecutive poll failures"
                    ) from exc
                interval = min(
                    interval * self._settings.poll_backoff,
                    self._settings.poll_interval_max_s,
                )
                continue

            status_id = poll_data.get("status", {}).get("id") if isinstance(
                poll_data.get("status"), dict
            ) else poll_data.get("status_id")

            if status_id in _PENDING_STATUSES:
                interval = min(
                    interval * self._settings.poll_backoff,
                    self._settings.poll_interval_max_s,
                )
                continue

            # Final status
            return self._build_response(poll_data, requests_made, t0)

    # ── Private helpers ─────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        """Build request headers. Key is read at send time only."""
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-RapidAPI-Host": self._settings.judge0_rapidapi_host,
        }
        key = self._settings.judge0_rapidapi_key
        if key is not None:
            h["X-RapidAPI-Key"] = key.get_secret_value()
        return h

    def _check_rate_limit(self, resp: httpx.Response) -> None:
        """Check for 429 or zero remaining quota."""
        if resp.status_code == 429:
            remaining = resp.headers.get("X-RateLimit-Requests-Remaining", "?")
            limit = resp.headers.get("X-RateLimit-Requests-Limit", "?")
            logger.warning(
                "event=rate_limited remaining=%s limit=%s", remaining, limit,
            )
            raise SandboxQuotaExceeded("Rate limit exceeded (HTTP 429)")
        remaining = resp.headers.get("X-RateLimit-Requests-Remaining")
        if remaining is not None:
            try:
                if int(remaining) <= 0:
                    logger.warning("event=quota_exhausted remaining=0")
                    raise SandboxQuotaExceeded("Rate limit remaining is 0")
            except ValueError:
                pass

    def _submit_with_retries(
        self, url: str, payload: Dict[str, Any],
    ) -> httpx.Response:
        """POST with retries on 5xx and connection errors.

        NOTE: A retry may create a duplicate submission on Judge0.
        This is acceptable — we only read results from the token we get back.
        """
        last_exc: Exception | None = None
        for attempt in range(1 + self._settings.create_max_retries):
            try:
                resp = self._client.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                )
                self._check_rate_limit(resp)

                if resp.status_code in (401, 403):
                    raise SandboxAuthError("Authentication failed")
                if resp.status_code == 422:
                    raise SandboxBadRequest("Limits rejected by provider")
                if 400 <= resp.status_code < 500:
                    raise SandboxBadRequest(
                        f"Request rejected (HTTP {resp.status_code})"
                    )
                if resp.status_code >= 500:
                    last_exc = SandboxUnavailable(
                        f"Server error (HTTP {resp.status_code})"
                    )
                    if attempt < self._settings.create_max_retries:
                        wait = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                        self._sleep(wait)
                        continue
                    raise last_exc

                return resp

            except (SandboxQuotaExceeded, SandboxAuthError, SandboxBadRequest,
                    SandboxBadResponse):
                raise
            except SandboxUnavailable:
                raise
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < self._settings.create_max_retries:
                    wait = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                    self._sleep(wait)
                    continue

        raise SandboxUnavailable("All submission retries exhausted") from last_exc

    def _is_final(self, data: Dict[str, Any]) -> bool:
        """Check if a response contains a final (non-pending) status."""
        status = data.get("status")
        if isinstance(status, dict):
            sid = status.get("id")
        else:
            sid = data.get("status_id")
        if sid is None:
            return False
        return sid not in _PENDING_STATUSES

    def _build_response(
        self,
        data: Dict[str, Any],
        requests_made: int,
        t0: float,
    ) -> SandboxResponse:
        """Convert Judge0 JSON response to SandboxResponse."""
        # Extract status_id
        status = data.get("status")
        if isinstance(status, dict):
            status_id = status.get("id", 0)
        else:
            status_id = data.get("status_id", 0)
        if not isinstance(status_id, int):
            try:
                status_id = int(status_id)
            except (TypeError, ValueError):
                status_id = 0

        outcome = _STATUS_MAP.get(status_id, SandboxOutcome.internal_error)

        return SandboxResponse(
            outcome=outcome,
            stdout=_b64_decode(data.get("stdout")),
            stderr=_b64_decode(data.get("stderr")),
            compile_output=_b64_decode(data.get("compile_output")),
            message=data.get("message") or "",
            exit_code=data.get("exit_code"),
            signal=data.get("exit_signal"),
            cpu_time_ms=_parse_time_ms(data.get("time")),
            wall_time_ms=_parse_time_ms(data.get("wall_time")),
            memory_kb=float(data.get("memory") or 0),
            raw_status=str(status_id),
            provider="judge0",
            requests_made=requests_made,
            latency_ms=round((self._monotonic() - t0) * 1000, 1),
        )
