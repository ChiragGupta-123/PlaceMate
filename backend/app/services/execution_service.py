# filepath: backend/app/services/execution_service.py
"""
PlaceMate — Execution Service
================================

Orchestrates a single code-execution run:
  1. Validates the submission.
  2. Builds the test harness.
  3. Calls the sandbox provider.
  4. Interprets the sandbox response into an ``ExecutionResult``.

This service **never raises**.  Every failure path returns an
``ExecutionResult`` with an appropriate status.

IMPORT RULES
────────────
This module may import ``app.schemas``, ``app.services``, ``app.core``
and ``app.tools.code_sandbox`` (the interface).  It must NEVER import
``app.tools.judge0``, ``app.graph`` or ``httpx``.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Callable, Dict, List

from app.core.config import Settings, get_settings
from app.schemas.execution import (
    ErrorKind,
    ExecutionResult,
    ExecutionStatus,
    TestOutcome,
    TestOutcomeKind,
    TestOutcomeResult,
)
from app.schemas.problem import ProblemSpec
from app.services import problem_bank
from app.services.harness import ParsedRun, ParsedTestResult, build_harness, parse_harness_output
from app.tools.code_sandbox import (
    SandboxAuthError,
    SandboxBadRequest,
    SandboxBadResponse,
    SandboxError,
    SandboxLimits,
    SandboxOutcome,
    SandboxQuotaExceeded,
    SandboxRequest,
    SandboxResponse,
    SandboxTimeout,
    SandboxUnavailable,
    UnsupportedLanguage,
)

logger = logging.getLogger(__name__)

# ── Module-level test seam ──────────────────────────────────────────
_service_override: ExecutionService | None = None


class ExecutionService:
    """Runs candidate code against a problem's test suite.

    Parameters
    ----------
    provider : object implementing SandboxInterface
    settings : application Settings
    """

    def __init__(
        self,
        provider: Any,
        settings: Settings | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings or get_settings()

    def run(
        self,
        source: str,
        language: str,
        problem_id: str,
        attempt: int = 1,
        trace_id: str | None = None,
    ) -> ExecutionResult:
        """Execute candidate code.  Never raises."""
        t0 = time.monotonic()
        try:
            return self._run_inner(source, language, problem_id, attempt, trace_id, t0)
        except Exception:
            logger.exception(
                "event=execution_service_crash problem_id=%s attempt=%d",
                problem_id, attempt,
            )
            return ExecutionResult(
                problem_id=problem_id,
                language=language,
                attempt=attempt,
                status=ExecutionStatus.infra_error,
                error="Internal execution service error",
                error_kind=ErrorKind.internal,
                latency_ms=_elapsed(t0),
            )

    def _run_inner(
        self,
        source: str,
        language: str,
        problem_id: str,
        attempt: int,
        trace_id: str | None,
        t0: float,
    ) -> ExecutionResult:
        base = dict(problem_id=problem_id, language=language, attempt=attempt)

        # ── 1. Validate ─────────────────────────────────────────────
        if language != "python":
            return ExecutionResult(
                **base,
                status=ExecutionStatus.unsupported_language,
                error=f"Language '{language}' is not supported",
                latency_ms=_elapsed(t0),
            )

        if not source or not source.strip():
            return ExecutionResult(
                **base,
                status=ExecutionStatus.invalid_submission,
                error="Empty submission",
                latency_ms=_elapsed(t0),
            )

        if len(source.encode("utf-8")) > self._settings.max_source_bytes:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.invalid_submission,
                error=f"Source exceeds {self._settings.max_source_bytes} byte limit",
                latency_ms=_elapsed(t0),
            )

        prob = problem_bank.get(problem_id)
        if prob is None:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.infra_error,
                error=f"Unknown problem_id: {problem_id}",
                error_kind=ErrorKind.config,
                latency_ms=_elapsed(t0),
            )

        # ── 2. Build harness and limits ──────────────────────────────
        nonce = secrets.token_hex(8)
        harness = build_harness(source, prob, nonce)

        total_timeout_ms = sum(t.timeout_ms for t in prob.public_tests)
        total_timeout_ms += sum(t.timeout_ms for t in prob.hidden_tests)
        if prob.perf_test:
            total_timeout_ms += prob.perf_test.timeout_ms

        cpu_time = min(self._settings.max_cpu_time_s, total_timeout_ms / 1000.0 + 1)
        wall_time = min(self._settings.max_wall_time_s, 2 * cpu_time + 1)
        mem_kb = min(self._settings.default_memory_kb, prob.memory_limit_kb)

        limits = SandboxLimits(
            cpu_time_s=cpu_time,
            wall_time_s=wall_time,
            memory_kb=mem_kb,
        )
        request = SandboxRequest(
            source_code=harness,
            language="python",
            limits=limits,
            tags={"problem_id": problem_id, "trace_id": trace_id or ""},
        )

        # ── 3. Call provider ─────────────────────────────────────────
        response: SandboxResponse | None = None
        try:
            response = self._provider.run(request)
            # Retry once on internal_error
            if response.outcome == SandboxOutcome.internal_error:
                response = self._provider.run(request)
        except SandboxQuotaExceeded:
            return self._infra_result(base, t0, ErrorKind.quota, "Sandbox quota exceeded", response)
        except SandboxAuthError:
            return self._infra_result(base, t0, ErrorKind.auth, "Sandbox authentication failed", response)
        except SandboxTimeout:
            return self._infra_result(base, t0, ErrorKind.timeout, "Sandbox poll deadline expired", response)
        except SandboxUnavailable:
            return self._infra_result(base, t0, ErrorKind.network, "Sandbox unavailable", response)
        except SandboxBadRequest:
            return self._infra_result(base, t0, ErrorKind.bad_response, "Sandbox rejected request", response)
        except SandboxBadResponse:
            return self._infra_result(base, t0, ErrorKind.bad_response, "Malformed sandbox response", response)
        except UnsupportedLanguage:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.unsupported_language,
                error="Language not supported by sandbox",
                latency_ms=_elapsed(t0),
            )
        except SandboxError:
            return self._infra_result(base, t0, ErrorKind.internal, "Unknown sandbox error", response)

        # ── 4. Interpret response ────────────────────────────────────
        return self._interpret(response, prob, nonce, base, t0, limits)

    def _infra_result(
        self,
        base: Dict[str, Any],
        t0: float,
        error_kind: ErrorKind,
        error: str,
        response: SandboxResponse | None,
    ) -> ExecutionResult:
        return ExecutionResult(
            **base,
            status=ExecutionStatus.infra_error,
            error=error,
            error_kind=error_kind,
            provider=response.provider if response else "",
            requests_made=response.requests_made if response else 0,
            latency_ms=_elapsed(t0),
        )

    def _interpret(
        self,
        response: SandboxResponse,
        prob: ProblemSpec,
        nonce: str,
        base: Dict[str, Any],
        t0: float,
        limits: SandboxLimits,
    ) -> ExecutionResult:
        parsed = parse_harness_output(response.stdout, nonce)

        # Build expected test layout
        layout: List[Dict[str, str]] = []
        for _ in prob.public_tests:
            layout.append({"kind": "public"})
        for _ in prob.hidden_tests:
            layout.append({"kind": "hidden"})
        if prob.perf_test:
            layout.append({"kind": "perf"})

        total_expected = len(layout)
        parsed_by_idx: Dict[int, ParsedTestResult] = {t.index: t for t in parsed.tests}

        # Handle compile error
        if parsed.compile_error is not None:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.compile_error,
                compile_error=parsed.compile_error,
                error=parsed.compile_error,
                stdout=response.stdout,
                stderr=response.stderr,
                partial=not parsed.done,
                provider=response.provider,
                requests_made=response.requests_made,
                latency_ms=_elapsed(t0),
            )

        # Handle missing function
        if parsed.fn_missing is not None:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.runtime_error,
                error=f"Function '{parsed.fn_missing}' not found",
                stdout=response.stdout,
                stderr=response.stderr,
                partial=not parsed.done,
                provider=response.provider,
                requests_made=response.requests_made,
                latency_ms=_elapsed(t0),
            )

        # Build TestOutcome list
        test_outcomes: List[TestOutcome] = []
        for i, tc_info in enumerate(layout):
            kind_str = tc_info["kind"]
            kind = TestOutcomeKind(kind_str)
            ptr = parsed_by_idx.get(i)
            if ptr is None:
                test_outcomes.append(TestOutcome(
                    index=i, kind=kind, passed=False, outcome=TestOutcomeResult.not_run,
                ))
            else:
                outcome_val = ptr.outcome
                # Map string to enum safely
                try:
                    outcome_enum = TestOutcomeResult(outcome_val)
                except ValueError:
                    outcome_enum = TestOutcomeResult.not_run
                test_outcomes.append(TestOutcome(
                    index=i, kind=kind,
                    passed=ptr.passed,
                    outcome=outcome_enum,
                    time_ms=ptr.time_ms,
                    input_repr=ptr.input_repr,
                    expected_repr=ptr.expected_repr,
                    actual_repr=ptr.actual_repr,
                    error=ptr.error,
                ))

        # Compute counts (public + hidden only, not perf)
        correctness_tests = [t for t in test_outcomes if t.kind in (TestOutcomeKind.public, TestOutcomeKind.hidden)]
        passed_count = sum(1 for t in correctness_tests if t.passed)
        failed_count = len(correctness_tests) - passed_count
        total_count = len(correctness_tests)
        all_correct = (passed_count == total_count and total_count > 0)

        # Compute perf_ok
        perf_ok: bool | None = None
        if prob.perf_test is not None:
            perf_tests = [t for t in test_outcomes if t.kind == TestOutcomeKind.perf]
            if not perf_tests or not perf_tests[0].passed:
                perf_ok = False
            elif perf_tests[0].time_ms is not None and prob.perf_budget_ms is not None:
                perf_ok = perf_tests[0].time_ms <= prob.perf_budget_ms
            else:
                perf_ok = perf_tests[0].passed

        is_partial = not parsed.done

        # Determine failed_case
        failed_case: str | None = None
        first_fail = next(
            (t for t in test_outcomes
             if not t.passed and t.kind in (TestOutcomeKind.public, TestOutcomeKind.hidden)),
            None,
        )
        if first_fail is not None:
            if first_fail.kind == TestOutcomeKind.public:
                parts = []
                if first_fail.input_repr:
                    parts.append(f"Input: {first_fail.input_repr}")
                if first_fail.expected_repr:
                    parts.append(f"Expected: {first_fail.expected_repr}")
                if first_fail.actual_repr:
                    parts.append(f"Got: {first_fail.actual_repr}")
                if first_fail.error:
                    parts.append(f"Error: {first_fail.error}")
                failed_case = "; ".join(parts) if parts else f"Test {first_fail.index} failed"
            else:
                failed_case = f"hidden test #{first_fail.index - len(prob.public_tests) + 1} failed"

        # Determine status
        status: ExecutionStatus
        if all_correct:
            status = ExecutionStatus.passed
        elif not parsed.tests and response.outcome == SandboxOutcome.timeout:
            status = ExecutionStatus.time_limit_exceeded
        elif not parsed.tests and response.outcome == SandboxOutcome.internal_error:
            return ExecutionResult(
                **base,
                status=ExecutionStatus.infra_error,
                error="Sandbox internal error with no test results",
                error_kind=ErrorKind.provider_internal,
                provider=response.provider,
                requests_made=response.requests_made,
                latency_ms=_elapsed(t0),
            )
        elif not parsed.tests and response.outcome in (SandboxOutcome.ok, SandboxOutcome.runtime_error):
            status = ExecutionStatus.runtime_error
        else:
            # Find first failing correctness test
            if first_fail is not None:
                if first_fail.outcome == TestOutcomeResult.wrong_answer:
                    status = ExecutionStatus.failed
                elif first_fail.outcome == TestOutcomeResult.timeout:
                    status = ExecutionStatus.time_limit_exceeded
                elif first_fail.outcome == TestOutcomeResult.exception:
                    # Check memory limit
                    if (response.memory_kb >= 0.9 * limits.memory_kb
                            and limits.memory_kb > 0):
                        status = ExecutionStatus.memory_limit_exceeded
                    else:
                        status = ExecutionStatus.runtime_error
                else:
                    status = ExecutionStatus.failed
            else:
                status = ExecutionStatus.failed

        # Runtime: max across all test times
        runtime_ms = max(
            (t.time_ms for t in test_outcomes if t.time_ms is not None),
            default=response.cpu_time_ms,
        )

        result = ExecutionResult(
            **base,
            status=status,
            all_correct=all_correct,
            perf_ok=perf_ok,
            passed=passed_count,
            failed=failed_count,
            total=total_count,
            tests=test_outcomes,
            failed_case=failed_case,
            stdout=response.stdout,
            stderr=response.stderr,
            runtime_ms=runtime_ms,
            memory_kb=response.memory_kb,
            partial=is_partial,
            provider=response.provider,
            requests_made=response.requests_made,
            latency_ms=_elapsed(t0),
        )

        logger.info(
            "event=execution_complete problem_id=%s status=%s all_correct=%s "
            "perf_ok=%s attempts=%d requests_made=%d latency_ms=%.1f provider=%s",
            result.problem_id, result.status.value, result.all_correct,
            result.perf_ok, result.attempt, result.requests_made,
            result.latency_ms, result.provider,
        )
        return result


def _elapsed(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)


# ── Singleton / test seam ───────────────────────────────────────────

_cached_service: ExecutionService | None = None


def get_execution_service() -> ExecutionService:
    """Return the cached execution service, or build one from settings."""
    global _cached_service, _service_override
    if _service_override is not None:
        return _service_override
    if _cached_service is None:
        from app.tools import get_sandbox_provider
        settings = get_settings()
        provider = get_sandbox_provider(settings)
        _cached_service = ExecutionService(provider=provider, settings=settings)
    return _cached_service


def set_execution_service(service: ExecutionService | None) -> None:
    """Test seam: override the singleton returned by ``get_execution_service``.

    Pass ``None`` to clear the override and revert to the default.
    """
    global _service_override
    _service_override = service
