# filepath: backend/tests/unit/test_limits.py
"""
Regression tests for Step 3.3: limits + status mapping (B3, C2, D2).

All tests are offline (no network, no subprocess).
Uses FakeSandbox to drive execution_service._interpret without touching Judge0.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.graph.constants import MAX_INFRA_RETRIES
from app.schemas.execution import ExecutionStatus
from app.schemas.problem import ProblemSpec
from app.services.execution_service import ExecutionService
from app.tools.code_sandbox import (
    SandboxLimits,
    SandboxOutcome,
    SandboxResponse,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _settings(**overrides) -> Settings:
    """Return a Settings object with sandbox_provider=fake (no key needed)."""
    return Settings(sandbox_provider="fake", **overrides)


def _simple_problem() -> ProblemSpec:
    return ProblemSpec.model_validate({
        "id": "two_sum",
        "title": "Two Sum",
        "function_name": "two_sum",
        "comparator": "unordered_list",
        "public_tests": [
            {"args": [[2, 7], 9], "expected": [0, 1], "timeout_ms": 1000},
        ],
        "hidden_tests": [
            {"args": [[3, 3], 6], "expected": [0, 1], "timeout_ms": 1000},
        ],
        "perf_test": {
            "args": [{"$range": [0, 100]}, 197],
            "expected": [98, 99],
            "timeout_ms": 3000,
        },
        "perf_budget_ms": 2000,
    })


def _make_sandbox_response(
    outcome: SandboxOutcome = SandboxOutcome.ok,
    stdout: str = "",
    stderr: str = "",
    cpu_time_ms: float = 10.0,
    memory_kb: int = 1000,
) -> SandboxResponse:
    return SandboxResponse(
        outcome=outcome,
        stdout=stdout,
        stderr=stderr,
        cpu_time_ms=cpu_time_ms,
        memory_kb=memory_kb,
        provider="fake",
        requests_made=1,
    )


def _make_service(provider_response: SandboxResponse) -> ExecutionService:
    """Build a real ExecutionService wired to a mock provider."""
    mock_provider = MagicMock()
    mock_provider.run.return_value = provider_response
    return ExecutionService(provider=mock_provider, settings=_settings())


# ── B3: wall/cpu cap tests ───────────────────────────────────────────

def test_max_wall_time_s_is_at_most_15():
    """B3 regression: config must never set wall time > 15s (Judge0 CE hard cap)."""
    s = _settings()
    assert s.max_wall_time_s <= 15.0, (
        f"max_wall_time_s={s.max_wall_time_s} exceeds Judge0 CE cap of 15s (B3 bug)"
    )


def test_max_cpu_time_s_is_at_most_10():
    """D2 regression: config must never set cpu time > 10s."""
    s = _settings()
    assert s.max_cpu_time_s <= 10.0, (
        f"max_cpu_time_s={s.max_cpu_time_s} exceeds D2 cap of 10s"
    )


def test_execution_service_wall_time_cap():
    """B3: SandboxLimits.wall_time_s built by execution service must never exceed 15s."""
    captured: list[SandboxLimits] = []

    def _capture_run(request):
        captured.append(request.limits)
        # Return a sandbox response with no harness output (infra error path)
        return _make_sandbox_response(SandboxOutcome.ok, stdout="")

    mock_provider = MagicMock()
    mock_provider.run.side_effect = _capture_run
    svc = ExecutionService(provider=mock_provider, settings=_settings())

    # Run with a problem that has large combined timeout budget
    big_problem = ProblemSpec.model_validate({
        "id": "two_sum",
        "title": "Two Sum",
        "function_name": "two_sum",
        "comparator": "exact",
        "public_tests": [
            {"args": [[1, 2], 3], "expected": [0, 1], "timeout_ms": 5000},
            {"args": [[1, 2], 3], "expected": [0, 1], "timeout_ms": 5000},
            {"args": [[1, 2], 3], "expected": [0, 1], "timeout_ms": 5000},
        ],
        "hidden_tests": [],
    })

    from unittest.mock import patch
    with patch("app.services.execution_service.problem_bank.get", return_value=big_problem):
        svc.run("def two_sum(a, b): return [0,1]", "python", "two_sum", attempt=1)

    assert captured, "Provider was never called"
    limits = captured[0]
    assert limits.wall_time_s <= 15.0, (
        f"wall_time_s={limits.wall_time_s} exceeds 15s cap (B3 bug)"
    )
    assert limits.cpu_time_s <= 10.0, (
        f"cpu_time_s={limits.cpu_time_s} exceeds 10s cap"
    )


# ── C2: sandbox timeout always maps to time_limit_exceeded ───────────

def test_c2_timeout_with_no_parsed_tests_maps_to_tle():
    """C2 (baseline): sandbox timeout with zero parsed tests → time_limit_exceeded."""
    svc = _make_service(_make_sandbox_response(SandboxOutcome.timeout, stdout=""))
    from unittest.mock import patch
    prob = _simple_problem()
    with patch("app.services.execution_service.problem_bank.get", return_value=prob):
        result = svc.run("def two_sum(a, b): return [0]", "python", "two_sum")
    assert result.status == ExecutionStatus.time_limit_exceeded


def test_c2_timeout_with_partial_results_still_maps_to_tle():
    """C2 (critical): sandbox timeout with PARTIAL parsed tests → still time_limit_exceeded, NOT passed.

    Previously, if test[0] passed before sandbox wall-clock expired,
    the service could return ExecutionStatus.passed even though the run timed out.
    """
    import json
    import secrets as _secrets

    prob = _simple_problem()
    nonce = _secrets.token_hex(8)
    marker = f"@@PM:{nonce}@@"
    # Emit one passing test result then no 'done' sentinel (truncated by timeout)
    partial_stdout = (
        marker + json.dumps({
            "type": "test", "index": 0, "kind": "public",
            "passed": True, "outcome": "pass", "time_ms": 5.2,
            "input_repr": "([2, 7], 9)", "expected_repr": "[0, 1]", "actual_repr": "[0, 1]",
            "error": None,
        }) + "\n"
    )

    svc = _make_service(_make_sandbox_response(SandboxOutcome.timeout, stdout=partial_stdout))
    with patch("app.services.execution_service.problem_bank.get", return_value=prob), \
         patch("app.services.execution_service.secrets.token_hex", return_value=nonce):
        result = svc.run("def two_sum(a, b): return [0, 1]", "python", "two_sum")

    assert result.status == ExecutionStatus.time_limit_exceeded, (
        f"Expected time_limit_exceeded but got {result.status} (C2 bug: "
        "partial pass with sandbox timeout should not count as passed)"
    )


# ── D2: problem JSON timeout values ──────────────────────────────────

def test_problem_json_public_hidden_timeout_ms():
    """D2: all public and hidden tests must have timeout_ms=1000."""
    from app.services.problem_bank import get as get_problem

    for pid in ("two_sum", "valid_parentheses"):
        prob = get_problem(pid)
        assert prob is not None, f"Problem '{pid}' not found in bank"
        for t in prob.public_tests:
            assert t.timeout_ms == 1000, (
                f"{pid} public test has timeout_ms={t.timeout_ms}, expected 1000 (D2)"
            )
        for t in prob.hidden_tests:
            assert t.timeout_ms == 1000, (
                f"{pid} hidden test has timeout_ms={t.timeout_ms}, expected 1000 (D2)"
            )


def test_problem_json_perf_timeout_and_budget():
    """D2: perf test must have timeout_ms=3000 and perf_budget_ms=2000."""
    from app.services.problem_bank import get as get_problem

    for pid in ("two_sum", "valid_parentheses"):
        prob = get_problem(pid)
        assert prob is not None
        assert prob.perf_test is not None, f"{pid} has no perf_test"
        assert prob.perf_test.timeout_ms == 3000, (
            f"{pid} perf timeout_ms={prob.perf_test.timeout_ms}, expected 3000 (D2)"
        )
        assert prob.perf_budget_ms == 2000, (
            f"{pid} perf_budget_ms={prob.perf_budget_ms}, expected 2000 (D2)"
        )


# ── D2: MAX_INFRA_RETRIES constant ───────────────────────────────────

def test_max_infra_retries_constant():
    """D2: MAX_INFRA_RETRIES must be 2."""
    assert MAX_INFRA_RETRIES == 2, f"MAX_INFRA_RETRIES={MAX_INFRA_RETRIES}, expected 2 (D2)"
