# filepath: backend/tests/unit/test_executor.py
"""Unit tests for the executor node (B6/B7 + C3 accounting).

The executor talks to ``ExecutionService`` through the module-level test
seam, so these tests use a stub service and never touch a sandbox.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.graph.constants import MAX_INFRA_RETRIES
from app.graph.nodes.executor import executor_node
from app.schemas.execution import ExecutionResult, ExecutionStatus
from app.services.execution_service import set_execution_service


class _StubService:
    """Minimal ExecutionService stand-in that records calls."""

    def __init__(
        self,
        result: ExecutionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error

    def run(
        self,
        source: str,
        language: str,
        problem_id: str,
        attempt: int = 1,
        trace_id: str | None = None,
    ) -> ExecutionResult:
        self.calls.append(
            {
                "source": source,
                "language": language,
                "problem_id": problem_id,
                "attempt": attempt,
                "trace_id": trace_id,
            }
        )
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


@pytest.fixture(autouse=True)
def _clear_service_override() -> Any:
    yield
    set_execution_service(None)


def _state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "code_submission": "def two_sum(nums, target): return [0, 1]",
        "code_language": "python",
        "current_problem_id": "two_sum",
        "execution_attempts": 0,
        "infra_retries": 0,
        "session_id": "sess-1",
    }
    state.update(overrides)
    return state


def _result(status: ExecutionStatus, **overrides: Any) -> ExecutionResult:
    data: dict[str, Any] = {
        "problem_id": "two_sum",
        "language": "python",
        "attempt": 1,
        "status": status,
    }
    data.update(overrides)
    return ExecutionResult(**data)


# ── B7: None problem id must never crash ────────────────────────────


def test_none_problem_id_is_normalised_before_service_call() -> None:
    stub = _StubService(result=_result(ExecutionStatus.passed))
    set_execution_service(stub)  # type: ignore[arg-type]

    executor_node(_state(current_problem_id=None))

    assert stub.calls[0]["problem_id"] == ""


def test_none_problem_id_does_not_crash_in_except_handler() -> None:
    stub = _StubService(error=RuntimeError("boom"))
    set_execution_service(stub)  # type: ignore[arg-type]

    delta = executor_node(_state(current_problem_id=None))

    assert delta["execution_result"]["status"] == "infra_error"
    assert delta["execution_result"]["problem_id"] == ""


# ── C3: accounting ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.passed,
        ExecutionStatus.failed,
        ExecutionStatus.compile_error,
        ExecutionStatus.runtime_error,
        ExecutionStatus.time_limit_exceeded,
        ExecutionStatus.memory_limit_exceeded,
    ],
)
def test_evaluated_run_increments_attempts_only(status: ExecutionStatus) -> None:
    set_execution_service(_StubService(result=_result(status)))  # type: ignore[arg-type]

    delta = executor_node(_state(execution_attempts=1, infra_retries=1))

    assert delta["execution_attempts"] == 2
    assert delta["infra_retries"] == 1


@pytest.mark.parametrize(
    "status",
    [ExecutionStatus.unsupported_language, ExecutionStatus.invalid_submission],
)
def test_unscorable_submission_increments_neither_counter(
    status: ExecutionStatus,
) -> None:
    set_execution_service(_StubService(result=_result(status)))  # type: ignore[arg-type]

    delta = executor_node(_state(execution_attempts=2, infra_retries=1))

    assert delta["execution_attempts"] == 2
    assert delta["infra_retries"] == 1


def test_infra_error_increments_retries_and_not_attempts() -> None:
    set_execution_service(  # type: ignore[arg-type]
        _StubService(result=_result(ExecutionStatus.infra_error, error="down"))
    )

    delta = executor_node(_state(execution_attempts=3, infra_retries=0))

    assert delta["execution_attempts"] == 3
    assert delta["infra_retries"] == 1
    assert delta["execution_result"]["status"] == "infra_error"
    assert delta["messages"]


# ── D2: two retries, the third infra error becomes not_evaluated ────


@pytest.mark.parametrize("retries_before", [0, MAX_INFRA_RETRIES - 1])
def test_infra_error_within_retry_budget_stays_retryable(
    retries_before: int,
) -> None:
    set_execution_service(  # type: ignore[arg-type]
        _StubService(result=_result(ExecutionStatus.infra_error, error="down"))
    )

    delta = executor_node(_state(infra_retries=retries_before))

    assert delta["infra_retries"] == retries_before + 1
    assert delta["execution_result"]["status"] == "infra_error"


def test_third_infra_error_becomes_not_evaluated() -> None:
    set_execution_service(  # type: ignore[arg-type]
        _StubService(result=_result(ExecutionStatus.infra_error, error="down"))
    )

    delta = executor_node(_state(execution_attempts=2, infra_retries=MAX_INFRA_RETRIES))

    assert delta["infra_retries"] == MAX_INFRA_RETRIES + 1
    assert delta["execution_attempts"] == 2
    assert delta["execution_result"]["status"] == "not_evaluated"
    assert delta["messages"]
    assert "not" in delta["messages"][0].content.lower()


# ── History / messages ──────────────────────────────────────────────


def test_history_contains_exactly_the_new_result() -> None:
    set_execution_service(  # type: ignore[arg-type]
        _StubService(result=_result(ExecutionStatus.passed))
    )

    delta = executor_node(_state())

    assert len(delta["execution_history"]) == 1
    assert delta["execution_history"][0]["status"] == "passed"
    assert "messages" not in delta


def test_user_message_only_for_non_candidate_fault_statuses() -> None:
    invalid = _result(ExecutionStatus.invalid_submission, error="Empty submission")
    set_execution_service(_StubService(result=invalid))  # type: ignore[arg-type]

    delta = executor_node(_state(code_submission=""))

    assert delta["messages"][0].content
