# filepath: backend/tests/unit/test_routers.py
"""
Unit tests for routing functions.

Routers are pure functions of state — no graph, no LLM, no sandbox needed.
Covers the full Phase 3 label matrix for after_executor_router and the
existing after_human_input_router paths.
"""

from __future__ import annotations

import pytest

from app.graph.constants import MAX_EXECUTION_ATTEMPTS, OPTIMIZATION_PROBE_MODES
from app.graph.edges.routers import after_executor_router, after_human_input_router


# ── after_executor_router ───────────────────────────────────────────

def _exec_state(
    status: str = "passed",
    perf_ok: bool | None = True,
    attempts: int = 1,
    mode: str = "standard",
    probe_done: bool = False,
    result_override: dict | None = None,
) -> dict:
    result = result_override if result_override is not None else {
        "status": status,
        "perf_ok": perf_ok,
    }
    return {
        "execution_result": result,
        "execution_attempts": attempts,
        "interview_mode": mode,
        "optimization_probe_done": probe_done,
    }


class TestAfterExecutorRouter:
    # ── None / empty result never crashes ──────────────────────────
    def test_none_result_returns_fail(self) -> None:
        assert after_executor_router({"execution_result": None}) == "fail"

    def test_empty_state_returns_fail(self) -> None:
        assert after_executor_router({}) == "fail"

    # ── unsupported / invalid ───────────────────────────────────────
    def test_unsupported_language_returns_retry_input(self) -> None:
        assert after_executor_router(
            _exec_state(status="unsupported_language")
        ) == "retry_input"

    def test_invalid_submission_returns_retry_input(self) -> None:
        assert after_executor_router(
            _exec_state(status="invalid_submission")
        ) == "retry_input"

    # ── infra error ─────────────────────────────────────────────────
    def test_infra_error_returns_infra_error(self) -> None:
        assert after_executor_router(
            _exec_state(status="infra_error")
        ) == "infra_error"

    # ── passed + perf ────────────────────────────────────────────────
    def test_passed_perf_ok_standard_probes_once(self) -> None:
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=True, mode="standard", probe_done=False)
        ) == "probe"

    def test_passed_after_probe_done_returns_pass(self) -> None:
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=True, mode="standard", probe_done=True)
        ) == "pass"

    def test_passed_non_probe_mode_returns_pass(self) -> None:
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=True, mode="quick", probe_done=False)
        ) == "pass"

    def test_passed_perf_ok_none_probes_in_standard_mode(self) -> None:
        # perf_ok=None means no perf test — still probe on first pass
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=None, mode="standard", probe_done=False)
        ) == "probe"

    def test_correct_but_slow_routes_optimize(self) -> None:
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=False, attempts=1)
        ) == "optimize"

    def test_correct_but_slow_at_max_attempts_returns_pass(self) -> None:
        assert after_executor_router(
            _exec_state(status="passed", perf_ok=False, attempts=MAX_EXECUTION_ATTEMPTS)
        ) == "pass"

    # ── failing statuses ────────────────────────────────────────────
    @pytest.mark.parametrize("status", [
        "failed", "compile_error", "runtime_error",
        "time_limit_exceeded", "memory_limit_exceeded",
    ])
    def test_failing_status_returns_fail(self, status: str) -> None:
        assert after_executor_router(
            _exec_state(status=status, attempts=1)
        ) == "fail"

    def test_exhausted_after_max_attempts(self) -> None:
        assert after_executor_router(
            _exec_state(status="failed", attempts=MAX_EXECUTION_ATTEMPTS)
        ) == "exhausted"

    def test_not_exhausted_below_max(self) -> None:
        assert after_executor_router(
            _exec_state(status="failed", attempts=MAX_EXECUTION_ATTEMPTS - 1)
        ) == "fail"


# ── after_human_input_router ────────────────────────────────────────

def test_coding_question_with_code_goes_to_executor() -> None:
    state = {
        "interview_phase": "coding",
        "current_question_type": "coding",
        "code_submission": "def f(): pass",
        "total_turns": 1,
    }
    assert after_human_input_router(state) == "coding"


def test_assessment_phase_goes_to_assessor() -> None:
    assert after_human_input_router({"interview_phase": "assessment"}) == "assessment"


def test_text_answer_goes_to_technical() -> None:
    state = {
        "interview_phase": "technical",
        "current_question_type": "technical",
        "code_submission": "",
        "total_turns": 1,
    }
    assert after_human_input_router(state) == "technical"
