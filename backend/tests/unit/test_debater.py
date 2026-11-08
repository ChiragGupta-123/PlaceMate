# filepath: backend/tests/unit/test_debater.py
"""Unit tests for the mock debater node (B4).

The debater must speak the Phase-3 status vocabulary and must set
``optimization_probe_done`` when it asks the complexity probe.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.graph.nodes.debater import debater_node

EVALUATED_FAILURES = [
    "failed",
    "compile_error",
    "runtime_error",
    "time_limit_exceeded",
    "memory_limit_exceeded",
]


def _state(status: str, **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "execution_result": {"status": status, "error": "boom", "perf_ok": True},
        "code_submission": "def f(): pass",
        "follow_up_count": 2,
    }
    state.update(overrides)
    return state


def test_passed_sets_optimization_probe_done() -> None:
    delta = debater_node(_state("passed"))

    assert delta["optimization_probe_done"] is True
    assert delta["follow_up_count"] == 3
    assert delta["current_question_type"] == "follow_up"
    assert delta["is_follow_up_pass_through"] is True
    assert len(delta["messages"]) == 1


def test_passed_but_slow_sets_optimization_probe_done() -> None:
    state = _state("passed")
    state["execution_result"]["perf_ok"] = False

    delta = debater_node(state)

    assert delta["optimization_probe_done"] is True
    assert "o(n)" in delta["current_question"].lower()


@pytest.mark.parametrize("status", EVALUATED_FAILURES)
def test_failing_statuses_do_not_mark_probe_done(status: str) -> None:
    delta = debater_node(_state(status))

    assert delta.get("optimization_probe_done") is not True
    assert delta["current_question"]
    assert delta["follow_up_count"] == 3


def test_compile_error_is_named_in_the_challenge() -> None:
    delta = debater_node(_state("compile_error"))

    assert "compile" in delta["current_question"].lower()


def test_runtime_error_includes_the_error_text() -> None:
    delta = debater_node(_state("runtime_error"))

    assert "boom" in delta["current_question"]


def test_time_limit_status_mentions_complexity() -> None:
    delta = debater_node(_state("time_limit_exceeded"))

    assert "time" in delta["current_question"].lower()


def test_memory_limit_status_mentions_memory() -> None:
    delta = debater_node(_state("memory_limit_exceeded"))

    assert "memory" in delta["current_question"].lower()


def test_none_execution_result_does_not_crash() -> None:
    delta = debater_node({"execution_result": None, "follow_up_count": 0})

    assert delta["current_question"]
    assert delta.get("optimization_probe_done") is not True


def test_legacy_pass_partial_status_is_not_treated_as_probe() -> None:
    delta = debater_node(_state("partial"))

    assert delta.get("optimization_probe_done") is not True
