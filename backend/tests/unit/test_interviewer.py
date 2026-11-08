# filepath: backend/tests/unit/test_interviewer.py
"""Unit tests for the mock interviewer node (C3 counter resets)."""

from __future__ import annotations

from typing import Any

from app.graph.nodes.interviewer import interviewer_node


def _coding_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "interview_phase": "coding",
        "current_turn": 0,
        "total_turns": 1,
        "target_company": "Acme",
        "target_role": "SDE",
        "interview_mode": "standard",
        "execution_attempts": 3,
        "infra_retries": 2,
        "optimization_probe_done": True,
        "is_follow_up_pass_through": False,
    }
    state.update(overrides)
    return state


def test_new_question_resets_both_counters_and_probe_flag() -> None:
    delta = interviewer_node(_coding_state())

    assert delta["execution_attempts"] == 0
    assert delta["infra_retries"] == 0
    assert delta["optimization_probe_done"] is False


def test_new_question_uses_the_mock_problem_id_and_counts_a_turn() -> None:
    delta = interviewer_node(_coding_state())

    assert delta["current_problem_id"] == "two_sum"
    assert delta["current_question_type"] == "coding"
    assert delta["total_turns"] == 2
    assert delta["current_turn"] == 1


def test_debater_pass_through_only_clears_the_flag() -> None:
    delta = interviewer_node(
        _coding_state(is_follow_up_pass_through=True)
    )

    assert delta == {"is_follow_up_pass_through": False}


def test_assessment_phase_asks_nothing() -> None:
    delta = interviewer_node(_coding_state(interview_phase="assessment"))

    assert delta["current_question"] == ""
    assert delta["interview_phase"] == "assessment"
