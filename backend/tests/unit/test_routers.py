# filepath: backend/tests/unit/test_routers.py
"""Routers are pure functions of state, so they need no graph to test."""

from __future__ import annotations

from app.graph.edges.routers import after_executor_router, after_human_input_router


def test_executor_pass_goes_to_interviewer():
    assert after_executor_router({"execution_result": {"status": "pass"}}) == "pass"


def test_executor_failure_goes_to_debater():
    for status in ("fail", "partial", "error"):
        assert after_executor_router({"execution_result": {"status": status}}) == "fail"


def test_coding_question_with_code_goes_to_executor():
    state = {
        "interview_phase": "coding",
        "current_question_type": "coding",
        "code_submission": "def f(): pass",
        "total_turns": 1,
    }
    assert after_human_input_router(state) == "coding"


def test_assessment_phase_goes_to_assessor():
    assert after_human_input_router({"interview_phase": "assessment"}) == "assessment"
