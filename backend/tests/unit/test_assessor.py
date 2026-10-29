# filepath: backend/tests/unit/test_assessor.py
"""Unit tests for the mock assessor node (B5).

Only evaluated runs may influence the score.  ``infra_error`` and
``not_evaluated`` runs must be invisible to the rubric.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.graph.nodes.assessor import assessor_node
from app.rubrics.sde import RUBRIC_CATEGORIES


def _history(*statuses: str) -> list[dict[str, Any]]:
    return [
        {"status": status, "all_correct": status == "passed"}
        for status in statuses
    ]


def _state(history: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "execution_history": history,
        "code_submission": "def f(): pass",
        "total_turns": 4,
        "target_company": "Acme",
        "target_role": "SDE",
    }
    state.update(overrides)
    return state


def _correctness(delta: dict[str, Any]) -> float:
    return delta["rubric_scores"]["code_correctness"]


@pytest.mark.parametrize("unscored", ["infra_error", "not_evaluated"])
def test_unscored_problems_are_marked_not_scored(unscored: str) -> None:
    delta = assessor_node(_state(_history(unscored)))

    # The rubric keys stay unchanged; the category is simply not scored.
    assert set(delta["rubric_scores"]) == set(RUBRIC_CATEGORIES)
    assert _correctness(delta) == 0.0
    assert "not scored (technical problem)" in delta["final_feedback"]
    assert "No code submission provided" not in delta["weaknesses"]
    assert "Code passes all test cases" not in delta["strengths"]


@pytest.mark.parametrize("unscored", ["infra_error", "not_evaluated"])
def test_unscored_total_is_rescaled_over_scored_categories(unscored: str) -> None:
    delta = assessor_node(_state(_history(unscored)))

    scored = sum(
        score
        for category, score in delta["rubric_scores"].items()
        if category != "code_correctness"
    )
    excluded_weight = RUBRIC_CATEGORIES["code_correctness"]
    expected = scored / (100.0 - excluded_weight) * 100.0

    assert delta["total_score"] == pytest.approx(expected)
    assert delta["total_score"] > sum(delta["rubric_scores"].values())


def test_no_code_at_all_is_still_penalised() -> None:
    delta = assessor_node(_state([]))

    assert "No code submission provided" in delta["weaknesses"]


def test_passed_run_earns_full_correctness() -> None:
    delta = assessor_node(_state(_history("passed")))

    assert _correctness(delta) == RUBRIC_CATEGORIES["code_correctness"]
    assert "Code passes all test cases" in delta["strengths"]


def test_not_evaluated_is_ignored_next_to_a_real_failure() -> None:
    delta = assessor_node(_state(_history("not_evaluated", "failed")))

    assert _correctness(delta) < RUBRIC_CATEGORIES["code_correctness"]
    assert delta["final_verdict"] != "strong_hire"


def test_infra_error_is_ignored_next_to_a_real_pass() -> None:
    delta = assessor_node(_state(_history("infra_error", "passed")))

    assert _correctness(delta) == RUBRIC_CATEGORIES["code_correctness"]


def test_scorecard_shape_and_completion() -> None:
    delta = assessor_node(_state(_history("passed")))

    assert set(delta["rubric_scores"]) == set(RUBRIC_CATEGORIES)
    assert delta["total_score"] == sum(delta["rubric_scores"].values())
    assert delta["interview_phase"] == "complete"
    assert delta["final_verdict"]
    assert "Acme" in delta["final_feedback"]
    assert "SDE" in delta["final_feedback"]


def test_empty_history_is_safe() -> None:
    delta = assessor_node(_state([]))

    assert _correctness(delta) == 0.0
    assert delta["interview_phase"] == "complete"
