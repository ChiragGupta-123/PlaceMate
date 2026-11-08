# filepath: backend/tests/integration/test_workflow.py
"""
End-to-end workflow tests (Step 3.6c) — fully offline.

The REAL compiled graph is driven through interrupt/resume cycles with a
MemorySaver.  The only substituted layer is the sandbox: an
``ExecutionService(FakeSandbox(scenarios=...))`` is installed through
``set_execution_service`` and reset by an autouse fixture.

Assertions are made on graph state (execution_attempts, infra_retries,
optimization_probe_done, execution_history, final_verdict) rather than on
message text, except where the requirement is explicitly about what the
candidate is told (the D2 "not scored (technical problem)" feedback).
"""

from __future__ import annotations

import uuid
from typing import Iterator, List

import pytest
from langgraph.types import Command

from app.core.config import Settings
from app.core.enums import InterviewMode
from app.graph.state import make_initial_state
from app.graph.workflow import build_interview_graph
from app.rubrics.sde import RUBRIC_CATEGORIES
from app.services.execution_service import ExecutionService, set_execution_service
from app.tools.fake_sandbox import FakeSandbox

# ── Candidate submissions (never executed locally) ────────────────────

GOOD_CODE = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, num in enumerate(nums):\n"
    "        if target - num in seen:\n"
    "            return [seen[target - num], i]\n"
    "        seen[num] = i\n"
    "    return []\n"
)

SLOW_CODE = (
    "def two_sum(nums, target):\n"
    "    for i in range(len(nums)):\n"
    "        for j in range(i + 1, len(nums)):\n"
    "            if nums[i] + nums[j] == target:\n"
    "                return [i, j]\n"
    "    return []\n"
)

TEXT_ANSWER = "Here is my reasoning, step by step."


# ── Execution-service fixture (offline sandbox only) ─────────────────

def _settings() -> Settings:
    return Settings(sandbox_provider="fake")


@pytest.fixture(autouse=True)
def _offline_execution_service() -> Iterator[None]:
    """Never let a test reach a real provider; always clean the seam up."""
    set_execution_service(None)
    try:
        yield
    finally:
        set_execution_service(None)


def _install_sandbox(scenarios: List[str]) -> FakeSandbox:
    sandbox = FakeSandbox(scenarios=scenarios)
    set_execution_service(
        ExecutionService(provider=sandbox, settings=_settings()),
    )
    return sandbox


# ── Graph driver ─────────────────────────────────────────────────────

class _Interview:
    """Drives the real graph: start → interrupt → Command(resume=...) → ..."""

    def __init__(self, mode: str = InterviewMode.STANDARD.value) -> None:
        self.graph = build_interview_graph(checkpointer_backend="memory")
        self.config = {"configurable": {"thread_id": f"wf-{uuid.uuid4().hex}"}}
        self.graph.invoke(
            make_initial_state(
                session_id="wf-test",
                candidate_name="Test Candidate",
                target_company="Acme",
                target_role="SDE",
                interview_mode=mode,
            ),
            self.config,
        )

    def _snapshot(self):
        return self.graph.get_state(self.config)

    @property
    def values(self) -> dict:
        return self._snapshot().values

    @property
    def next_nodes(self) -> tuple:
        return self._snapshot().next

    def at_human_input(self) -> bool:
        return self.next_nodes == ("human_input",)

    def finished(self) -> bool:
        return self.next_nodes == ()

    def resume(
        self, answer: str = "", code: str = "", language: str = "python",
    ) -> dict:
        payload: dict = {"answer": answer, "language": language}
        if code:
            payload["code"] = code
        self.graph.invoke(Command(resume=payload), self.config)
        return self.values


def _reach_coding(interview: _Interview) -> dict:
    """Answer intro + the four technical questions → first coding question."""
    for _ in range(5):
        assert interview.at_human_input(), "expected an interview pause"
        interview.resume(answer=TEXT_ANSWER)

    values = interview.values
    assert values["current_question_type"] == "coding"
    assert values["current_problem_id"] == "two_sum"
    assert interview.at_human_input()
    return values


def _finish_with_text(interview: _Interview, cap: int = 10) -> dict:
    """Answer whatever is asked until the graph reaches END."""
    steps = 0
    while not interview.finished():
        assert interview.at_human_input(), (
            f"graph stopped at {interview.next_nodes} instead of human_input"
        )
        interview.resume(answer=TEXT_ANSWER)
        steps += 1
        assert steps <= cap, "graph did not terminate (possible infinite loop)"
    return interview.values


# ── 1. pass → complexity probe → text reply → assessment → END ───────

def test_pass_triggers_complexity_probe_then_reaches_assessment() -> None:
    _install_sandbox(["pass"])
    interview = _Interview()
    _reach_coding(interview)

    values = interview.resume(code=GOOD_CODE)

    assert values["execution_result"]["status"] == "passed"
    assert values["execution_result"]["perf_ok"] is True
    assert values["execution_attempts"] == 1
    assert values["optimization_probe_done"] is True, "standard mode must probe"
    assert values["current_question_type"] == "follow_up"
    assert values["follow_up_count"] == 1
    assert interview.at_human_input(), "the probe must wait for the candidate"

    values = interview.resume(answer="O(N) time and O(N) space, one pass.")

    assert values["current_problem_id"] == "valid_parentheses", "moved on"

    values = _finish_with_text(interview)

    assert interview.finished()
    assert values["interview_phase"] == "complete"
    assert values["final_verdict"] in {"strong_hire", "hire", "lean_no_hire", "no_hire"}
    assert values["rubric_scores"]["code_correctness"] == RUBRIC_CATEGORIES["code_correctness"]
    assert [r["status"] for r in values["execution_history"]] == ["passed"]


# ── 2. four wrong answers → exhausted, moves on ──────────────────────

def test_four_wrong_answers_exhaust_attempts_and_move_on() -> None:
    _install_sandbox(["wrong_answer"] * 4)
    interview = _Interview()
    _reach_coding(interview)

    for attempt in range(1, 5):
        values = interview.resume(code=GOOD_CODE)

        assert values["execution_result"]["status"] == "failed"
        if attempt < 4:
            assert values["execution_attempts"] == attempt
            assert values["follow_up_count"] == attempt
            assert values["current_question_type"] == "follow_up"
            assert interview.at_human_input()
        else:
            # 4th evaluated run hits MAX_EXECUTION_ATTEMPTS → "exhausted"
            assert values["follow_up_count"] == 3, "no 4th debater challenge"
            assert values["current_problem_id"] == "valid_parentheses"
            assert values["execution_attempts"] == 0, "counters reset for a new question"
            assert interview.at_human_input()

    values = interview.values
    assert [r["status"] for r in values["execution_history"]] == ["failed"] * 4
    assert interview.next_nodes == ("human_input",)


# ── 3. correct but slow → optimize → faster resubmission passes ──────

def test_slow_solution_routes_to_optimize_then_resubmission_passes() -> None:
    _install_sandbox(["slow", "pass"])
    interview = _Interview()
    _reach_coding(interview)

    values = interview.resume(code=SLOW_CODE)

    assert values["execution_result"]["status"] == "passed"
    assert values["execution_result"]["perf_ok"] is False
    assert values["execution_attempts"] == 1
    assert values["optimization_probe_done"] is True
    assert values["current_question_type"] == "follow_up"
    assert "O(N²)" in values["complexity_analysis"]
    assert interview.at_human_input()

    values = interview.resume(code=GOOD_CODE, answer="Switched to a hash map.")

    assert [r["status"] for r in values["execution_history"]] == ["passed", "passed"]
    assert values["execution_history"][-1]["perf_ok"] is True
    assert values["current_problem_id"] == "valid_parentheses", "moved on after pass"
    assert interview.at_human_input()


# ── 4a. infra error ×2 then pass → retry works, attempts untouched ───

def test_two_infra_failures_then_pass_counts_one_attempt() -> None:
    _install_sandbox(["unavailable", "unavailable", "pass"])
    interview = _Interview()
    _reach_coding(interview)

    values = interview.resume(code=GOOD_CODE)

    assert values["execution_result"]["status"] == "infra_error"
    assert values["infra_retries"] == 1
    assert values["execution_attempts"] == 0, "infra errors never consume attempts"
    assert values["current_question_type"] == "coding"
    assert interview.at_human_input()

    values = interview.resume(code=GOOD_CODE)

    assert values["execution_result"]["status"] == "infra_error"
    assert values["infra_retries"] == 2
    assert values["execution_attempts"] == 0
    assert interview.at_human_input()

    values = interview.resume(code=GOOD_CODE)

    assert values["execution_result"]["status"] == "passed"
    assert values["execution_attempts"] == 1, "only the evaluated run counted"
    assert values["infra_retries"] == 2
    assert len(values["execution_history"]) == 3
    assert values["optimization_probe_done"] is True


# ── 4b. infra error ×3 → not_evaluated, assessor skips the problem ───

def test_three_infra_failures_are_not_evaluated_and_not_scored() -> None:
    _install_sandbox(["unavailable", "unavailable", "unavailable"])
    interview = _Interview()
    _reach_coding(interview)

    values = interview.resume(code=GOOD_CODE)
    assert values["infra_retries"] == 1

    values = interview.resume(code=GOOD_CODE)
    assert values["infra_retries"] == 2

    values = interview.resume(code=GOOD_CODE)

    history_statuses = [r["status"] for r in values["execution_history"]]
    assert history_statuses == ["infra_error", "infra_error", "not_evaluated"]
    assert values["current_problem_id"] == "valid_parentheses", "moved on"
    assert interview.at_human_input()

    values = _finish_with_text(interview)

    assert interview.finished()
    assert values["final_verdict"]
    assert "not scored (technical problem)" in values["final_feedback"]


# ── 5. unsupported language → retry_input, counters untouched ────────

def test_unsupported_language_retries_input_without_counting() -> None:
    _install_sandbox(["pass"])
    interview = _Interview()
    _reach_coding(interview)

    values = interview.resume(code=GOOD_CODE, language="java")

    run = values["execution_history"][-1]
    assert run["status"] == "unsupported_language"
    assert run["provider"] == "", "the sandbox must not be called at all"
    assert run["requests_made"] == 0
    assert values["execution_attempts"] == 0
    assert values["infra_retries"] == 0
    assert values["current_question_type"] == "coding", "question unchanged"
    assert values["current_problem_id"] == "two_sum"
    assert interview.at_human_input()

    # The candidate may fix the language and resubmit on the same question.
    values = interview.resume(code=GOOD_CODE, language="python")

    assert values["execution_result"]["status"] == "passed"
    assert values["execution_attempts"] == 1


# ── 6. quick_fire mode never asks a coding question ──────────────────

def test_quick_fire_mode_skips_coding() -> None:
    _install_sandbox([])
    interview = _Interview(mode=InterviewMode.QUICK_FIRE.value)
    phases = [interview.values["interview_phase"]]

    steps = 0
    while not interview.finished():
        assert interview.at_human_input()
        interview.resume(answer=TEXT_ANSWER)
        phases.append(interview.values["interview_phase"])
        steps += 1
        assert steps <= 10, "quick_fire did not terminate"

    values = interview.values
    assert "coding" not in phases
    assert values["execution_history"] == []
    assert values["code_submissions"] == []
    assert values["interview_phase"] == "complete"
    assert values["final_verdict"]
