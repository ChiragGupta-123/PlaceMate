# filepath: backend/app/graph/state.py
"""
PlaceMate — Interview State
============================

WHY THIS FILE EXISTS:
    LangGraph is a *stateful* graph framework.  Every node receives the
    full state, reads what it needs, and returns ONLY the fields it wants
    to update.  This file is the single source of truth for that schema.

DEPENDENCY RULE (prevents circular imports):
    state.py imports ONLY from app.core.enums and third-party libraries.
    It must never import nodes, routers, mocks or workflow.  Everything
    else in app.graph may import state.py, never the reverse.

LANGGRAPH CONCEPT — State & Reducers:
    Default behaviour is OVERWRITE: returning {"current_turn": 4} replaces
    the old value.  A *reducer* changes how an update is merged.  The
    `messages` field uses `add_messages`, so returning {"messages": [m]}
    APPENDS m instead of erasing the history.  Fields without a reducer
    (e.g. code_submissions, execution_history) are plain overwrites, so
    nodes currently copy-append-return the whole list.
    (Hardening step: give those lists an operator.add reducer.)

NOTE ON TYPES:
    Enum members are stored as their string VALUES ("standard", "coding").
    That is what the checkpointer serialises, so the schema says `str`.

NOTE ON `total=False`:
    The state is populated progressively, so no key is required to exist
    at every moment.  Use make_initial_state() to start with every key set.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.core.enums import Difficulty, InterviewMode, InterviewPhase


class InterviewState(TypedDict, total=False):
    """The complete interview state that flows through every LangGraph node."""

    # ── 1. Session identity ─────────────────────────────────────────
    session_id: str
    candidate_name: str

    # ── 2. Interview configuration ──────────────────────────────────
    target_company: str
    target_role: str
    interview_mode: str                      # InterviewMode value
    difficulty: str                          # Difficulty value

    # ── 3. Phase & turn tracking ────────────────────────────────────
    interview_phase: str                     # InterviewPhase value
    current_turn: int                        # Turn counter within current phase
    total_turns: int                         # Total turns across all phases
    max_turns_per_phase: int
    phases_completed: list[str]

    # ── 4. Current question ─────────────────────────────────────────
    current_question: str
    current_question_type: str | None        # QuestionType value

    # ── 5. Candidate response ───────────────────────────────────────
    candidate_answer: str
    code_submission: str
    code_language: str

    # ── 6. Code execution ───────────────────────────────────────────
    execution_result: dict[str, Any] | None
    execution_history: list[dict[str, Any]]

    # ── 7. Debater analysis ─────────────────────────────────────────
    complexity_analysis: str
    follow_up_count: int
    is_follow_up_pass_through: bool

    # ── 8. Assessment / scorecard ───────────────────────────────────
    rubric_scores: dict[str, float] | None
    total_score: float
    final_feedback: str
    final_verdict: str
    strengths: list[str]
    weaknesses: list[str]

    # ── 9. Conversation history (THE critical reducer) ──────────────
    messages: Annotated[list[AnyMessage], add_messages]

    # ── 10. Code submission history ─────────────────────────────────
    code_submissions: list[dict[str, Any]]


def make_initial_state(
    *,
    session_id: str,
    candidate_name: str,
    target_company: str,
    target_role: str,
    interview_mode: str = InterviewMode.STANDARD.value,
    difficulty: str = Difficulty.MEDIUM.value,
    starting_phase: str = InterviewPhase.INTRO.value,
) -> InterviewState:
    """
    Build a fully-populated starting state.

    Replaces the 25-field dict that was copy-pasted into every demo/test.
    tests/unit/test_state.py asserts this factory covers every key of
    InterviewState, so the two cannot silently drift apart.
    """
    return {
        "session_id": session_id,
        "candidate_name": candidate_name,
        "target_company": target_company,
        "target_role": target_role,
        "interview_mode": interview_mode,
        "difficulty": difficulty,
        "interview_phase": starting_phase,
        "current_turn": 0,
        "total_turns": 0,
        "max_turns_per_phase": 3,
        "phases_completed": [],
        "current_question": "",
        "current_question_type": None,
        "candidate_answer": "",
        "code_submission": "",
        "code_language": "python",
        "execution_result": None,
        "execution_history": [],
        "complexity_analysis": "",
        "follow_up_count": 0,
        "is_follow_up_pass_through": False,
        "rubric_scores": None,
        "total_score": 0.0,
        "final_feedback": "",
        "final_verdict": "",
        "strengths": [],
        "weaknesses": [],
        "messages": [],
        "code_submissions": [],
    }
