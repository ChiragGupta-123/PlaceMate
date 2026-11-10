# filepath: backend/app/graph/nodes/interviewer.py
"""
PlaceMate — Interviewer Node (Phase 4: QuestionService)
=======================================================

WHY THIS FILE EXISTS:
    The Interviewer is the "host" of the interview.  It decides WHEN to
    ask a question; the actual question text comes from the QuestionService
    (mock bank today, LLM when configured).

DETERMINISTIC VS GENERATED
──────────────────────────
Everything that controls the interview stays in code:
    • phase advancement (and the quick-fire shortcut),
    • the turn counter and per-phase budgets,
    • the question TYPE for the current phase,
    • the assessment/complete terminal behaviour,
    • the counter resets for every new question.

The service only supplies text (+ a bank problem id for coding).  It is
called through ``get_question_service()`` so the graph never imports
``app.tools``.

HOW IT CONNECTS:
    • workflow.py registers this as: graph.add_node("interviewer", interviewer_node)
    • The graph enters this node at START and after every cycle.
    • After this node runs, the graph hits an INTERRUPT — it pauses
      and waits for the candidate's response.
    • When resumed, the router inspects the state and decides where
      to go next.

LANGGRAPH CONCEPT — Nodes:
    A LangGraph node is just a function: (state) -> partial_state_update.
    The function receives the FULL current state and returns a dict
    with ONLY the fields it wants to change.  LangGraph merges the
    update back into the state using the configured reducers.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage

from app.core.enums import Difficulty, InterviewMode, InterviewPhase, QuestionType
from app.graph.mocks.mock_questions import MOCK_QUESTIONS
from app.services.interviewer_prompts import (
    MAX_COVERED_TOPICS,
    MAX_TRANSCRIPT_TURNS,
    CandidateTurn,
    update_interview_summary,
)
from app.services.question_service import (
    FallbackQuestion,
    QuestionRequest,
    get_question_service,
)

_INTERVIEWER_PREFIX = "[Interviewer] "


def interviewer_node(state: dict) -> dict:
    """Ask the next question via the QuestionService.

    Logic:
        1. Determine the current phase (default to INTRO if not set).
        2. Pick the mock question for phase + turn as the deterministic
           fallback (the service returns it in mock mode or on failure).
        3. If the phase has run out of questions, advance to the next phase.
        4. Ask the service for the question and return the state update.
    """

    # ── Read current state ──────────────────────────────────────────
    phase_str: str = state.get("interview_phase", InterviewPhase.INTRO)
    phase = InterviewPhase(phase_str) if isinstance(phase_str, str) else phase_str
    current_turn: int = state.get("current_turn", 0)
    company: str = state.get("target_company", "Unknown Company")
    role: str = state.get("target_role", "Software Engineer")
    mode_str: str = state.get("interview_mode", InterviewMode.STANDARD)
    mode = InterviewMode(mode_str) if isinstance(mode_str, str) else mode_str

    # ── Check if we just came from Debater ──────────────────────────
    if state.get("is_follow_up_pass_through"):
        # Debater just asked a follow-up. Pass through to human_input.
        # We must clear the flag so we don't pass through again!
        return {"is_follow_up_pass_through": False}

    # ── Determine next phase if needed ──────────────────────────────
    # If we're in a phase with no more mock questions, advance.
    phase_questions = MOCK_QUESTIONS.get(phase, [])

    if current_turn >= len(phase_questions):
        phase = _advance_phase(phase, mode)
        current_turn = 0
        phase_questions = MOCK_QUESTIONS.get(phase, [])

    summary = _updated_summary(state)

    # ── Handle ASSESSMENT or COMPLETE — nothing to ask ──────────────
    if phase in (InterviewPhase.ASSESSMENT, InterviewPhase.COMPLETE):
        return {
            "interview_phase": phase.value,
            "current_question": "",
            "current_question_type": None,
            "interview_summary": summary,
        }

    # ── Build the deterministic fallback for this turn ──────────────
    q_index = min(current_turn, len(phase_questions) - 1)
    q_data = phase_questions[q_index]

    fallback = FallbackQuestion(
        text=q_data["question"].replace("{company}", company).replace("{role}", role),
        question_type=q_data["type"],
        problem_id=q_data.get("problem_id"),  # None for non-coding questions
    )

    # ── Ask the service (LLM or mock, never raises) ─────────────────
    request = QuestionRequest(
        phase=phase,
        mode=mode,
        company=company,
        role=role,
        difficulty=_difficulty(state),
        candidate_name=state.get("candidate_name", "Candidate"),
        current_turn=current_turn,
        expected_question_type=_expected_question_type(phase),
        fallback=fallback,
        history=_recent_history(state.get("messages", [])),
        covered_topics=state.get("covered_topics", []),
        interview_summary=summary,
        used_problem_ids=state.get("asked_problem_ids", []),
        llm_calls_so_far=state.get("llm_calls", 0),
    )
    result = get_question_service().get_question(request)

    # ── Build the AI message ────────────────────────────────────────
    ai_message = AIMessage(content=f"{_INTERVIEWER_PREFIX}{result.text}")

    # ── Return state update ─────────────────────────────────────────
    # Note: we return `messages: [ai_message]` — because of the
    # add_messages reducer, this APPENDS to the conversation history
    # rather than replacing it.
    return {
        "current_question": result.text,
        "current_question_type": result.question_type.value,
        "interview_phase": phase.value,
        "current_turn": current_turn + 1,
        "total_turns": state.get("total_turns", 0) + 1,
        "current_problem_id": result.problem_id,
        "execution_attempts": 0,
        "infra_retries": 0,
        "optimization_probe_done": False,
        "llm_calls": state.get("llm_calls", 0) + result.llm_calls_used,
        "interview_summary": summary,
        "covered_topics": _merge_topics(
            state.get("covered_topics", []), result.topic
        ),
        "asked_problem_ids": _merge_problem_ids(
            state.get("asked_problem_ids", []), result.problem_id
        ),
        "messages": [ai_message],
    }


# ── Helpers ──────────────────────────────────────────────────────────

def _expected_question_type(phase: InterviewPhase) -> QuestionType:
    """The question type is decided by code, never by the model."""
    if phase == InterviewPhase.INTRO:
        return QuestionType.BEHAVIORAL
    if phase == InterviewPhase.CODING:
        return QuestionType.CODING
    return QuestionType.TECHNICAL


def _difficulty(state: dict) -> Difficulty:
    value = state.get("difficulty", Difficulty.MEDIUM.value)
    if isinstance(value, Difficulty):
        return value
    try:
        return Difficulty(value)
    except ValueError:
        return Difficulty.MEDIUM


def _updated_summary(state: dict) -> str:
    """Fold the previous Q/A into the rolling summary (no LLM call)."""
    answer = state.get("candidate_answer", "") or ""
    if not answer and state.get("code_submission"):
        answer = "[code submitted]"
    return update_interview_summary(
        state.get("interview_summary", "") or "",
        state.get("current_question", "") or "",
        answer,
    )


def _merge_topics(existing: Sequence[str], topic: str) -> list[str]:
    topics = [item for item in existing if item]
    if topic and topic not in topics:
        topics.append(topic)
    return topics[-MAX_COVERED_TOPICS:]


def _merge_problem_ids(
    existing: Sequence[str],
    problem_id: str | None,
) -> list[str]:
    ids = list(existing)
    if problem_id and problem_id not in ids:
        ids.append(problem_id)
    return ids


def _recent_history(messages: Sequence[Any]) -> list[CandidateTurn]:
    """Pair the last AI questions with the candidate's answers."""
    turns: list[CandidateTurn] = []
    pending: str | None = None

    for message in messages:
        if isinstance(message, AIMessage):
            pending = _message_text(message)
            if pending.startswith(_INTERVIEWER_PREFIX):
                pending = pending[len(_INTERVIEWER_PREFIX):]
        elif isinstance(message, HumanMessage) and pending is not None:
            turns.append(
                CandidateTurn(question=pending, answer=_message_text(message))
            )
            pending = None

    return turns[-MAX_TRANSCRIPT_TURNS:]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _advance_phase(current: InterviewPhase, mode: InterviewMode) -> InterviewPhase:
    """
    Determine the next interview phase based on the current phase
    and interview mode.

    Quick Fire mode skips CODING and goes from TECHNICAL → ASSESSMENT.
    Other modes follow the full progression.
    """
    phase_order: list[InterviewPhase] = [
        InterviewPhase.INTRO,
        InterviewPhase.TECHNICAL,
        InterviewPhase.CODING,
        InterviewPhase.ASSESSMENT,
    ]

    # Quick Fire: skip coding
    if mode == InterviewMode.QUICK_FIRE:
        phase_order = [
            InterviewPhase.INTRO,
            InterviewPhase.TECHNICAL,
            InterviewPhase.ASSESSMENT,
        ]

    try:
        idx = phase_order.index(current)
        if idx + 1 < len(phase_order):
            return phase_order[idx + 1]
        return InterviewPhase.ASSESSMENT
    except ValueError:
        return InterviewPhase.ASSESSMENT
