# filepath: backend/app/graph/nodes/interviewer.py
"""
PlaceMate — Interviewer Node (Phase 1: Mock)
=============================================

WHY THIS FILE EXISTS:
    The Interviewer is the "host" of the interview.  In the real system
    (Phase 4+), this will call an LLM to generate company-specific,
    role-aware, adaptive questions.

    In Phase 1, it's a mock: it returns hardcoded questions based on
    the current phase and turn number.  This lets us prove the graph
    topology (routing, cycles, interrupts) without needing an API key.

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

    For example, if this node returns:
        {"current_question": "What is a hash map?", "messages": [ai_msg]}
    Then LangGraph will:
        1. Overwrite state["current_question"] with the new string.
        2. APPEND the ai_msg to state["messages"] (because of add_messages).
        3. Leave ALL other fields untouched.

DATA FLOW:
    Reads:  interview_phase, current_turn, target_company, target_role, interview_mode
    Writes: current_question, current_question_type, interview_phase,
            current_turn, messages
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.enums import InterviewMode, InterviewPhase
from app.graph.mocks.mock_questions import MOCK_QUESTIONS


# Mock questions live in app/graph/mocks/mock_questions.py (temporary; Phase 4
# replaces them with LLM + RAG generated questions).


# In Phase 4, these will come from the LLM + RAG context.
# For now, hardcoded questions let us test the graph flow.
# ──────────────────────────────────────────────────────────────────────




def interviewer_node(state: dict) -> dict:
    """
    Mock interviewer node.

    Logic:
        1. Determine the current phase (default to INTRO if not set).
        2. Pick a question from the mock bank based on phase + turn.
        3. If the phase has run out of questions, advance to the next phase.
        4. Return the question, updated phase, and an AI message.

    In Phase 4, this function will call the LLM instead of indexing
    into a hardcoded list.
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

    # ── Handle ASSESSMENT or COMPLETE — nothing to ask ──────────────
    if phase in (InterviewPhase.ASSESSMENT, InterviewPhase.COMPLETE):
        # #region agent log
        try:
            import json as _json, time as _time
            from pathlib import Path as _Path
            _lp = _Path(__file__).resolve().parents[3] / "debug-af240c.log"
            with _lp.open("a", encoding="utf-8") as _f:
                _f.write(_json.dumps({"sessionId":"af240c","hypothesisId":"C","location":"interviewer.py:assessment_passthrough","message":"interviewer returning empty q still edges to human_input","data":{"phase": phase.value,"current_turn": current_turn,"code_submission": bool(state.get("code_submission")),"code_submissions_len": len(state.get("code_submissions", []))},"timestamp": int(_time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        return {
            "interview_phase": phase.value,
            "current_question": "",
            "current_question_type": None,
        }

    # ── Pick the question ───────────────────────────────────────────
    q_index = min(current_turn, len(phase_questions) - 1)
    q_data = phase_questions[q_index]

    question_text = q_data["question"].replace("{company}", company).replace("{role}", role)
    question_type = q_data["type"]

    # ── Build the AI message ────────────────────────────────────────
    ai_message = AIMessage(content=f"[Interviewer] {question_text}")

    # ── Return state update ─────────────────────────────────────────
    # Note: we return `messages: [ai_message]` — because of the
    # add_messages reducer, this APPENDS to the conversation history
    # rather than replacing it.
    return {
        "current_question": question_text,
        "current_question_type": question_type.value,
        "interview_phase": phase.value,
        "current_turn": current_turn + 1,
        "total_turns": state.get("total_turns", 0) + 1,
        "messages": [ai_message],
    }


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
