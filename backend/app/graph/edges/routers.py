# filepath: backend/app/graph/edges/routers.py
"""
PlaceMate — Routing Functions
==============================

WHY THIS FILE EXISTS:
    In LangGraph, routing functions are the "brains" that decide
    WHERE the graph goes next.  They inspect the current state
    and return the NAME of the next node to execute.

    These functions are passed to:
        graph.add_conditional_edges(source_node, routing_function, path_map)

    The routing function receives the current state and returns a string
    that maps to a node name via the path_map.

HOW IT CONNECTS:
    • workflow.py uses these functions in add_conditional_edges().
    • Each router reads state fields set by the preceding node.
    • The return value MUST be a key in the path_map dict, which
      maps to a NodeName.

LANGGRAPH CONCEPT — Conditional Edges:
    LangGraph has two types of edges:

    1. UNCONDITIONAL:  graph.add_edge("a", "b")
       → After node A, ALWAYS go to node B.

    2. CONDITIONAL:  graph.add_conditional_edges("a", router_fn, {"x": "b", "y": "c"})
       → After node A, call router_fn(state).
       → If it returns "x", go to node B.
       → If it returns "y", go to node C.

    This is how we implement branching, loops, and dynamic interview flow.
    The router function is a pure Python function — no LLM needed.
    It just reads state and returns a string.

DATA FLOW:
    after_human_input_router:
        Reads:  interview_phase, current_question_type, current_turn
        Returns: "technical" | "coding" | "assessment"

    after_executor_router:
        Reads:  execution_result
        Returns: "pass" | "fail"
"""

from __future__ import annotations

from app.core.enums import InterviewMode, InterviewPhase, QuestionType
from app.graph.constants import DEFAULT_MAX_TURNS_PER_PHASE, MODE_TURN_BUDGETS


def after_human_input_router(state: dict) -> str:
    """
    Decides where to go AFTER the interviewer asks a question
    and the candidate responds (post-interrupt).

    Decision logic:
        1. If phase is ASSESSMENT → go to assessor (terminal).
        2. If the last question was a CODING question and the
           candidate submitted code → go to executor.
        3. Otherwise → loop back to interviewer (next question).

    The "loop back to interviewer" path is what makes the graph
    cyclic for technical questions.  The turn counter in the
    interviewer node handles phase advancement.

    Returns one of: "technical", "coding", "assessment"
    These are KEYS in the path_map, not NodeName values.
    """

    phase_str = state.get("interview_phase", InterviewPhase.INTRO.value)
    question_type = state.get("current_question_type")
    code = state.get("code_submission", "")
    phase = InterviewPhase(phase_str)
    mode_str = state.get("interview_mode", InterviewMode.STANDARD.value)
    mode = InterviewMode(mode_str)
    current_turn = state.get("current_turn", 0)

    # ── 1. Check if we should go to assessment ──────────────────────
    if phase in (InterviewPhase.ASSESSMENT, InterviewPhase.COMPLETE):
        return "assessment"

    # Check turn budget — if current phase turns exhausted, advance
    turn_budgets = MODE_TURN_BUDGETS.get(mode, {})
    max_turns = turn_budgets.get(phase, DEFAULT_MAX_TURNS_PER_PHASE)
    total_turns = state.get("total_turns", 0)

    # Safety: if we've done a LOT of total turns, force assessment
    overall_max = sum(turn_budgets.values()) if turn_budgets else 8
    # #region agent log
    try:
        import json as _json, time as _time
        from pathlib import Path as _Path
        _lp = _Path(__file__).resolve().parents[3] / "debug-af240c.log"
        with _lp.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"af240c","hypothesisId":"B","location":"routers.py:after_human_input","message":"budget check","data":{"phase": phase_str,"question_type": question_type,"current_turn": current_turn,"max_turns": max_turns,"total_turns": total_turns,"overall_max": overall_max,"used_max_turns": False,"code_present": bool(code and str(code).strip())},"timestamp": int(_time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    if total_turns >= overall_max:
        return "assessment"

    # ── 2. If coding question or follow up and candidate submitted code → execute
    if question_type in (QuestionType.CODING.value, QuestionType.FOLLOW_UP.value) and code.strip():
        return "coding"

    # ── 3. If we're in follow-up phase, check follow-up budget ──────
    if question_type == QuestionType.FOLLOW_UP.value:
        follow_up_count = state.get("follow_up_count", 0)
        max_follow_ups = turn_budgets.get(
            InterviewPhase.FOLLOW_UP, DEFAULT_MAX_TURNS_PER_PHASE
        )
        if follow_up_count >= max_follow_ups:
            # We exhausted follow-ups. Return to interviewer to advance to the next coding question.
            return "technical"
        # If candidate answered with text but no code, route to interviewer to ask next coding question
        return "technical"

    # ── 4. Default: continue with more interviewer questions ────────
    return "technical"


def after_executor_router(state: dict) -> str:
    """
    Decides where to go AFTER code execution.

    Returns one of:
        "retry_input"  → back to human_input (unsupported lang / invalid)
        "infra_error"  → to interviewer (sandbox failure, not candidate's fault)
        "optimize"     → to debater (correct but slow, attempts remain)
        "probe"        → to debater (correct + fast, complexity probe)
        "pass"         → to interviewer (move on)
        "exhausted"    → to interviewer (attempts used up, move on)
        "fail"         → to debater (wrong answer / runtime error)
    """
    from app.graph.constants import MAX_EXECUTION_ATTEMPTS, OPTIMIZATION_PROBE_MODES

    result = state.get("execution_result") or {}
    status = result.get("status", "failed")
    attempts = state.get("execution_attempts", 0)
    mode = state.get("interview_mode", "standard")

    # 1. Unsupported language or invalid submission → retry
    if status in ("unsupported_language", "invalid_submission"):
        return "retry_input"

    # 2. Infrastructure error → not candidate's fault
    if status == "infra_error":
        return "infra_error"

    # 3. Passed
    if status == "passed":
        perf_ok = result.get("perf_ok")

        # 3a. Correct but slow → optimize route (if attempts remain)
        if perf_ok is False:
            if attempts >= MAX_EXECUTION_ATTEMPTS:
                return "pass"
            return "optimize"

        # 3b. Correct + fast → probe once in standard/deep_dive modes
        if mode in OPTIMIZATION_PROBE_MODES and not state.get("optimization_probe_done", False):
            return "probe"

        # 3c. Just pass
        return "pass"

    # 4. Failing statuses
    if attempts >= MAX_EXECUTION_ATTEMPTS:
        return "exhausted"
    return "fail"
