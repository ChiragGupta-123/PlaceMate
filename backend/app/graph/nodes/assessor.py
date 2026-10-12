# filepath: backend/app/graph/nodes/assessor.py
"""
PlaceMate — Assessor Node (Phase 1: Mock)
==========================================

WHY THIS FILE EXISTS:
    The Assessor is the FINAL node in the interview graph.  It runs
    after all questioning is complete and produces the structured
    hiring scorecard.

    In the real system (Phase 7+), this will:
      1. Analyze the entire conversation transcript.
      2. Review all code submissions and execution results.
      3. Apply a predefined rubric (not arbitrary LLM scores).
      4. Produce a structured Pydantic output.

    In Phase 1, it returns a mock scorecard with hardcoded scores
    to prove the graph terminates correctly and produces output.

HOW IT CONNECTS:
    • workflow.py registers: graph.add_node("assessor", assessor_node)
    • The router sends us here when interview_phase == "assessment".
    • After this node runs, the graph reaches END — interview complete.
    • This is the terminal node: no outgoing conditional edges.

LANGGRAPH CONCEPT — Terminal Nodes:
    In LangGraph, you mark the end of the graph with:
        graph.add_edge("assessor", END)

    END is a special constant from langgraph.graph.  When the graph
    reaches END, execution stops and the final state is returned.

    The Assessor → END edge is unconditional (no routing logic).
    Every interview must eventually reach this path.

DATA FLOW:
    Reads:  messages (full transcript), execution_history, target_company,
            target_role, interview_mode
    Writes: rubric_scores, total_score, final_feedback, final_verdict,
            strengths, weaknesses, interview_phase, messages
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.enums import InterviewPhase
from app.rubrics.sde import RUBRIC_CATEGORIES


# The rubric (weights summing to 100) lives in app/rubrics/sde.py so the Phase 7
# LLM assessor and this mock share one definition.


# This rubric is the SAME structure that the real LLM assessor will
# follow.  Scores are out of their max and sum to 100.
#
# By defining the rubric NOW, we ensure:
#   1. The scorecard schema is stable from Phase 1.
#   2. The frontend can be built against this structure.
#   3. The LLM (Phase 7) must conform to this, not invent its own.
# ──────────────────────────────────────────────────────────────────────



def assessor_node(state: dict) -> dict:
    """
    Mock assessor that produces a structured hiring scorecard.

    In Phase 1, scores are based on simple heuristics:
      - Did the candidate submit code?
      - Did any tests pass?
      - How many turns did the interview take?

    The STRUCTURE of the output is what matters here — the real
    LLM assessor (Phase 7) will fill in real analysis.
    """

    # ── Gather evidence from state ──────────────────────────────────
    execution_history: list = state.get("execution_history", [])
    total_turns: int = state.get("total_turns", 0)
    company: str = state.get("target_company", "Unknown")
    role: str = state.get("target_role", "Software Engineer")

    # ── Calculate mock scores ───────────────────────────────────────
    has_code = bool(state.get("code_submission"))
    any_pass = any(
        r.get("status") in ("pass", "partial")
        for r in execution_history
    )
    all_pass = any(
        r.get("status") == "pass"
        for r in execution_history
    )

    # #region agent log
    try:
        import json as _json, time as _time
        from pathlib import Path as _Path
        _lp = _Path(__file__).resolve().parents[3] / "debug-af240c.log"
        with _lp.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"af240c","hypothesisId":"A","location":"assessor.py:evidence","message":"assessor has_code vs history","data":{"has_code": has_code,"code_submission": bool(state.get("code_submission")),"code_submissions_len": len(state.get("code_submissions", [])),"exec_statuses": [r.get("status") for r in execution_history],"any_pass": any_pass,"all_pass": all_pass,"total_turns": total_turns},"timestamp": int(_time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion

    # Mock scoring heuristic
    rubric_scores: dict[str, float] = {}
    for category, max_score in RUBRIC_CATEGORIES.items():
        if category == "code_correctness":
            if all_pass:
                rubric_scores[category] = max_score
            elif any_pass:
                rubric_scores[category] = max_score * 0.6
            elif has_code:
                rubric_scores[category] = max_score * 0.3
            else:
                rubric_scores[category] = 0.0
        elif category == "problem_solving":
            # More turns = more engagement (simple heuristic)
            rubric_scores[category] = min(max_score, total_turns * 3.0)
        else:
            # Default: 60% for having participated
            rubric_scores[category] = max_score * 0.6 if total_turns > 0 else 0.0

    total_score = sum(rubric_scores.values())

    # ── Determine verdict ───────────────────────────────────────────
    if total_score >= 80:
        verdict = "strong_hire"
    elif total_score >= 60:
        verdict = "hire"
    elif total_score >= 40:
        verdict = "lean_no_hire"
    else:
        verdict = "no_hire"

    # ── Generate feedback ───────────────────────────────────────────
    strengths = []
    weaknesses = []

    if all_pass:
        strengths.append("Code passes all test cases")
    if any_pass and not all_pass:
        strengths.append("Partial solution shows problem-solving ability")
        weaknesses.append("Solution does not handle all edge cases")
    if not has_code:
        weaknesses.append("No code submission provided")

    # #region agent log
    try:
        import json as _json, time as _time
        from pathlib import Path as _Path
        _lp = _Path(__file__).resolve().parents[3] / "debug-af240c.log"
        with _lp.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"af240c","hypothesisId":"A","location":"assessor.py:feedback","message":"scorecard lists","data":{"strengths": strengths,"weaknesses": weaknesses},"timestamp": int(_time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion
    if total_turns <= 2:
        weaknesses.append("Limited engagement with interview questions")
    if total_turns >= 5:
        strengths.append("Strong engagement and persistence")

    feedback = (
        f"Interview Assessment for {company} — {role}\n"
        f"{'=' * 50}\n"
        f"Total Score: {total_score:.1f} / 100\n"
        f"Verdict: {verdict.upper().replace('_', ' ')}\n\n"
        f"Category Breakdown:\n"
    )
    for cat, score in rubric_scores.items():
        max_s = RUBRIC_CATEGORIES[cat]
        feedback += f"  {cat.replace('_', ' ').title():.<30} {score:.1f} / {max_s:.1f}\n"

    feedback += f"\nStrengths: {', '.join(strengths) if strengths else 'N/A'}\n"
    feedback += f"Areas for Improvement: {', '.join(weaknesses) if weaknesses else 'N/A'}\n"

    ai_message = AIMessage(content=f"[Assessor]\n{feedback}")

    return {
        "rubric_scores": rubric_scores,
        "total_score": total_score,
        "final_feedback": feedback,
        "final_verdict": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "interview_phase": InterviewPhase.COMPLETE.value,
        "messages": [ai_message],
    }
