# filepath: backend/app/graph/nodes/human_input.py
"""Interrupt point: pauses the graph until the candidate responds."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt


def human_input_node(state: dict) -> Command:
    """
    This node is the INTERRUPT POINT — the graph pauses here.

    HOW INTERRUPT WORKS:
    ────────────────────
    1. Graph reaches this node after the Interviewer asks a question.
    2. interrupt() is called with a prompt describing what we're waiting for.
    3. The checkpointer saves the entire state.
    4. The graph returns an Interrupt to the caller (FastAPI / demo script).
    5. The caller collects the candidate's response.
    6. The caller resumes the graph with:
           graph.invoke(Command(resume={"answer": "...", "code": "..."}), config)
    7. interrupt() returns the resume value.
    8. This node processes the value and returns a state update.
    9. The graph continues to the router.

    The resume value is a dict with optional keys:
      - "answer": str  — verbal response to a question
      - "code": str     — code submission
      - "language": str — programming language

    This is the human-in-the-loop pattern.
    """

    # ── PAUSE HERE — wait for candidate ─────────────────────────────
    # The string argument is metadata about what we're waiting for.
    # It's included in the Interrupt object returned to the caller.
    question_type = state.get("current_question_type", "technical")
    if question_type == "coding":
        prompt = "Waiting for candidate code submission..."
    else:
        prompt = "Waiting for candidate response..."

    # This call BLOCKS the graph.  Execution stops here until
    # Command(resume=...) is called with the same thread_id.
    candidate_input = interrupt(prompt)

    # ── Process the resume value ────────────────────────────────────
    # After resume, candidate_input contains whatever was passed in
    # Command(resume=...).
    answer = candidate_input.get("answer", "") if isinstance(candidate_input, dict) else str(candidate_input)
    code = candidate_input.get("code", "") if isinstance(candidate_input, dict) else ""
    language = candidate_input.get("language", "python") if isinstance(candidate_input, dict) else "python"

    # Build a human message for the conversation history
    if code:
        content = f"[Code Submission ({language})]\n```{language}\n{code}\n```"
        if answer:
            content = f"{answer}\n\n{content}"
    else:
        content = answer

    human_message = HumanMessage(content=content)

    # Track code submission in history
    updates: dict[str, Any] = {
        "candidate_answer": answer,
        "messages": [human_message],
    }

    if code:
        updates["code_submission"] = code
        updates["code_language"] = language
        existing_submissions = list(state.get("code_submissions", []))
        existing_submissions.append({
            "code": code,
            "language": language,
            "turn": state.get("total_turns", 0),
        })
        updates["code_submissions"] = existing_submissions
    else:
        updates["code_submission"] = ""
        updates["code_language"] = ""

    # #region agent log
    try:
        import json as _json, time as _time
        from pathlib import Path as _Path
        _lp = _Path(__file__).resolve().parents[3] / "debug-af240c.log"
        with _lp.open("a", encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"af240c","hypothesisId":"A","location":"human_input.py:after_resume","message":"human_input code fields","data":{"cleared_code": not bool(code),"code_len": len(code or ""),"q_type": question_type,"phase": state.get("interview_phase"),"hist_len": len(state.get("code_submissions", [])),"exec_len": len(state.get("execution_history", []))},"timestamp": int(_time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion

    # (current_turn is now managed by interviewer_node to avoid skipping questions on follow-up)
    return Command(update=updates)
