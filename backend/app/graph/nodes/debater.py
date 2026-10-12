# filepath: backend/app/graph/nodes/debater.py
"""
PlaceMate — Debater Node (Phase 1: Mock)
=========================================

WHY THIS FILE EXISTS:
    The Debater is the "adversarial" agent — it challenges the
    candidate's solution.  A real interviewer doesn't just say
    "wrong, next question."  They say:
      "Your solution is O(N²). Can you do better?"
      "What happens if the array is empty?"
      "You used extra space. Can you do it in-place?"

    In the real system (Phase 6+), the Debater will use an LLM to
    analyze code complexity, edge cases, and weak reasoning.

    In Phase 1, it returns mock adversarial challenges based on the
    execution result to test the graph's cyclic behavior:
      Interviewer → Executor → Debater → back to Interviewer

HOW IT CONNECTS:
    • workflow.py registers: graph.add_node("debater", debater_node)
    • We arrive here when the executor_router routes a failed/partial
      execution to the debater.
    • After the debater runs, we ALWAYS go back to the Interviewer.
    • This creates the CYCLE in the graph — a key LangGraph feature.

LANGGRAPH CONCEPT — Cycles:
    Unlike simple DAG pipelines, LangGraph supports cycles.  Our graph
    has this cycle:

      Interviewer → Executor → Debater → Interviewer → ...

    This is essential for realistic interview behavior: the candidate
    gets multiple chances to improve, just like in a real interview.

    LangGraph handles cycles via conditional edges.  The cycle
    terminates when the router decides to move on (e.g., turn limit
    reached, or code now passes).

DATA FLOW:
    Reads:  execution_result, code_submission, follow_up_count
    Writes: complexity_analysis, follow_up_count, interview_phase, messages
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.enums import QuestionType


def debater_node(state: dict) -> dict:
    """
    Mock adversarial debater.

    Inspects the execution result and generates a challenge:
      - Partial pass (inefficient): challenge on time complexity.
      - Full fail: ask about the approach / edge cases.

    In Phase 6, this becomes an LLM call that actually analyzes
    the code's AST, complexity, and edge case coverage.
    """

    execution_result: dict | None = state.get("execution_result")
    code = state.get("code_submission", "")
    follow_up_count: int = state.get("follow_up_count", 0)

    # ── Generate mock challenge ─────────────────────────────────────
    if execution_result and execution_result.get("status") == "partial":
        # Inefficient solution — challenge on complexity
        challenge = (
            "I notice your solution works for small inputs but times out "
            "on larger test cases. It appears to be O(N²) due to nested "
            "loops.\n\n"
            "💡 Can you think of a way to solve this in O(N) time? "
            "Hint: consider what data structure could help you look up "
            "values in constant time.\n\n"
            "Please revise your solution and resubmit."
        )
        analysis = "Detected O(N²) nested loop pattern. Likely needs hash map optimization."
    elif execution_result and execution_result.get("status") == "fail":
        # Failed execution — ask about approach
        error = execution_result.get("error", "unknown error")
        challenge = (
            f"Your code encountered an error: {error}\n\n"
            "Let's step back. Can you walk me through your approach "
            "before we look at the code?\n"
            "- What data structure are you using?\n"
            "- What's your algorithm's basic idea?\n"
            "- What edge cases should we handle?\n\n"
            "Take a moment to think, then you can either explain "
            "your approach or resubmit corrected code."
        )
        analysis = f"Code failed with: {error}. Candidate needs to debug."
    else:
        # Generic follow-up (shouldn't normally reach here in Phase 1)
        challenge = (
            "Interesting approach! Let me push you a bit further:\n"
            "- What's the space complexity of your solution?\n"
            "- What happens with duplicate values?\n"
            "- Is your solution stable / deterministic?\n\n"
            "Please explain your reasoning."
        )
        analysis = "Generic follow-up on solution quality."

    ai_message = AIMessage(content=f"[Debater] {challenge}")

    return {
        "complexity_analysis": analysis,
        "follow_up_count": follow_up_count + 1,
        "current_question": challenge,
        "current_question_type": QuestionType.FOLLOW_UP.value,
        "is_follow_up_pass_through": True,
        "messages": [ai_message],
    }
