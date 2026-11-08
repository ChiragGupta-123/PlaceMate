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
    follow_up_count: int = state.get("follow_up_count") or 0
    status = execution_result.get("status") if execution_result else None
    error = (execution_result or {}).get("error") or "unknown error"

    # ── Generate mock challenge ─────────────────────────────────────
    probe_done = False
    if status == "passed":
        # Correct solution → ask the complexity probe exactly once.
        # The router guards this with optimization_probe_done, so the
        # debater is responsible for setting the flag.
        probe_done = True
        if execution_result is not None and execution_result.get("perf_ok") is False:
            challenge = (
                "Your solution passes every test case, but it is too slow on "
                "the largest inputs. It appears to be O(N²) due to nested "
                "loops.\n\n"
                "💡 Can you think of a way to solve this in O(N) time? "
                "Hint: consider what data structure could help you look up "
                "values in constant time.\n\n"
                "Please revise your solution and resubmit."
            )
            analysis = (
                "Correct but over the perf budget. Detected O(N²) pattern; "
                "likely needs hash map optimization."
            )
        else:
            challenge = (
                "Your solution is correct and within the performance budget. "
                "Before we move on, let's probe the design:\n"
                "- What is the time complexity, and why?\n"
                "- What is the space complexity?\n"
                "- Which input shape is the worst case for your approach?"
            )
            analysis = "Correct and performant. Complexity probe requested."
    elif status == "failed":
        # Wrong answer — ask about the approach
        challenge = (
            f"Your code produced an incorrect result: {error}\n\n"
            "Let's step back. Can you walk me through your approach "
            "before we look at the code?\n"
            "- What data structure are you using?\n"
            "- What's your algorithm's basic idea?\n"
            "- What edge cases should we handle?\n\n"
            "Take a moment to think, then you can either explain "
            "your approach or resubmit corrected code."
        )
        analysis = f"Wrong answer: {error}. Candidate needs to revisit the approach."
    elif status == "compile_error":
        challenge = (
            f"Your code did not compile: {error}\n\n"
            "Let's fix the syntax first. Check the reported line, then "
            "resubmit. If the message is unclear, walk me through the code "
            "line by line."
        )
        analysis = f"Compile error: {error}."
    elif status == "runtime_error":
        challenge = (
            f"Your code hit a runtime error: {error}\n\n"
            "That means the program started but stopped unexpectedly. "
            "Which input triggers it, and what assumption does your code "
            "make that might not hold?\n\n"
            "Please fix it and resubmit."
        )
        analysis = f"Runtime error: {error}. Candidate needs to debug."
    elif status == "time_limit_exceeded":
        challenge = (
            "Your solution exceeded the time limit on the larger test cases. "
            "That usually means the algorithm is O(N²) or worse.\n\n"
            "💡 Can you think of a way to solve this in O(N) time? "
            "What data structure would let you avoid the nested loop?\n\n"
            "Please revise your solution and resubmit."
        )
        analysis = "Time limit exceeded. Likely O(N²) or worse; needs optimization."
    elif status == "memory_limit_exceeded":
        challenge = (
            "Your solution exceeded the memory limit. It is using more space "
            "than the problem allows.\n\n"
            "- Which data structures hold the most data?\n"
            "- Can you process the input in a single pass instead of storing it?\n\n"
            "Please revise your solution and resubmit."
        )
        analysis = "Memory limit exceeded. Candidate needs to reduce space usage."
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

    delta: dict = {
        "complexity_analysis": analysis,
        "follow_up_count": follow_up_count + 1,
        "current_question": challenge,
        "current_question_type": QuestionType.FOLLOW_UP.value,
        "is_follow_up_pass_through": True,
        "messages": [ai_message],
    }
    if probe_done:
        delta["optimization_probe_done"] = True
    return delta
