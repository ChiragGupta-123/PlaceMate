# filepath: backend/app/graph/nodes/executor.py
"""
PlaceMate — Executor Node (Phase 1: Mock)
==========================================

WHY THIS FILE EXISTS:
    The Executor node simulates running the candidate's code in a
    sandboxed environment.  In the real system (Phase 3+), this will:
      1. Send code to Judge0 / Piston via an execution service.
      2. Poll for results.
      3. Return structured test case outcomes.

    In Phase 1, it returns a mock result — either PASS or FAIL —
    based on a simple heuristic (whether the code contains "def ").
    This lets us test the critical routing logic:
      • Code passes → back to Interviewer for next question.
      • Code fails → go to Debater for adversarial follow-up.

HOW IT CONNECTS:
    • workflow.py registers: graph.add_node("executor", executor_node)
    • The router sends us here when current_question_type == "coding"
      and the candidate has submitted code.
    • After execution, the executor_router decides: INTERVIEWER or DEBATER.

LANGGRAPH CONCEPT — Tool Nodes vs Regular Nodes:
    LangGraph has a concept of "tool nodes" (ToolNode) for LLM tool
    calling.  Our Executor is NOT a tool node — it's a regular node
    that happens to interact with an external service.

    In Phase 3, the Executor will call Judge0 via an HTTP client,
    but it's still a regular graph node.  Tool nodes are specifically
    for when an LLM decides to call a tool (function calling).

DATA FLOW:
    Reads:  code_submission, code_language
    Writes: execution_result, execution_history, messages
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from app.graph.mocks.mock_execution import mock_execute


def executor_node(state: dict) -> dict:
    """
    Mock code execution node.

    In Phase 1, we simulate execution:
      - If the candidate's code contains "def " (a function definition),
        we consider it a "reasonable" attempt and return partial pass.
      - If it contains a nested loop pattern, we flag it as inefficient.
      - Otherwise, we return a failure.

    This mock logic exists ONLY to test the graph's routing:
      pass  → route to INTERVIEWER
      fail  → route to DEBATER

    The actual Judge0 integration comes in Phase 3.
    """

    code = state.get("code_submission", "")
    language = state.get("code_language", "python")

    # ── Mock execution logic ────────────────────────────────────────
    result = mock_execute(code, language)

    # ── Build the AI message ────────────────────────────────────────
    status = result["status"]
    if status == "pass":
        msg = (
            f"[Executor] ✅ All test cases passed! "
            f"({result['passed']}/{result['total']} tests)"
        )
    elif status == "partial":
        msg = (
            f"[Executor] ⚠️ Partial pass. "
            f"{result['passed']}/{result['total']} test cases passed.\n"
            f"Failed case: {result.get('failed_case', 'N/A')}"
        )
    else:
        msg = (
            f"[Executor] ❌ Execution failed.\n"
            f"Error: {result.get('error', 'Unknown error')}"
        )

    ai_message = AIMessage(content=msg)

    # ── Append to execution history ─────────────────────────────────
    # Note: execution_history doesn't use add_messages reducer,
    # so we need to manually append to the existing list.
    existing_history: list[dict[str, Any]] = list(
        state.get("execution_history", [])
    )
    existing_history.append(result)

    return {
        "execution_result": result,
        "execution_history": existing_history,
        "messages": [ai_message],
    }
