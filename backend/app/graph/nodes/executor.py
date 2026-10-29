# filepath: backend/app/graph/nodes/executor.py
"""
PlaceMate — Executor Node (Phase 3)
=====================================

Thin graph node that delegates to ``ExecutionService``.

This node knows LangGraph state fields only.  It must NEVER import
``app.tools``, ``httpx``, or know about Judge0.

DATA FLOW:
    Reads:  code_submission, code_language, current_problem_id,
            execution_attempts, infra_retries, session_id
    Writes: execution_result, execution_history (delta), execution_attempts,
            infra_retries, messages (infra/invalid/unsupported/not_evaluated)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from langchain_core.messages import AIMessage

from app.graph.constants import MAX_INFRA_RETRIES
from app.schemas.execution import ErrorKind, ExecutionResult, ExecutionStatus
from app.services.execution_service import get_execution_service

logger = logging.getLogger(__name__)

# Statuses where we add a candidate-facing message
_USER_MESSAGE_STATUSES = {
    ExecutionStatus.infra_error,
    ExecutionStatus.unsupported_language,
    ExecutionStatus.invalid_submission,
}

# Statuses that count as a scored attempt toward MAX_EXECUTION_ATTEMPTS
_EVALUATED_STATUSES = {
    ExecutionStatus.passed,
    ExecutionStatus.failed,
    ExecutionStatus.compile_error,
    ExecutionStatus.runtime_error,
    ExecutionStatus.time_limit_exceeded,
    ExecutionStatus.memory_limit_exceeded,
}

_NOT_EVALUATED_MESSAGE = (
    "[Executor] ⚠️ We couldn't run your code due to repeated infrastructure "
    "issues. This problem will not be scored and you will not be penalised. "
    "Let's move on."
)


def executor_node(state: dict) -> dict:
    """Execute the candidate's code submission and return a state delta.

    Calls ``ExecutionService`` and never raises — a last-resort
    ``try/except`` catches any leak and returns an ``infra_error``.
    """
    code: str = state.get("code_submission") or ""
    language: str = state.get("code_language") or "python"
    problem_id: str = state.get("current_problem_id") or ""
    attempts: int = state.get("execution_attempts") or 0
    infra_retries: int = state.get("infra_retries") or 0
    session_id: str = state.get("session_id") or ""

    try:
        result = get_execution_service().run(
            source=code,
            language=language,
            problem_id=problem_id,
            attempt=attempts + 1,
            trace_id=session_id,
        )
    except Exception:
        logger.error(
            "event=executor_node_crash problem_id=%s traceback=%s",
            problem_id, traceback.format_exc(),
        )
        result = ExecutionResult(
            problem_id=problem_id,
            language=language,
            attempt=attempts + 1,
            status=ExecutionStatus.infra_error,
            error="Internal executor error",
            error_kind=ErrorKind.internal,
        )

    # ── C3 accounting ───────────────────────────────────────────────
    # Evaluated runs consume a candidate attempt; infra failures consume
    # an infra retry; unsupported/invalid consume neither.
    if result.status == ExecutionStatus.infra_error:
        infra_retries += 1
        if infra_retries > MAX_INFRA_RETRIES:
            # D2: the third infra error is not the candidate's fault.
            result = result.model_copy(update={
                "status": ExecutionStatus.not_evaluated,
                "error": "Sandbox unavailable; problem was not evaluated",
            })
    elif result.status in _EVALUATED_STATUSES:
        attempts += 1

    result_dict: dict[str, Any] = result.model_dump(mode="json")

    delta: dict[str, Any] = {
        "execution_result": result_dict,
        "execution_history": [result_dict],
        "execution_attempts": attempts,
        "infra_retries": infra_retries,
    }

    # For infra/unsupported/invalid statuses, add a candidate-facing message
    if result.status in _USER_MESSAGE_STATUSES and result.user_message:
        delta["messages"] = [AIMessage(content=result.user_message)]
    elif result.status == ExecutionStatus.not_evaluated:
        delta["messages"] = [AIMessage(content=_NOT_EVALUATED_MESSAGE)]

    return delta
