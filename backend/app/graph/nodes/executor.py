# filepath: backend/app/graph/nodes/executor.py
"""
PlaceMate — Executor Node (Phase 3)
=====================================

Thin graph node that delegates to ``ExecutionService``.

This node knows LangGraph state fields only.  It must NEVER import
``app.tools``, ``httpx``, or know about Judge0.

DATA FLOW:
    Reads:  code_submission, code_language, current_problem_id,
            execution_attempts, session_id
    Writes: execution_result, execution_history (delta), execution_attempts,
            messages (only for infra/invalid/unsupported statuses)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from langchain_core.messages import AIMessage

from app.schemas.execution import ErrorKind, ExecutionResult, ExecutionStatus
from app.services.execution_service import get_execution_service

logger = logging.getLogger(__name__)

# Statuses where we add a candidate-facing message
_USER_MESSAGE_STATUSES = {
    ExecutionStatus.infra_error,
    ExecutionStatus.unsupported_language,
    ExecutionStatus.invalid_submission,
}


def executor_node(state: dict) -> dict:
    """Execute the candidate's code submission and return a state delta.

    Calls ``ExecutionService`` and never raises — a last-resort
    ``try/except`` catches any leak and returns an ``infra_error``.
    """
    code: str = state.get("code_submission", "")
    language: str = state.get("code_language", "python")
    problem_id: str = state.get("current_problem_id", "")
    attempts: int = state.get("execution_attempts", 0)
    session_id: str = state.get("session_id", "")

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

    result_dict: dict[str, Any] = result.model_dump(mode="json")

    delta: dict[str, Any] = {
        "execution_result": result_dict,
        "execution_history": [result_dict],
        "execution_attempts": attempts + 1,
    }

    # For infra/unsupported/invalid statuses, add a candidate-facing message
    if result.status in _USER_MESSAGE_STATUSES and result.user_message:
        delta["messages"] = [AIMessage(content=result.user_message)]

    return delta
