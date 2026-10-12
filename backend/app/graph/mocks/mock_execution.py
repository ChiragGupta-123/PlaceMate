# filepath: backend/app/graph/mocks/mock_execution.py
"""TEMPORARY fake sandbox. Replaced by tools/ sandbox in Phase 3."""
from __future__ import annotations

from typing import Any


def mock_execute(code: str, language: str) -> dict[str, Any]:
    """
    Simulate code execution and return a structured result.

    This is the contract that the real Judge0 service must also follow.
    By defining the result shape NOW, we ensure the graph, router,
    and debater can all be built against a stable interface.

    Result schema:
    {
        "status": "pass" | "partial" | "fail" | "error",
        "passed": int,
        "failed": int,
        "total": int,
        "stdout": str,
        "stderr": str,
        "error": str | None,
        "runtime_ms": float,
        "memory_kb": float,
        "failed_case": str | None,
        "language": str,
    }
    """

    if not code or not code.strip():
        return {
            "status": "error",
            "passed": 0,
            "failed": 0,
            "total": 5,
            "stdout": "",
            "stderr": "No code submitted",
            "error": "Empty submission",
            "runtime_ms": 0,
            "memory_kb": 0,
            "failed_case": None,
            "language": language,
        }

    # Heuristic: nested loops → partial pass (inefficient solution)
    if "for " in code and code.count("for ") >= 2:
        return {
            "status": "partial",
            "passed": 3,
            "failed": 2,
            "total": 5,
            "stdout": "[2, 7] -> [0, 1]\n",
            "stderr": "",
            "error": None,
            "runtime_ms": 450.0,  # Suspiciously slow
            "memory_kb": 8192,
            "failed_case": "Input: nums=[1..100000], target=199999 — Time Limit Exceeded",
            "language": language,
        }

    # Heuristic: has a function definition → full pass
    if "def " in code or "function " in code:
        return {
            "status": "pass",
            "passed": 5,
            "failed": 0,
            "total": 5,
            "stdout": "[2, 7] -> [0, 1]\n[3, 5] -> [1, 2]\n",
            "stderr": "",
            "error": None,
            "runtime_ms": 12.0,
            "memory_kb": 4096,
            "failed_case": None,
            "language": language,
        }

    # Default: fail
    return {
        "status": "fail",
        "passed": 1,
        "failed": 4,
        "total": 5,
        "stdout": "",
        "stderr": "NameError: name 'twoSum' is not defined",
        "error": "Runtime error",
        "runtime_ms": 5.0,
        "memory_kb": 2048,
        "failed_case": "Input: nums=[2, 7, 11, 15], target=9 — NameError",
        "language": language,
    }
