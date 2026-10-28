# filepath: backend/app/schemas/problem.py
"""
PlaceMate — Problem Specification Schemas
===========================================

Defines ``ProblemSpec`` and ``TestSpec`` — the contract that describes
a coding problem and its test cases.

The problem bank (``services/problem_bank.py``) validates JSON files
against these models.  The harness builder reads ``ProblemSpec`` to
generate the test-runner script.

DESIGN
──────
• Args and expected values must be JSON-serializable.
• A tiny declarative DSL is supported for large inputs in perf tests:
    {"$range": [start, stop]}  →  list(range(start, stop))
    {"$repeat": [value, count]} →  [value] * count
  This avoids multi-MB JSON files for performance test cases.
• DSL is intentionally NOT expanded on the host.  The harness script
  carries the compact DSL literals and expands them inside the sandbox.
  This keeps the script small (< 20 KB) and avoids ~694 KB payloads
  that would breach Judge0's stdin size limit (B8 fix).
• The ``perf_test`` is optional; not every problem needs one.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Comparator(str, Enum):
    """How to compare the candidate's return value to the expected value."""
    exact = "exact"
    unordered_list = "unordered_list"
    float_tolerance = "float_tolerance"


def expand_dsl(value: Any) -> Any:
    """Recursively expand the tiny DSL used in perf-test args.

    Supported:
        {"$range": [start, stop]}   → list(range(start, stop))
        {"$repeat": [value, count]} → [value] * count

    This function is intentionally NOT called during Pydantic validation
    so that the harness can embed raw DSL and expand it inside the sandbox.
    It is kept here for host-side unit tests and documentation only.
    """
    if isinstance(value, dict):
        if "$range" in value:
            args = value["$range"]
            return list(range(args[0], args[1]))
        if "$repeat" in value:
            args = value["$repeat"]
            return [args[0]] * args[1]
        return {k: expand_dsl(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_dsl(item) for item in value]
    return value


class TestSpec(BaseModel):
    """A single test case for a coding problem."""
    model_config = {"extra": "ignore"}

    args: list[Any]
    expected: Any
    timeout_ms: int = 5000


class ProblemSpec(BaseModel):
    """Complete specification for a coding problem."""
    model_config = {"extra": "ignore"}

    id: str
    title: str
    statement: str = ""
    function_name: str
    comparator: Comparator = Comparator.exact
    public_tests: list[TestSpec] = Field(min_length=1)
    hidden_tests: list[TestSpec] = Field(default_factory=list)
    perf_test: TestSpec | None = None
    perf_budget_ms: int | None = None
    expected_time_complexity: str | None = None
    expected_space_complexity: str | None = None
    memory_limit_kb: int = 256_000
    difficulty: str = "medium"
