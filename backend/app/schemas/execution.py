# filepath: backend/app/schemas/execution.py
"""
PlaceMate — Execution Result Schemas
======================================

Pydantic v2 models for code-execution results.  These are the contract
between the ExecutionService and every consumer (executor node, debater,
assessor, future API).

DESIGN RULES
─────────────
• Hidden-test values are redacted at the model level.  No consumer
  can accidentally leak them into state, prompts or the UI.
• stdout/stderr/error are truncated to bounded sizes.
• ``model_dump(mode="json")`` is always safe for LangGraph state.
• Legacy field names (passed, failed, total, status, stdout, stderr,
  error, runtime_ms, memory_kb, failed_case, language) are kept so
  readers written against the Phase-1 mock dict keep working.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Limits for truncation ───────────────────────────────────────────
_MAX_OUTPUT = 2000
_MAX_ERROR = 500
_MAX_REPR = 200


class ExecutionStatus(str, Enum):
    """All possible outcomes of a code-execution run."""
    passed = "passed"
    failed = "failed"
    compile_error = "compile_error"
    runtime_error = "runtime_error"
    time_limit_exceeded = "time_limit_exceeded"
    memory_limit_exceeded = "memory_limit_exceeded"
    infra_error = "infra_error"
    unsupported_language = "unsupported_language"
    invalid_submission = "invalid_submission"


class ErrorKind(str, Enum):
    """Classifies the root cause of an infra_error."""
    network = "network"
    timeout = "timeout"
    quota = "quota"
    auth = "auth"
    rate_limited = "rate_limited"
    provider_internal = "provider_internal"
    bad_response = "bad_response"
    config = "config"
    internal = "internal"


class TestOutcomeKind(str, Enum):
    public = "public"
    hidden = "hidden"
    perf = "perf"


class TestOutcomeResult(str, Enum):
    passed = "pass"
    wrong_answer = "wrong_answer"
    exception = "exception"
    timeout = "timeout"
    not_run = "not_run"


def _trunc(v: str | None, limit: int) -> str | None:
    if v is None:
        return None
    if len(v) > limit:
        return v[:limit - 3] + "..."
    return v


class TestOutcome(BaseModel):
    """Result of a single test case.

    Hidden and perf test values are redacted at the model level:
    input_repr, expected_repr and actual_repr are forced to None,
    and error keeps only the exception class name.
    """
    model_config = {"extra": "ignore"}

    index: int
    kind: TestOutcomeKind
    passed: bool = False
    outcome: TestOutcomeResult = TestOutcomeResult.not_run
    time_ms: float | None = None
    input_repr: str | None = None
    expected_repr: str | None = None
    actual_repr: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _redact_hidden(self) -> TestOutcome:
        """Redact values for hidden and perf tests so they never leak."""
        if self.kind in (TestOutcomeKind.hidden, TestOutcomeKind.perf):
            self.input_repr = None
            self.expected_repr = None
            self.actual_repr = None
            if self.error:
                # Keep only the exception class name
                first_line = self.error.split("\n")[0].strip()
                if ":" in first_line:
                    self.error = first_line.split(":")[0].strip()
                else:
                    self.error = first_line
        # Truncate repr fields
        self.input_repr = _trunc(self.input_repr, _MAX_REPR)
        self.expected_repr = _trunc(self.expected_repr, _MAX_REPR)
        self.actual_repr = _trunc(self.actual_repr, _MAX_REPR)
        self.error = _trunc(self.error, _MAX_ERROR)
        return self


class ExecutionResult(BaseModel):
    """Complete result of running candidate code against a problem.

    Legacy field names (passed, failed, total, status, stdout, stderr,
    error, runtime_ms, memory_kb, failed_case, language) are kept so
    old readers keep working.
    """
    model_config = {"extra": "ignore"}

    # ── Identity ────────────────────────────────────────────────────
    problem_id: str = ""
    language: str = "python"
    attempt: int = 1

    # ── Status ──────────────────────────────────────────────────────
    status: ExecutionStatus = ExecutionStatus.infra_error
    all_correct: bool = False
    perf_ok: bool | None = None

    # ── Counts (over public and hidden tests only) ──────────────────
    passed: int = 0
    failed: int = 0
    total: int = 0
    tests: list[TestOutcome] = Field(default_factory=list)

    # ── Output and errors ───────────────────────────────────────────
    failed_case: str | None = None
    stdout: str = ""
    stderr: str = ""
    compile_error: str | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None

    # ── Metrics ─────────────────────────────────────────────────────
    runtime_ms: float = 0.0
    memory_kb: float = 0.0
    partial: bool = False
    provider: str = ""
    requests_made: int = 0
    latency_ms: float = 0.0

    @field_validator("stdout", "stderr", mode="after")
    @classmethod
    def _truncate_output(cls, v: str) -> str:
        return _trunc(v, _MAX_OUTPUT) or ""

    @field_validator("compile_error", mode="after")
    @classmethod
    def _truncate_compile_error(cls, v: str | None) -> str | None:
        return _trunc(v, _MAX_OUTPUT)

    @field_validator("error", mode="after")
    @classmethod
    def _truncate_error(cls, v: str | None) -> str | None:
        return _trunc(v, _MAX_ERROR)

    @property
    def user_message(self) -> str:
        """Candidate-safe message for non-candidate-fault statuses."""
        if self.status == ExecutionStatus.infra_error:
            return (
                "[Executor] ⚠️ We couldn't run your code right now due to "
                "a temporary infrastructure issue. Your submission was not "
                "scored. Please try again."
            )
        if self.status == ExecutionStatus.unsupported_language:
            return (
                f"[Executor] ⚠️ Language '{self.language}' is not supported. "
                "Please resubmit in Python."
            )
        if self.status == ExecutionStatus.invalid_submission:
            msg = self.error or "Invalid submission"
            return f"[Executor] ⚠️ {msg}. Please check and resubmit."
        return ""
