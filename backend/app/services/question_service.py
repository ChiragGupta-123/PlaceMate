# filepath: backend/app/services/question_service.py
"""
PlaceMate — Question Service (Step 4.3)
=======================================

Produces the next interview question.  This is the ONLY place that talks
to the interviewer LLM.

WHAT IS DETERMINISTIC AND WHAT IS NOT
─────────────────────────────────────
The graph still decides the phase, the turn, the question TYPE and the
turn budget.  The LLM only writes the question text and — for coding
questions — picks a ``problem_id`` from the curated ``ProblemBank``.
Everything the model returns is validated by the Step 4.2 schema, so it
can never smuggle routing fields into the state.

FALLBACK CONTRACT
─────────────────
The node builds the mock question for the same turn and passes it in
``QuestionRequest.fallback``.  This service NEVER imports ``app.graph``:
mock mode, exhausted budgets and every runtime LLM failure simply return
that node-supplied fallback.

RESILIENCE
──────────
• bad JSON / schema-invalid output → 2 attempts total
• timeout / unavailable            → 2 attempts total (1 retry)
• quota / auth / configuration     → 1 attempt (no retry)
• invalid or already-used coding id → treated as schema-invalid
• MAX_LLM_CALLS_PER_SESSION caps the whole session; every request,
  including retries, counts toward it.

Configuration errors are different: if ``question_source == "llm"`` the
factory builds the provider eagerly and lets ``LLMConfigurationError``
propagate (fail fast at startup).  Only RUNTIME LLM errors fall back.

IMPORT RULES
────────────
May import ``app.core``, ``app.services`` and ``app.tools.llm_client``.
Must NEVER import ``app.graph``, a concrete LLM adapter or ``httpx``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.core.config import Settings, get_settings
from app.core.enums import Difficulty, InterviewMode, InterviewPhase, QuestionType
from app.services import problem_bank as _default_problem_bank
from app.services.interviewer_prompts import (
    CandidateTurn,
    PromptBuildError,
    QuestionOutputError,
    build_interviewer_prompts,
    parse_question_draft,
)
from app.tools.llm_client import (
    LLMAuthError,
    LLMBadOutput,
    LLMConfigurationError,
    LLMError,
    LLMInterface,
    LLMQuotaExceeded,
    LLMRequest,
    LLMTimeout,
    LLMUnavailable,
)

logger = logging.getLogger(__name__)

# Session-wide cap on LLM requests (retries included).
MAX_LLM_CALLS_PER_SESSION: int = 20

# Errors worth a second attempt.
_RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (
    LLMBadOutput,
    LLMTimeout,
    LLMUnavailable,
)

# Errors where retrying only wastes quota or repeats a permanent failure.
_NON_RETRYABLE_ERRORS: tuple[type[LLMError], ...] = (
    LLMQuotaExceeded,
    LLMAuthError,
    LLMConfigurationError,
)

_MAX_ATTEMPTS: int = 2


# ── Value objects ────────────────────────────────────────────────────

@dataclass(frozen=True)
class FallbackQuestion:
    """The deterministic mock question for the current turn."""

    text: str
    question_type: QuestionType
    problem_id: str | None = None


@dataclass(frozen=True)
class QuestionRequest:
    """Everything the service needs for ONE question."""

    phase: InterviewPhase
    mode: InterviewMode
    company: str
    role: str
    difficulty: Difficulty
    candidate_name: str
    current_turn: int
    expected_question_type: QuestionType
    fallback: FallbackQuestion
    history: Sequence[CandidateTurn] = ()
    covered_topics: Sequence[str] = ()
    interview_summary: str = ""
    used_problem_ids: Sequence[str] = ()
    llm_calls_so_far: int = 0
    retrieved_context: Sequence[str] = ()  # Phase 5 placeholder


@dataclass(frozen=True)
class QuestionResult:
    """The question to ask plus the bookkeeping the node needs."""

    text: str
    question_type: QuestionType
    topic: str
    problem_id: str | None
    source: str  # "llm" | "fallback"
    llm_calls_used: int


# ── Service ──────────────────────────────────────────────────────────

class QuestionService:
    """Generate one interview question, with a deterministic fallback.

    Parameters
    ----------
    llm : LLMInterface | None
        ``None`` in mock mode.
    settings : Settings | None
        Defaults to the cached application settings.
    problem_bank : object | None
        Anything exposing ``all_ids()``.  Defaults to the real bank.
    use_llm : bool | None
        Override the opt-in rule.  By default the LLM path is used when an
        LLM is installed or ``question_source == "llm"``.
    max_llm_calls : int
        Session cap; overridable so tests can exhaust it quickly.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        settings: Settings | None = None,
        problem_bank: Any | None = None,
        *,
        use_llm: bool | None = None,
        max_llm_calls: int = MAX_LLM_CALLS_PER_SESSION,
    ) -> None:
        self._llm = llm
        self._settings = settings or get_settings()
        self._problem_bank = (
            problem_bank if problem_bank is not None else _default_problem_bank
        )
        self._max_llm_calls = max(0, int(max_llm_calls))
        self._use_llm = (
            use_llm
            if use_llm is not None
            else (llm is not None or self._settings.question_source == "llm")
        )

    # ── Public API ──────────────────────────────────────────────────

    def get_question(self, request: QuestionRequest) -> QuestionResult:
        """Return the next question.  Never raises for runtime failures."""
        if not self._use_llm or self._llm is None:
            return self._fallback(request, calls_used=0)

        allowed_ids = self._allowed_problem_ids(request)
        if (
            request.expected_question_type == QuestionType.CODING
            and not allowed_ids
        ):
            logger.info(
                "event=question_no_unused_problem_ids phase=%s used=%d",
                request.phase.value,
                len(request.used_problem_ids),
            )
            return self._fallback(request, calls_used=0)

        if request.llm_calls_so_far >= self._max_llm_calls:
            logger.info(
                "event=question_call_cap_reached used=%d cap=%d",
                request.llm_calls_so_far,
                self._max_llm_calls,
            )
            return self._fallback(request, calls_used=0)

        try:
            bundle = build_interviewer_prompts(
                phase=request.phase,
                mode=request.mode,
                company=request.company,
                role=request.role,
                difficulty=request.difficulty,
                candidate_name=request.candidate_name,
                history=request.history,
                covered_topics=request.covered_topics,
                rolling_summary=request.interview_summary,
                retrieved_context=request.retrieved_context,
                allowed_problem_ids=allowed_ids,
            )
        except PromptBuildError:
            logger.warning(
                "event=question_prompt_build_failed phase=%s",
                request.phase.value,
            )
            return self._fallback(request, calls_used=0)

        calls_used = 0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            if request.llm_calls_so_far + calls_used >= self._max_llm_calls:
                break

            try:
                response = self._llm.generate_structured(
                    self._llm_request(bundle, request, attempt)
                )
            except LLMError as exc:
                calls_used += 1
                logger.warning(
                    "event=question_llm_error phase=%s attempt=%d error=%s",
                    request.phase.value,
                    attempt,
                    type(exc).__name__,
                )
                if not isinstance(exc, _RETRYABLE_ERRORS):
                    break
                continue
            except Exception as exc:  # defensive: adapters must not crash us
                calls_used += 1
                logger.warning(
                    "event=question_llm_crash phase=%s attempt=%d error=%s",
                    request.phase.value,
                    attempt,
                    type(exc).__name__,
                )
                break

            calls_used += 1

            if response.data is None:
                logger.warning(
                    "event=question_unparseable_output phase=%s attempt=%d",
                    request.phase.value,
                    attempt,
                )
                continue

            try:
                draft = parse_question_draft(
                    response.data,
                    allowed_problem_ids=allowed_ids,
                    expected_question_type=request.expected_question_type,
                )
            except QuestionOutputError:
                logger.warning(
                    "event=question_invalid_output phase=%s attempt=%d",
                    request.phase.value,
                    attempt,
                )
                continue

            logger.info(
                "event=question_from_llm phase=%s type=%s attempts=%d",
                request.phase.value,
                draft.question_type.value,
                calls_used,
            )
            return QuestionResult(
                text=draft.question,
                question_type=draft.question_type,
                topic=draft.topic,
                problem_id=draft.problem_id,
                source="llm",
                llm_calls_used=calls_used,
            )

        logger.info(
            "event=question_fallback phase=%s type=%s calls=%d",
            request.phase.value,
            request.expected_question_type.value,
            calls_used,
        )
        return self._fallback(request, calls_used=calls_used)

    # ── Internals ───────────────────────────────────────────────────

    def _allowed_problem_ids(self, request: QuestionRequest) -> list[str]:
        if request.expected_question_type != QuestionType.CODING:
            return []
        used = set(request.used_problem_ids)
        return [
            problem_id
            for problem_id in self._problem_bank.all_ids()
            if problem_id not in used
        ]

    def _llm_request(
        self,
        bundle: Any,
        request: QuestionRequest,
        attempt: int,
    ) -> LLMRequest:
        return LLMRequest(
            system_prompt=bundle.system,
            user_prompt=bundle.user,
            max_output_tokens=self._settings.llm_max_output_tokens,
            timeout_s=self._settings.llm_read_timeout_s,
            tags={
                "phase": request.phase.value,
                "question_type": request.expected_question_type.value,
                "attempt": str(attempt),
            },
        )

    def _fallback(
        self,
        request: QuestionRequest,
        *,
        calls_used: int,
    ) -> QuestionResult:
        fallback = request.fallback
        return QuestionResult(
            text=fallback.text,
            question_type=fallback.question_type,
            topic=fallback.question_type.value,
            problem_id=fallback.problem_id,
            source="fallback",
            llm_calls_used=calls_used,
        )


# ── Singleton / test seam ────────────────────────────────────────────

_cached_service: QuestionService | None = None
_service_override: QuestionService | None = None


def get_question_service() -> QuestionService:
    """Return the cached service, or build one from settings.

    In ``question_source == "llm"`` mode the provider is built eagerly, so
    an invalid configuration raises ``LLMConfigurationError`` here (fail
    fast) instead of silently degrading every question.
    """
    global _cached_service, _service_override
    if _service_override is not None:
        return _service_override
    if _cached_service is None:
        settings = get_settings()
        llm: LLMInterface | None = None
        if settings.question_source == "llm":
            from app.tools import get_llm_provider

            llm = get_llm_provider(settings)
        _cached_service = QuestionService(
            llm=llm,
            settings=settings,
            problem_bank=_default_problem_bank,
        )
    return _cached_service


def set_question_service(service: QuestionService | None) -> None:
    """Test seam: override the singleton returned by ``get_question_service``.

    Pass ``None`` to clear the override (and the cached default) and revert
    to a fresh service built from settings.
    """
    global _service_override, _cached_service
    _service_override = service
    if service is None:
        _cached_service = None


def reset_question_service() -> None:
    """Drop both the override and the cached default (test isolation)."""
    global _service_override, _cached_service
    _service_override = None
    _cached_service = None
