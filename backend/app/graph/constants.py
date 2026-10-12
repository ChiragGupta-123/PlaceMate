# filepath: backend/app/graph/constants.py
"""
PlaceMate — Graph Constants
============================

Graph-specific constants only: node names and per-mode turn budgets.

The shared domain enums (InterviewPhase, InterviewMode, QuestionType,
Difficulty) live in app/core/enums.py so the API, DB models and schemas
can use them WITHOUT importing anything from app.graph.

Dependency rule: this file may import from app.core only.
"""
from enum import Enum

from app.core.enums import InterviewMode, InterviewPhase


class NodeName(str, Enum):
    """
    Every node in the LangGraph StateGraph gets a unique string name.
    These names are used in:
      - graph.add_node(name, function)
      - graph.add_edge(source, target)
      - conditional routing return values

    We extend `str` so members compare equal to their plain strings, but ALWAYS
    pass `NodeName.X.value` to LangGraph (add_node / add_edge / path maps).
    Why: on Python 3.12+, f"{NodeName.X}" renders as 'NodeName.X', not 'x'.
    LangGraph builds internal channel keys from node names, so passing the
    member itself can persist 'NodeName.X' into checkpoints and break
    resuming a saved interview under a different Python version.
    """
    INTERVIEWER = "interviewer"
    EXECUTOR = "executor"
    DEBATER = "debater"
    ASSESSOR = "assessor"
    HUMAN_INPUT = "human_input"


# ----------------------------------------------------------------
# TURN BUDGETS PER MODE
# ----------------------------------------------------------------
# These dictionaries define the maximum number of turns (questions)
# allowed in each phase, per interview mode.  The router checks
# current_turn against these to decide when to advance phases.
# ----------------------------------------------------------------

MODE_TURN_BUDGETS: dict[InterviewMode, dict[InterviewPhase, int]] = {
    InterviewMode.QUICK_FIRE: {
        InterviewPhase.INTRO: 1,
        InterviewPhase.TECHNICAL: 2,
        InterviewPhase.CODING: 1,
        InterviewPhase.FOLLOW_UP: 1,
    },
    InterviewMode.STANDARD: {
        InterviewPhase.INTRO: 1,
        InterviewPhase.TECHNICAL: 3,
        InterviewPhase.CODING: 2,
        InterviewPhase.FOLLOW_UP: 2,
    },
    InterviewMode.DEEP_DIVE: {
        InterviewPhase.INTRO: 1,
        InterviewPhase.TECHNICAL: 4,
        InterviewPhase.CODING: 3,
        InterviewPhase.FOLLOW_UP: 3,
    },
}

# Default max turns for any phase not listed above
DEFAULT_MAX_TURNS_PER_PHASE: int = 2
