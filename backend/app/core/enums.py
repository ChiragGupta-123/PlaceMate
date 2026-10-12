# filepath: backend/app/core/enums.py
"""Shared enums used by graph, API schemas and DB models."""
from enum import Enum


class InterviewPhase(str, Enum):
    """
    The interview progresses through explicit phases.
    The router inspects the current phase to decide where to go next.

    Flow:
        INTRO → TECHNICAL → CODING → FOLLOW_UP → ASSESSMENT → COMPLETE

    Not every interview hits every phase — Quick Fire mode may skip
    CODING entirely and go straight from TECHNICAL to ASSESSMENT.
    """
    INTRO = "intro"
    TECHNICAL = "technical"
    CODING = "coding"
    EXECUTION = "execution"           # Sub-phase: code is being executed
    FOLLOW_UP = "follow_up"           # Sub-phase: debater follow-up
    ASSESSMENT = "assessment"
    COMPLETE = "complete"


class InterviewMode(str, Enum):
    """
    Interview duration/depth modes.  These are set once at the start
    and determine turn budgets and phase structure.

    The system does NOT use wall-clock hard stops.  Instead, each mode
    defines how many turns/questions the candidate gets per phase.
    """
    QUICK_FIRE = "quick_fire"         # 30 min — 1 easy/medium problem
    STANDARD = "standard"             # 45 min — 2 core + 1 DSA + debate
    DEEP_DIVE = "deep_dive"           # 60 min — 3 core + 1 hard + deep debate


class QuestionType(str, Enum):
    """
    Categorizes the kind of question being asked.
    Used by the interviewer to decide what to generate
    and by the router to decide where to go after the answer.
    """
    TECHNICAL = "technical"           # Core CS / domain knowledge
    CODING = "coding"                 # Write code to solve a problem
    SYSTEM_DESIGN = "system_design"   # Architecture / design question
    BEHAVIORAL = "behavioral"         # Soft skills (less common)
    FOLLOW_UP = "follow_up"           # Adversarial follow-up from debater


class Difficulty(str, Enum):
    """Difficulty level for questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
