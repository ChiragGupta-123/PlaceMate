# filepath: backend/app/rubrics/sde.py
"""Default SDE hiring rubric (weights sum to 100)."""

RUBRIC_CATEGORIES: dict[str, float] = {
    "problem_solving": 20.0,
    "technical_knowledge": 20.0,
    "code_correctness": 20.0,
    "complexity_analysis": 15.0,
    "communication": 10.0,
    "debugging": 10.0,
    "adaptability": 5.0,
}
