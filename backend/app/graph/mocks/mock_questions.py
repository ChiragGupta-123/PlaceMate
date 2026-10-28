# filepath: backend/app/graph/mocks/mock_questions.py
"""TEMPORARY Phase-1 question bank. Delete in Phase 4."""
from app.core.enums import InterviewPhase, QuestionType

MOCK_QUESTIONS: dict[InterviewPhase, list[dict]] = {
    InterviewPhase.INTRO: [
        {
            "question": (
                "Welcome to your {company} {role} interview! "
                "Tell me about yourself and why you're interested in this role."
            ),
            "type": QuestionType.BEHAVIORAL,
        },
    ],
    InterviewPhase.TECHNICAL: [
        {
            "question": "Explain the difference between a process and a thread. When would you use each?",
            "type": QuestionType.TECHNICAL,
        },
        {
            "question": "What is the time complexity of common operations on a hash table? What causes worst-case behavior?",
            "type": QuestionType.TECHNICAL,
        },
        {
            "question": "Explain how a database index works. What are the trade-offs of adding an index?",
            "type": QuestionType.TECHNICAL,
        },
        {
            "question": "What is the CAP theorem? Give a real-world example of a system that sacrifices one of the three properties.",
            "type": QuestionType.TECHNICAL,
        },
    ],
    InterviewPhase.CODING: [
        {
            "question": (
                "Given an array of integers and a target sum, find two numbers "
                "that add up to the target. Return their indices.\n\n"
                "Example: nums = [2, 7, 11, 15], target = 9  →  [0, 1]\n\n"
                "Please write your solution and submit your code."
            ),
            "type": QuestionType.CODING,
            "problem_id": "two_sum",
        },
        {
            "question": (
                "Given a string containing just the characters '(', ')', '{', '}', "
                "'[' and ']', determine if the input string is valid.\n\n"
                "A string is valid if:\n"
                "- Open brackets are closed by the same type.\n"
                "- Open brackets are closed in the correct order.\n\n"
                "Please write your solution and submit your code."
            ),
            "type": QuestionType.CODING,
            "problem_id": "valid_parentheses",
        },
    ],
}
