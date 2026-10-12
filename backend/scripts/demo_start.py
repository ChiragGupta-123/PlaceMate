# filepath: backend/scripts/demo_start.py
"""
Phase 2 persistence demo, part 1: start an interview and pause.

Run from backend/ (as a module, so `app` is importable):
    python -m scripts.demo_start
    python -m scripts.demo_resume        # in a NEW process
"""

from __future__ import annotations

import uuid
from pathlib import Path

from app.graph.state import make_initial_state
from app.graph.workflow import build_interview_graph

THREAD_FILE = Path(__file__).resolve().parent.parent / "data" / ".last_thread_id"


def main() -> None:
    print("Starting Interview (persistence demo)...")
    graph = build_interview_graph(checkpointer_backend="sqlite")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = make_initial_state(
        session_id="demo_persistence",
        candidate_name="Test Candidate",
        target_company="Persistence Inc",
        target_role="Backend Engineer",
    )

    # Runs START -> INTERVIEWER -> HUMAN_INPUT, then pauses at interrupt().
    graph.invoke(state, config)

    messages = graph.get_state(config).values.get("messages", [])
    if messages:
        print(f"\n[AI]: {messages[-1].content}")

    THREAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THREAD_FILE.write_text(thread_id)
    print(f"\nInterview paused. Thread ID: {thread_id}")
    print("Run `python -m scripts.demo_resume` in a new process to continue.")


if __name__ == "__main__":
    main()
