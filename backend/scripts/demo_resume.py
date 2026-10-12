# filepath: backend/scripts/demo_resume.py
"""
Phase 2 persistence demo, part 2: resume a paused interview in a NEW process.

Run from backend/:
    python -m scripts.demo_resume [thread_id]
"""

from __future__ import annotations

import sys

from langgraph.types import Command

from app.graph.workflow import build_interview_graph
from scripts.demo_start import THREAD_FILE


def main() -> None:
    if len(sys.argv) > 1:
        thread_id = sys.argv[1]
    elif THREAD_FILE.exists():
        thread_id = THREAD_FILE.read_text().strip()
    else:
        print("Please provide a thread_id (or run scripts.demo_start first).")
        sys.exit(1)

    print(f"Resuming Interview for Thread ID: {thread_id}")
    graph = build_interview_graph(checkpointer_backend="sqlite")
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        print("Error: Thread ID not found or has no state.")
        sys.exit(1)

    before = snapshot.values
    messages_before = len(before.get("messages", []))
    turns_before = before.get("total_turns", 0)
    print(f"State before resume: {messages_before} messages, total_turns {turns_before}")

    answer = {"answer": "This is a cross-process persistence test response."}
    print(f"\n[Human]: {answer['answer']}")
    graph.invoke(Command(resume=answer), config)

    after = graph.get_state(config).values
    messages_after = len(after.get("messages", []))
    turns_after = after.get("total_turns", 0)
    print(f"State after resume:  {messages_after} messages, total_turns {turns_after}")

    # Resuming must add exactly 1 human + 1 AI message and advance one turn.
    if messages_after != messages_before + 2 or turns_after != turns_before + 1:
        print("ERROR: unexpected state change on resume (double side effect?)")
        sys.exit(1)

    print("Success! Interview resumed in a new process.")


if __name__ == "__main__":
    main()
