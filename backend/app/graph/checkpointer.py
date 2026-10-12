# filepath: backend/app/graph/checkpointer.py
"""
PlaceMate — Checkpointer Factory
==================================

WHY THIS FILE EXISTS:
    LangGraph needs a "checkpointer" to persist state between
    interrupt/resume cycles.  Without a checkpointer, when the graph
    pauses (interrupt) and you later resume it, ALL state is lost.

    This factory creates the appropriate checkpointer based on
    the environment:
      - Phase 1-2: MemorySaver (in-memory, no external deps)
      - Phase 2+:  SqliteSaver (durable, file-based)
      - Production: PostgresSaver (scalable, shared)

    By abstracting the checkpointer behind a factory, the rest of
    the codebase doesn't care which one is being used.

HOW IT CONNECTS:
    • workflow.py calls get_checkpointer() and passes the result
      to graph.compile(checkpointer=...).
    • The checkpointer is also used when invoking the graph with
      a thread_id — this is how we resume a specific interview.

LANGGRAPH CONCEPT — Checkpointers & Thread IDs:
    When you compile a graph with a checkpointer:

        graph = builder.compile(checkpointer=memory_saver)

    Every invocation MUST include a thread_id in the config:

        config = {"configurable": {"thread_id": "interview-123"}}
        graph.invoke(input, config)

    The checkpointer saves the graph's state (including which node
    was last executed) keyed by thread_id.  When the graph hits an
    interrupt(), it saves state and returns.  When you call
    graph.invoke(Command(resume=...), config) with the SAME thread_id,
    the checkpointer loads the saved state and continues.

    This is how PlaceMate supports:
      - Pause while waiting for candidate response.
      - Resume when the candidate submits an answer.
      - Recover from server crashes mid-interview.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer(backend: str = "memory", db_path: Path | None = None):
    """
    Create and return a checkpointer for the interview graph.

    Phase 1: Returns MemorySaver (in-memory).
    ──────────────────────────────────────────
    MemorySaver stores all state in a Python dict in memory.
    It's perfect for development and testing because:
      - No external dependencies (no database, no files).
      - Instant read/write.
      - State disappears when the process exits.

    Phase 2: Returns SqliteSaver (durable).
    ───────────────────────────────────────
    Uses a local SQLite file to persist graph state. This allows
    an interview to survive a process restart.

    Phase 8 (FastAPI):
    ──────────────────
    We will want to swap SqliteSaver for AsyncSqliteSaver (or AsyncPostgresSaver)
    so the checkpointing doesn't block the async event loop.
    This file is designed so that swap is a one-file change.
    """
    if backend == "sqlite":
        # Resolve path relative to this file, not CWD
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "checkpoints.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # We manage the connection here. check_same_thread=False is needed
        # if the graph is invoked from different threads (e.g. in tests/FastAPI).
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        return SqliteSaver(conn)
    else:
        return MemorySaver()
