# filepath: backend/app/graph/workflow.py
"""
PlaceMate — Interview Workflow (Graph Builder)
================================================

WHY THIS FILE EXISTS:
    This is where the StateGraph is ASSEMBLED: it takes the nodes, routers,
    state schema and checkpointer and wires them into a compiled graph.
    It contains wiring ONLY — no business logic.

DEPENDENCY RULE (prevents circular imports):
    workflow.py is the top of the app.graph import tree.  It imports
    state, constants, checkpointer, nodes and routers.  NONE of those may
    import workflow.py (nodes/routers must never import the builder).

LANGGRAPH CONCEPT — compile():
    builder.compile(checkpointer=...) validates the graph, attaches the
    checkpointer and returns an executable.  The compiled graph is
    reusable: create it once, invoke it with many thread_ids.

LANGGRAPH CONCEPT — interrupt():
    interrupt() is called INSIDE human_input_node (see nodes/human_input.py).
    The checkpointer saves state, control returns to the caller, and
    Command(resume=...) with the SAME thread_id continues the graph.

NODE NAMES:
    Always pass NodeName.X.value (a plain str) to LangGraph — see the note
    in app/graph/constants.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.checkpointer import get_checkpointer
from app.graph.constants import NodeName
from app.graph.edges.routers import after_executor_router, after_human_input_router
from app.graph.nodes import (
    assessor_node,
    debater_node,
    executor_node,
    human_input_node,
    interviewer_node,
)
from app.graph.state import InterviewState


def build_interview_graph(
    checkpointer_backend: str = "memory",
    sqlite_path: Path | None = None,
) -> Any:
    """
    Build and compile the PlaceMate interview graph.

    Args:
        checkpointer_backend: "memory" (default) or "sqlite".
        sqlite_path: optional DB file for the sqlite backend (tests pass a
            tmp path so they never touch the real data/ directory).

    Graph topology:
    ───────────────
                        START
                          │
                          ▼
                    INTERVIEWER
                          │
                          ▼
                    HUMAN_INPUT  ← interrupt() pauses here
                          │
                          ▼
                  ┌── ROUTER ──┐──────────┐
                  │            │          │
                  ▼            ▼          ▼
             INTERVIEWER   EXECUTOR    ASSESSOR
                  ↑            │          │
                  │        ┌───┴───┐      │
                  │        │       │      ▼
                  │      pass    fail     END
                  │        │       │
                  │        │       ▼
                  │        │    DEBATER
                  │        │       │
                  └────────┴───────┘
    """

    interviewer = NodeName.INTERVIEWER.value
    human_input = NodeName.HUMAN_INPUT.value
    executor = NodeName.EXECUTOR.value
    debater = NodeName.DEBATER.value
    assessor = NodeName.ASSESSOR.value

    # ── 1. Graph builder with our single state schema ───────────────
    builder = StateGraph(InterviewState)

    # ── 2. Nodes ────────────────────────────────────────────────────
    builder.add_node(interviewer, interviewer_node)
    builder.add_node(human_input, human_input_node)
    builder.add_node(executor, executor_node)
    builder.add_node(debater, debater_node)
    builder.add_node(assessor, assessor_node)

    # ── 3. Entry point: START → INTERVIEWER ─────────────────────────
    builder.set_entry_point(interviewer)

    # ── 4. Unconditional edges ──────────────────────────────────────
    builder.add_edge(interviewer, human_input)   # always wait for candidate
    builder.add_edge(debater, interviewer)        # the Debater cycle
    builder.add_edge(assessor, END)               # interview is over

    # ── 5. Conditional edges ────────────────────────────────────────
    builder.add_conditional_edges(
        human_input,
        after_human_input_router,
        {
            "technical": interviewer,    # ask the next question
            "coding": executor,          # run submitted code
            "assessment": assessor,      # wrap up
        },
    )
    builder.add_conditional_edges(
        executor,
        after_executor_router,
        {
            "pass": interviewer,         # correct → next question
            "exhausted": interviewer,    # attempts used up → move on
            "infra_error": interviewer,  # sandbox failure → not candidate fault
            "fail": debater,             # wrong answer / error → challenge
            "optimize": debater,         # correct but slow → complexity challenge
            "probe": debater,            # correct + fast → probe complexity
            "retry_input": human_input,  # invalid/unsupported → let them fix
        },
    )

    # ── 6. Compile with checkpointer ────────────────────────────────
    checkpointer = get_checkpointer(backend=checkpointer_backend, db_path=sqlite_path)
    return builder.compile(checkpointer=checkpointer)
