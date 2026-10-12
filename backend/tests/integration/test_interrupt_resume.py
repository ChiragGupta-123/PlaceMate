# filepath: backend/tests/integration/test_interrupt_resume.py
"""
Interrupt / resume / durability tests (real SQLite checkpointer, tmp file).
"""

from __future__ import annotations

import uuid

from langgraph.types import Command

from app.core.enums import InterviewPhase
from app.graph.state import make_initial_state


def _config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_missing_thread_has_no_state(make_graph):
    snapshot = make_graph().get_state(_config())
    assert not snapshot.values


def test_graph_pauses_at_human_input(make_graph, initial_state):
    graph, config = make_graph(), _config()
    graph.invoke(initial_state, config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_input",)
    assert len(snapshot.values["messages"]) == 1          # just the intro question


def test_resume_in_a_fresh_graph_instance(make_graph, initial_state):
    """Simulates a server restart: new compiled graph, same DB file."""
    config = _config()
    make_graph().invoke(initial_state, config)

    restarted = make_graph()                               # "new process"
    before = restarted.get_state(config).values
    restarted.invoke(Command(resume={"answer": "hello"}), config)
    after = restarted.get_state(config).values

    assert len(after["messages"]) == len(before["messages"]) + 2   # +human +AI
    assert after["total_turns"] == before["total_turns"] + 1


def test_completed_interview_ignores_late_resume(make_graph):
    graph, config = make_graph(), _config()
    state = make_initial_state(
        session_id="done", candidate_name="c", target_company="x", target_role="y",
        starting_phase=InterviewPhase.ASSESSMENT.value,
    )
    graph.invoke(state, config)
    graph.invoke(Command(resume={"answer": "go to assessment"}), config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ()                              # reached END
    assert snapshot.values["final_verdict"]

    before = snapshot.values
    graph.invoke(Command(resume={"answer": "too late"}), config)
    assert graph.get_state(config).values == before


def test_threads_are_isolated(make_graph):
    graph = make_graph()
    c1, c2 = _config(), _config()
    graph.invoke(make_initial_state(session_id="1", candidate_name="a", target_company="Company 1", target_role="r"), c1)
    graph.invoke(make_initial_state(session_id="2", candidate_name="b", target_company="Company 2", target_role="r"), c2)
    assert graph.get_state(c1).values["target_company"] == "Company 1"
    assert graph.get_state(c2).values["target_company"] == "Company 2"
