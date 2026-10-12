# filepath: backend/tests/unit/test_state.py
"""The state schema and its initial-state factory must never drift apart."""

from __future__ import annotations

import typing

from app.graph.state import InterviewState, make_initial_state


def test_factory_covers_every_state_key():
    schema_keys = set(typing.get_type_hints(InterviewState, include_extras=True))
    factory_keys = set(
        make_initial_state(
            session_id="s", candidate_name="c", target_company="x", target_role="y"
        )
    )
    assert factory_keys == schema_keys


def test_factory_does_not_share_mutable_defaults():
    a = make_initial_state(session_id="a", candidate_name="c", target_company="x", target_role="y")
    b = make_initial_state(session_id="b", candidate_name="c", target_company="x", target_role="y")
    a["messages"].append("x")
    a["execution_history"].append({"status": "pass"})
    assert b["messages"] == [] and b["execution_history"] == []
