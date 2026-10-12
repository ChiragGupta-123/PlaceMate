# filepath: backend/tests/conftest.py
"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.graph.state import make_initial_state
from app.graph.workflow import build_interview_graph


@pytest.fixture
def initial_state():
    """A fresh, fully-populated interview state."""
    return make_initial_state(
        session_id="test",
        candidate_name="Test Candidate",
        target_company="Amazon",
        target_role="SDE",
    )


@pytest.fixture
def sqlite_path(tmp_path):
    """An isolated checkpoint DB so tests never touch data/checkpoints.sqlite."""
    return tmp_path / "checkpoints.sqlite"


@pytest.fixture
def make_graph(sqlite_path):
    """Factory: every call returns a NEW compiled graph on the SAME sqlite file
    (simulates restarting the server)."""

    def _make():
        return build_interview_graph("sqlite", sqlite_path=sqlite_path)

    return _make
