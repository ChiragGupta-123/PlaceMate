# filepath: backend/tests/unit/test_state.py
"""The state schema and its initial-state factory must never drift apart."""

from __future__ import annotations

from pathlib import Path
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


def test_no_debug_agent_log_blocks():
    """Regression test for B10: production code must never contain debug agent log blocks."""
    app_dir = Path(__file__).resolve().parents[2] / "app"
    offending: list[str] = []
    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if "region agent log" in content:
            offending.append(str(py_file.relative_to(app_dir)))
    assert not offending, f"Found debug agent log blocks in: {offending}"


def test_gitignore_is_utf8_and_ignores_junk():
    """Regression test for B11: .gitignore must be valid UTF-8 without BOM and contain all required rules."""
    repo_root = Path(__file__).resolve().parents[3]
    gitignore_path = repo_root / ".gitignore"
    assert gitignore_path.is_file(), f".gitignore not found at {gitignore_path}"

    raw = gitignore_path.read_bytes()
    assert not raw.startswith(b"\xff\xfe"), ".gitignore has UTF-16 LE BOM"
    assert not raw.startswith(b"\xfe\xff"), ".gitignore has UTF-16 BE BOM"
    assert not raw.startswith(b"\xef\xbb\xbf"), ".gitignore has UTF-8 BOM"

    text = raw.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]

    required_patterns = [
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        ".env",
        ".venv",
        "env/",
        "venv/",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "*.log",
        ".last_thread_id",
    ]
    for pattern in required_patterns:
        assert pattern in lines, f"Missing required pattern in .gitignore: {pattern}"
