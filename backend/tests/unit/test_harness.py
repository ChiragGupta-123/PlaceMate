# filepath: backend/tests/unit/test_harness.py
"""
Harness Unit & Regression Tests (Step 3.2: B1, B2, B8, B9, C4, D1).

Guards the harness script structure using offline AST parsing only.
Never runs candidate code locally.
"""

from __future__ import annotations

import ast
import json

from app.schemas.problem import ProblemSpec
from app.services.harness import build_harness


def _sample_problem_dict() -> dict:
    return {
        "id": "two_sum",
        "title": "Two Sum",
        "function_name": "twoSum",
        "comparator": "exact",
        "public_tests": [
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1], "timeout_ms": 1000}
        ],
        "hidden_tests": [
            {"args": [[3, 3], 6], "expected": [0, 1], "timeout_ms": 1000}
        ],
        "perf_test": {
            "args": [{"$range": [0, 100000]}, 199997],
            "expected": [99998, 99999],
            "timeout_ms": 3000,
        },
    }


def test_harness_ast_structure_and_json_encoding():
    """
    D1 & B1: Harness must be valid Python 3.8 AST and decode _ALL_TESTS to a list of dicts.
    Fails on B1 because double json.dumps causes _ALL_TESTS to evaluate to a str, not list.
    """
    spec = ProblemSpec.model_validate(_sample_problem_dict())
    script = build_harness(source="def twoSum(nums, target): return [0, 1]", problem=spec, nonce="test1234")

    # 1. Must parse with Python 3.8 syntax rules
    tree = ast.parse(script, feature_version=(3, 8))

    # 2. Locate `_ALL_TESTS = json.loads(...)`
    all_tests_node = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "_ALL_TESTS":
                    all_tests_node = stmt.value
                    break

    assert all_tests_node is not None, "Could not find _ALL_TESTS assignment in harness AST"
    assert isinstance(all_tests_node, ast.Call), "_ALL_TESTS must be assigned from a function call"
    assert isinstance(all_tests_node.func, ast.Attribute) and all_tests_node.func.attr == "loads"

    # Evaluate the string literal passed to json.loads(...)
    raw_arg = ast.literal_eval(all_tests_node.args[0])
    parsed_tests = json.loads(raw_arg)

    # B1 regression check: must be list[dict], NOT str
    assert isinstance(parsed_tests, list), f"_ALL_TESTS decoded to {type(parsed_tests).__name__}, expected list (B1 bug)"
    assert len(parsed_tests) == 3

    for item in parsed_tests:
        assert isinstance(item, dict)
        for key in ("kind", "args", "expected", "timeout_ms"):
            assert key in item, f"Missing key {key} in test item: {item}"


def test_harness_size_compact_dsl():
    """B8: Harness must keep DSL compact and not expand 100k items on host (< 20 KB, not ~694 KB)."""
    spec = ProblemSpec.model_validate(_sample_problem_dict())
    script = build_harness(source="def twoSum(nums, target): return [0, 1]", problem=spec, nonce="test1234")
    size_kb = len(script.encode("utf-8")) / 1024
    assert size_kb < 20.0, f"Harness script is {size_kb:.1f} KB (B8 bug: DSL expanded on host)"


def test_harness_per_test_timer_and_preamble():
    """B2, B9, C4: Harness must include per-test signal timer with TestTimeoutError and inject preamble into _ns."""
    spec = ProblemSpec.model_validate(_sample_problem_dict())
    script = build_harness(source="def twoSum(nums, target): return [0, 1]", problem=spec, nonce="test1234")

    # C4: Custom timeout exception class derived from BaseException
    assert "class TestTimeoutError(BaseException):" in script, "Missing TestTimeoutError(BaseException) (C4)"
    assert "signal.setitimer" in script, "Missing signal.setitimer (C4/B2)"
    assert "hasattr(signal, 'setitimer')" in script or 'hasattr(signal, "setitimer")' in script, "Missing hasattr(signal, 'setitimer') guard (C4)"
    assert "setitimer(signal.ITIMER_REAL, 0)" in script or "setitimer(getattr(signal, 'ITIMER_REAL', 0), 0)" in script, "Missing timer reset in finally (C4)"

    # B9: exec(_compiled, _ns) must have populated preamble
    assert "_ns = copy.copy(" in script or "_ns = dict(" in script or "_ns = {" in script and "sys" in script
