# filepath: backend/app/services/harness.py
"""
PlaceMate — Test Harness Builder and Parser
=============================================

``build_harness(source, problem, nonce)`` produces a single Python script
that the sandbox runs.  The script:

1. Compiles the candidate's code.
2. Locates the target function.
3. Runs every test case (public → hidden → perf), printing one
   structured result line per test.
4. Prints a ``done`` sentinel when finished.

``parse_harness_output(stdout, nonce)`` extracts the result lines from
sandbox stdout, tolerating garbage, forged markers and truncation.

NOTE: The parser is a **correctness** defense, not a security boundary.
The sandbox is the security boundary.

NEVER run the generated harness locally.  It is sent only to the sandbox.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.schemas.problem import Comparator, ProblemSpec

logger = logging.getLogger(__name__)

_MARKER_PREFIX = "@@PM:"
_MARKER_SUFFIX = "@@"


def build_harness(source: str, problem: ProblemSpec, nonce: str) -> str:
    """Build the test-runner script.  Python 3.8 compatible."""
    source_b64 = base64.b64encode(source.encode("utf-8")).decode("ascii")

    # Build the layout: list of {kind, timeout_ms} for every test
    layout: List[Dict[str, Any]] = []
    for t in problem.public_tests:
        layout.append({"kind": "public", "timeout_ms": t.timeout_ms})
    for t in problem.hidden_tests:
        layout.append({"kind": "hidden", "timeout_ms": t.timeout_ms})
    if problem.perf_test:
        layout.append({"kind": "perf", "timeout_ms": problem.perf_test.timeout_ms})

    # Build all test cases as a JSON list of {args, expected, kind}
    all_tests: List[Dict[str, Any]] = []
    for t in problem.public_tests:
        all_tests.append({
            "args": t.args, "expected": t.expected, "kind": "public",
            "timeout_ms": t.timeout_ms,
        })
    for t in problem.hidden_tests:
        all_tests.append({
            "args": t.args, "expected": t.expected, "kind": "hidden",
            "timeout_ms": t.timeout_ms,
        })
    if problem.perf_test:
        all_tests.append({
            "args": problem.perf_test.args,
            "expected": problem.perf_test.expected,
            "kind": "perf",
            "timeout_ms": problem.perf_test.timeout_ms,
        })

    comparator = problem.comparator.value
    fn_name = problem.function_name
    marker = f"{_MARKER_PREFIX}{nonce}{_MARKER_SUFFIX}"

    # The harness script — runs inside the sandbox on Python 3.8
    lines = [
        "from __future__ import annotations",
        f"PM_NONCE = {nonce!r}",
        f"PM_LAYOUT = {json.dumps(layout)!r}",
        f"PM_SOURCE_B64 = {source_b64!r}",
        "",
        "import sys, json, copy, time, base64, io, traceback",
        "from typing import List, Optional, Dict, Tuple, Set, Any",
        "import collections, heapq, bisect, math, itertools, functools",
        "",
        f"_MARKER = {marker!r}",
        f"_FN_NAME = {fn_name!r}",
        f"_COMPARATOR = {comparator!r}",
        f"_ALL_TESTS = json.loads({json.dumps(json.dumps(all_tests))!r})",
        "",
        "def _emit(data):",
        "    print(_MARKER + json.dumps(data, default=str), flush=True)",
        "",
        "def _compare(actual, expected, comp):",
        "    if comp == 'exact':",
        "        return actual == expected",
        "    elif comp == 'unordered_list':",
        "        if not isinstance(actual, list) or not isinstance(expected, list):",
        "            return actual == expected",
        "        return sorted(actual) == sorted(expected)",
        "    elif comp == 'float_tolerance':",
        "        try:",
        "            return abs(float(actual) - float(expected)) < 1e-6",
        "        except (TypeError, ValueError):",
        "            return actual == expected",
        "    return actual == expected",
        "",
        "# --- Compile candidate source ---",
        "_source_code = base64.b64decode(PM_SOURCE_B64).decode('utf-8', errors='replace')",
        "try:",
        "    _compiled = compile(_source_code, '<candidate>', 'exec')",
        "except SyntaxError as _se:",
        "    _emit({'type': 'compile_error', 'message': str(_se), 'line': getattr(_se, 'lineno', None)})",
        "    sys.exit(0)",
        "",
        "# --- Execute to define functions ---",
        "_ns = {}",
        "exec(_compiled, _ns)",
        "",
        "# --- Locate the function ---",
        "_fn = _ns.get(_FN_NAME)",
        "if _fn is None:",
        "    # Check for Solution class",
        "    _sol_cls = _ns.get('Solution')",
        "    if _sol_cls is not None:",
        "        try:",
        "            _sol = _sol_cls()",
        "            _fn = getattr(_sol, _FN_NAME, None)",
        "        except Exception:",
        "            pass",
        "if _fn is None:",
        "    _emit({'type': 'fn_missing', 'name': _FN_NAME})",
        "    sys.exit(0)",
        "",
        "# --- Run tests ---",
        "for _i, _tc in enumerate(_ALL_TESTS):",
        "    _kind = _tc['kind']",
        "    _timeout_s = _tc.get('timeout_ms', 5000) / 1000.0",
        "    _args = copy.deepcopy(_tc['args'])",
        "    _expected = _tc['expected']",
        "    _result = {",
        "        'type': 'test', 'index': _i, 'kind': _kind,",
        "        'passed': False, 'outcome': 'not_run',",
        "        'time_ms': None, 'error': None,",
        "    }",
        "    if _kind == 'public':",
        "        _result['input_repr'] = repr(_args)[:200]",
        "        _result['expected_repr'] = repr(_expected)[:200]",
        "    try:",
        "        _old_stdout = sys.stdout",
        "        _old_stderr = sys.stderr",
        "        _cap_out = io.StringIO()",
        "        _cap_err = io.StringIO()",
        "        sys.stdout = _cap_out",
        "        sys.stderr = _cap_err",
        "        _t0 = time.monotonic()",
        "        try:",
        "            _actual = _fn(*_args)",
        "        finally:",
        "            _elapsed = (time.monotonic() - _t0) * 1000",
        "            sys.stdout = _old_stdout",
        "            sys.stderr = _old_stderr",
        "        _result['time_ms'] = round(_elapsed, 2)",
        "        _ok = _compare(_actual, _expected, _COMPARATOR)",
        "        _result['passed'] = _ok",
        "        _result['outcome'] = 'pass' if _ok else 'wrong_answer'",
        "        if _kind == 'public':",
        "            _result['actual_repr'] = repr(_actual)[:200]",
        "    except BaseException as _exc:",
        "        sys.stdout = _old_stdout",
        "        sys.stderr = _old_stderr",
        "        _result['outcome'] = 'exception'",
        "        _result['error'] = type(_exc).__name__ + ': ' + str(_exc)[:300]",
        "        _elapsed = (time.monotonic() - _t0) * 1000 if '_t0' in dir() else 0",
        "        _result['time_ms'] = round(_elapsed, 2)",
        "    _emit(_result)",
        "",
        "_emit({'type': 'done'})",
    ]

    return "\n".join(lines) + "\n"


# ── Parser ──────────────────────────────────────────────────────────

@dataclass
class ParsedTestResult:
    """One parsed test result line from harness output."""
    index: int = 0
    kind: str = "public"
    passed: bool = False
    outcome: str = "not_run"
    time_ms: float | None = None
    input_repr: str | None = None
    expected_repr: str | None = None
    actual_repr: str | None = None
    error: str | None = None


@dataclass
class ParsedRun:
    """All parsed results from a harness run."""
    tests: list[ParsedTestResult] = field(default_factory=list)
    compile_error: str | None = None
    compile_error_line: int | None = None
    fn_missing: str | None = None
    done: bool = False


def parse_harness_output(stdout: str, nonce: str) -> ParsedRun:
    """Parse structured lines from harness stdout.

    This is a **correctness** defense, not a security boundary.
    It ignores lines without the exact nonce prefix, invalid JSON,
    truncated lines and unknown fields.  The first line for a given
    test index wins (no overwrites).
    """
    marker = f"{_MARKER_PREFIX}{nonce}{_MARKER_SUFFIX}"
    result = ParsedRun()
    seen_indices: set[int] = set()

    for raw_line in stdout.split("\n"):
        line = raw_line.strip()
        if not line.startswith(marker):
            continue
        json_part = line[len(marker):]
        try:
            data = json.loads(json_part)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        line_type = data.get("type")

        if line_type == "compile_error":
            if result.compile_error is None:
                result.compile_error = str(data.get("message", ""))[:2000]
                result.compile_error_line = data.get("line")

        elif line_type == "fn_missing":
            if result.fn_missing is None:
                result.fn_missing = str(data.get("name", ""))

        elif line_type == "done":
            result.done = True

        elif line_type == "test":
            idx = data.get("index")
            if not isinstance(idx, int):
                continue
            if idx in seen_indices:
                continue  # first line per index wins
            seen_indices.add(idx)

            tr = ParsedTestResult(
                index=idx,
                kind=str(data.get("kind", "public")),
                passed=bool(data.get("passed", False)),
                outcome=str(data.get("outcome", "not_run")),
                time_ms=data.get("time_ms"),
                input_repr=_trunc(data.get("input_repr"), 200),
                expected_repr=_trunc(data.get("expected_repr"), 200),
                actual_repr=_trunc(data.get("actual_repr"), 200),
                error=_trunc(data.get("error"), 500),
            )
            result.tests.append(tr)

    # Sort tests by index
    result.tests.sort(key=lambda t: t.index)
    return result


def _trunc(val: Any, limit: int) -> str | None:
    if val is None:
        return None
    s = str(val)
    if len(s) > limit:
        return s[:limit - 3] + "..."
    return s
