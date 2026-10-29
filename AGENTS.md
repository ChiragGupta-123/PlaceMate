# PlaceMate agent rules

Project: LangGraph mock technical-interview simulator (Python, FastAPI later). Backend is in backend/. 12-phase roadmap; we are in Phase 3 (Judge0 code execution). Never start a later phase.

HARD RULES
- Never read, print, log or commit .env or any API key. Credentials come from environment variables only (SecretStr in settings).
- Never run candidate code locally (no exec/eval/subprocess on it, in app code or tests). Never run the generated harness locally; tests may only ast.parse it.
- Strict type hints. Do not reorganize folders.
- Layering: executor_node -> ExecutionService -> SandboxInterface (Protocol) -> Judge0Sandbox / FakeSandbox. Only app/tools/judge0.py knows httpx and RapidAPI.
- Tests are offline with fakes. Real-provider checks use the `live` pytest marker.
- Never git commit or git push. I commit myself.

WORKFLOW
- One step at a time. After each step: run `python -m pytest -q` from backend/, show the output, summarize in 5 lines, then STOP.
- Every bug fix needs a regression test that fails before the fix and passes after. Show the failing run first. Never weaken or edit an existing test to make it pass.
- Read each file before editing it. Make the smallest change that fixes the bug.

DECISIONS (do not change)
C2 In _interpret(): if response.outcome == timeout and not all_correct, status = time_limit_exceeded, even when some tests parsed.
C3 Only evaluated runs (passed, failed, compile_error, runtime_error, time_limit_exceeded, memory_limit_exceeded) increment execution_attempts. infra_error increments infra_retries and leaves execution_attempts alone. unsupported_language and invalid_submission increment neither counter. Reset both counters in interviewer_node when a new question is asked.
D2 MAX_INFRA_RETRIES = 2 means two retries: the candidate may retry after the 1st and 2nd infra_error; the 3rd infra_error becomes status not_evaluated, the graph moves on, and the Assessor skips it. The candidate is never penalized for it.
D3 Every evaluated submission counts toward MAX_EXECUTION_ATTEMPTS (4). The router returns "pass" at attempts >= MAX. The complexity probe fires once, guarded by optimization_probe_done, which debater_node must set.