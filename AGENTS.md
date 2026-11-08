# PlaceMate agent rules

STATUS: Phases 1-2 done. Phase 3 is offline-complete (steps 3.1-3.6, 175 tests). Step 3.7 (live Judge0 probe) is PENDING: no payment method yet, so FakeSandbox only. Do NOT write or run anything that needs a real Judge0 key. Now in Phase 4 (real Interviewer LLM). Phases 5-12 are not started: never start a later phase.



Project: LangGraph mock technical-interview simulator (Python, FastAPI later). Backend is in backend/. 12-phase roadmap; we are in Phase 4 (Interviewer LLM). Never start a later phase.

HARD RULES
- Never read, print, log or commit .env or any API key. Credentials come from environment variables only (SecretStr in settings).ss
- Never run candidate code locally (no exec/eval/subprocess on it, in app code or tests). Never run the generated harness locally; tests may only ast.parse it.
- Strict type hints. Do not reorganize folders.
- Layering: executor_node -> ExecutionService -> SandboxInterface (Protocol) -> Judge0Sandbox / FakeSandbox. Only app/tools/judge0.py knows httpx and RapidAPI.
- Tests are offline with fakes. Real-provider checks use the `live` pytest marker.
- Never git commit or git push until and unless I say. I commit myself.
-After finishing a sub-step, STOP and wait for my "next". Never start the next sub-step on your own.

WORKFLOW
- One step at a time. After each step: run `python -m pytest -q` from backend/, show the output, summarize in 5 lines, then STOP.
- Every bug fix needs a regression test that fails before the fix and passes after. Show the failing run first. Never weaken or edit an existing test to make it pass.
- Read each file before editing it. Make the smallest change that fixes the bug.

DECISIONS (do not change)
C2 In _interpret(): if response.outcome == timeout and not all_correct, status = time_limit_exceeded, even when some tests parsed.
C3 Only evaluated runs (passed, failed, compile_error, runtime_error, time_limit_exceeded, memory_limit_exceeded) increment execution_attempts. infra_error increments infra_retries and leaves execution_attempts alone. unsupported_language and invalid_submission increment neither counter. Reset both counters in interviewer_node when a new question is asked.
D2 MAX_INFRA_RETRIES = 2 means two retries: the candidate may retry after the 1st and 2nd infra_error; the 3rd infra_error becomes status not_evaluated, the graph moves on, and the Assessor skips it. The candidate is never penalized for it.
D3 Every evaluated submission counts toward MAX_EXECUTION_ATTEMPTS (4). The router returns "pass" at attempts >= MAX. The complexity probe fires once, guarded by optimization_probe_done, which debater_node must set.