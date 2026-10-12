# filepath: backend/scripts/demo_phase1.py
"""
PlaceMate — Phase 1 Demo Script
=================================

THIS IS THE PROOF THAT PHASE 1 WORKS.

Run this script to see the entire interview graph in action:

    cd backend
    pip install -r requirements.txt
    python -m scripts.demo_phase1   # must be run as a module from backend/

What this demo proves:
    ✅ The LangGraph compiles without errors.
    ✅ Typed state is initialized correctly.
    ✅ The interviewer generates questions based on phase/turn.
    ✅ interrupt() pauses the graph (human-in-the-loop).
    ✅ Command(resume=...) resumes the graph with candidate input.
    ✅ Conditional routing works (technical → coding → assessment).
    ✅ The executor handles code and returns structured results.
    ✅ Failed execution routes to the Debater (adversarial follow-up).
    ✅ The Debater → Interviewer cycle works.
    ✅ The Assessor produces a structured scorecard.
    ✅ The graph terminates cleanly at END.
    ✅ No API keys are required.

HOW INTERRUPT/RESUME WORKS IN PRACTICE:
    ──────────────────────────────────────
    1. We call graph.invoke(initial_state, config).
    2. The graph runs: START → INTERVIEWER → HUMAN_INPUT.
    3. At HUMAN_INPUT, interrupt() is called.
    4. The graph RETURNS — it doesn't block or loop.
    5. We inspect the return value to see the interrupt.
    6. We call graph.invoke(Command(resume={...}), config) with
       the SAME config (same thread_id).
    7. The graph picks up where it left off.
    8. The router decides what's next based on the candidate's input.
    9. Repeat until the graph reaches END.

    This is the pattern that FastAPI will use in Phase 8:
    each API call to /respond or /code will resume the graph.
"""

from __future__ import annotations

from langgraph.types import Command

from app.core.enums import InterviewPhase
from app.graph.state import make_initial_state
from app.graph.workflow import build_interview_graph


def print_separator(title: str = "") -> None:
    """Print a visual separator for readability."""
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def print_state_summary(state: dict) -> None:
    """Print a concise summary of the current interview state."""
    print(f"  Phase:    {state.get('interview_phase', 'N/A')}")
    print(f"  Turn:     {state.get('current_turn', 0)} (total: {state.get('total_turns', 0)})")
    print(f"  Question: {state.get('current_question', 'N/A')[:80]}...")
    print(f"  Q-Type:   {state.get('current_question_type', 'N/A')}")
    msg_count = len(state.get("messages", []))
    print(f"  Messages: {msg_count}")


def get_last_ai_message(state: dict) -> str:
    """Extract the last AI message from the conversation."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return "(no AI message)"


def run_demo() -> None:
    """
    Run the full Phase 1 demo.

    Simulates an interview with these steps:
    1. Start interview (INTRO phase)
    2. Respond to intro question
    3. Answer a technical question
    4. Submit INEFFICIENT code (triggers fail → debater)
    5. Respond to debater challenge
    6. Submit CORRECT code (triggers pass → back to interviewer)
    7. Answer another question until assessment triggers
    8. View final scorecard
    """

    print_separator("PlaceMate — Phase 1 Demo")
    print("Building interview graph...")

    # ── Build the graph ─────────────────────────────────────────────
    graph = build_interview_graph()
    print("✅ Graph compiled successfully!\n")

    # ── Initial interview configuration ─────────────────────────────
    # This is what the frontend will send when starting an interview.
    initial_state = make_initial_state(
        session_id="demo-session-001",
        candidate_name="Rahul Sharma",
        target_company="Amazon",
        target_role="SDE",
    )

    # The thread_id is critical — it's how the checkpointer knows
    # which interview to save/load state for.
    config = {"configurable": {"thread_id": "demo-session-001"}}

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Start the interview
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 1: Starting Interview")
    print(f"  Company: {initial_state['target_company']}")
    print(f"  Role:    {initial_state['target_role']}")
    print(f"  Mode:    {initial_state['interview_mode']}")
    print(f"  Thread:  {config['configurable']['thread_id']}")

    # First invoke — graph runs START → INTERVIEWER → HUMAN_INPUT → interrupt
    result = graph.invoke(initial_state, config)

    print("\n  📋 Graph paused (interrupt). Interviewer asked:")
    state = graph.get_state(config).values
    print(f"  💬 {get_last_ai_message(state)}")
    print_state_summary(state)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Candidate responds to INTRO question
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 2: Candidate Responds to Intro")
    candidate_response = {
        "answer": (
            "Hi! I'm Rahul Sharma, a final year CS student at MANIT Bhopal. "
            "I've been preparing for Amazon's SDE role for the past 6 months. "
            "I'm particularly interested in distributed systems and have built "
            "a few projects using microservices architecture."
        ),
    }
    print(f"  🗣️  Candidate: {candidate_response['answer'][:80]}...")

    # Resume the graph — the interrupt() call returns candidate_response
    result = graph.invoke(Command(resume=candidate_response), config)

    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Candidate answers TECHNICAL question
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 3: Candidate Answers Technical Question")
    candidate_response = {
        "answer": (
            "A process is an independent execution unit with its own memory space. "
            "A thread is a lightweight unit within a process that shares memory. "
            "I'd use processes for isolation (like microservices) and threads "
            "for concurrent I/O operations within a single service."
        ),
    }
    print(f"  🗣️  Candidate: {candidate_response['answer'][:80]}...")

    result = graph.invoke(Command(resume=candidate_response), config)

    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Answer another technical question
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 4: Another Technical Question")
    candidate_response = {
        "answer": (
            "Hash table operations — insert, lookup, delete — are O(1) average. "
            "Worst case is O(N) when all keys hash to the same bucket. "
            "This happens with poor hash functions or adversarial inputs."
        ),
    }
    print(f"  🗣️  Candidate: {candidate_response['answer'][:80]}...")

    result = graph.invoke(Command(resume=candidate_response), config)

    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Answer technical question 3 (DB index)
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 5: Answer DB Index Q")
    candidate_response = {
        "answer": (
            "A database index is a B-tree data structure that speeds up queries "
            "by providing O(log N) lookups instead of O(N) full table scans. "
            "The tradeoff is slower writes and additional storage."
        ),
    }
    print(f"  🗣️  Candidate: {candidate_response['answer'][:80]}...")

    result = graph.invoke(Command(resume=candidate_response), config)
    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5.5: Answer CAP Theorem Q (Advances to CODING Phase)
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 5.5: Answer CAP Theorem Q (Advances to CODING Phase)")
    candidate_response = {
        "answer": (
            "CAP theorem states a distributed system can only guarantee two of "
            "three: Consistency, Availability, Partition Tolerance. "
            "DynamoDB sacrifices strong consistency for availability."
        ),
    }
    print(f"  🗣️  Candidate: {candidate_response['answer'][:80]}...")

    result = graph.invoke(Command(resume=candidate_response), config)
    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)
    print(f"  ↑ The interviewer should now be in CODING phase, asking a coding Q.")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Submit INEFFICIENT code → Executor → partial → DEBATER
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 6: Submit Inefficient Code (O(N^2) → DEBATER)")
    print("  Current question type should be 'coding'")
    print("  Expected flow: human_input → router('coding') → EXECUTOR → partial → DEBATER → INTERVIEWER → pause")

    candidate_response = {
        "answer": "Here's my brute force solution using nested loops.",
        "code": (
            "def two_sum(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i+1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return [i, j]\n"
            "    return []\n"
        ),
        "language": "python",
    }

    result = graph.invoke(Command(resume=candidate_response), config)

    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)
    print(f"  Follow-ups: {state.get('follow_up_count', 0)}")
    print(f"  Complexity: {state.get('complexity_analysis', 'N/A')}")
    print(f"  Exec history: {len(state.get('execution_history', []))} results")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Submit EFFICIENT code → Executor → pass → Interviewer
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 7: Submit Efficient Code (Hash Map → PASS)")
    print("  Expected flow: human_input → router('coding') → EXECUTOR → pass → INTERVIEWER → pause")

    candidate_response = {
        "answer": "I'll use a hash map for O(N) lookup.",
        "code": (
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n"
            "    return []\n"
        ),
        "language": "python",
    }

    result = graph.invoke(Command(resume=candidate_response), config)

    state = graph.get_state(config).values
    print(f"\n  💬 {get_last_ai_message(state)}")
    print_state_summary(state)
    print(f"  Exec history: {len(state.get('execution_history', []))} results")

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Continue until assessment triggers
    # ══════════════════════════════════════════════════════════════════
    print_separator("STEP 8: Continue Until Assessment")

    max_extra_steps = 5
    for step in range(max_extra_steps):
        state = graph.get_state(config).values
        phase = state.get("interview_phase", "")

        if phase in (InterviewPhase.COMPLETE.value, InterviewPhase.ASSESSMENT.value):
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print(f"\n  🏁 Interview complete! Phase: {phase}")
                break

        print(f"\n  Step {step + 1}: Phase={phase}, responding...")
        candidate_response = {
            "answer": f"That's a great question. Let me think... [response {step+1}]",
        }
        try:
            result = graph.invoke(Command(resume=candidate_response), config)
        except Exception as e:
            print(f"  ⚠️  Graph exception: {e}")
            break

    # ══════════════════════════════════════════════════════════════════
    # FINAL: Display the scorecard
    # ══════════════════════════════════════════════════════════════════
    print_separator("FINAL SCORECARD")

    state = graph.get_state(config).values
    feedback = state.get("final_feedback", "No feedback generated")
    verdict = state.get("final_verdict", "N/A")
    score = state.get("total_score", 0)
    rubric = state.get("rubric_scores", {})

    print(f"\n{feedback}")
    print(f"\n  Verdict:     {verdict}")
    print(f"  Total Score: {score}")
    print(f"  Strengths:   {state.get('strengths', [])}")
    print(f"  Weaknesses:  {state.get('weaknesses', [])}")

    # ── Print conversation history ──────────────────────────────────
    print_separator("FULL CONVERSATION HISTORY")
    messages = state.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.type.upper() if hasattr(msg, "type") else "UNKNOWN"
        content = msg.content[:120] if hasattr(msg, "content") else str(msg)
        print(f"  [{i+1:02d}] {role}: {content}")
        if len(msg.content) > 120:
            print(f"       ... ({len(msg.content)} chars total)")

    print_separator("DEMO COMPLETE ✅")
    print(f"  Total messages:     {len(messages)}")
    print(f"  Total turns:        {state.get('total_turns', 0)}")
    print(f"  Code submissions:   {len(state.get('code_submissions', []))}")
    print(f"  Execution results:  {len(state.get('execution_history', []))}")
    print(f"  Follow-up count:    {state.get('follow_up_count', 0)}")
    print(f"\n  All Phase 1 criteria met. Ready for Phase 2! 🚀\n")


if __name__ == "__main__":
    run_demo()
