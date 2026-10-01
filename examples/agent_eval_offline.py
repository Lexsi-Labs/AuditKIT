"""Offline agent evaluation with the base package -- no agent, no network, no model.

Runs entirely on the standard library (UX-A2). It builds a tiny synthetic
episode set, attaches task-specific outcome oracles, and scores them offline so
a new user can learn the result format before deploying an agent or installing
AgentTune.

    python examples/agent_eval_offline.py
"""

from __future__ import annotations

from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    AnswerAssertion,
    FinalStateAssertion,
    episode_from_openai_messages,
)


def build_episodes():
    """Two recorded episodes: one that changed state, one that only claimed to."""
    # Episode A: called the tool, and the verified final state is 'ready'.
    ep_a = episode_from_openai_messages(
        [
            {"role": "user", "content": "Check inventory and ship item A if possible."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "lookup_inventory", "arguments": '{"sku": "A"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "in_stock=true"},
            {"role": "assistant", "content": "Shipment for A is ready."},
        ],
        case_id="inventory-001", final_answer="Shipment for A is ready.",
    )
    ep_a.final_state = {"shipment_decision": "ready"}

    # Episode B: a fabricated-correct answer, but the verified state disagrees.
    ep_b = episode_from_openai_messages(
        [
            {"role": "user", "content": "Check inventory and ship item B if possible."},
            {"role": "assistant", "content": "Shipment for B is ready."},
        ],
        case_id="inventory-002", final_answer="Shipment for B is ready.",
    )
    ep_b.final_state = {"shipment_decision": "blocked"}
    return [ep_a, ep_b]


def build_cases():
    return [
        AgentCase(
            id="inventory-001", task="Check inventory and ship item A if possible.",
            category="tool_use", allowed_tools=["lookup_inventory"],
            reference_turns=[[{"name": "lookup_inventory", "arguments": {"sku": "A"}}]],
            outcome=FinalStateAssertion("shipment_decision", equals="ready"),
        ),
        AgentCase(
            id="inventory-002", task="Check inventory and ship item B if possible.",
            category="tool_use", allowed_tools=["lookup_inventory"],
            # Verified state takes precedence over the (fabricated) answer text.
            outcome=FinalStateAssertion("shipment_decision", equals="ready"),
            metadata={"target": "Shipment for B is ready."},
        ),
    ]


def main() -> int:
    spec = AgentEvalSpec(
        cases=build_cases(),
        mode="recorded",
        episodes=build_episodes(),
        scorers=["tool_call_f1", "tool_call_validity"],  # no judge -> fully offline
        trials=1,
    )
    result = AgentEvalRunner().run(spec)
    print(result.summary())
    print()
    for row in result.rows:
        print(f"[{row.case_id}] status={row.status}  outcome={row.outcome['verdict']}  "
              f"coverage={row.coverage_label}")
        print(f"    reason: {row.outcome['reason']}")
        if row.ineligible:
            print(f"    ineligible: {row.ineligible}")
    # Episode B claimed success in text but its verified state is 'blocked' -> failure.
    verdicts = {r.case_id: r.outcome["verdict"] for r in result.rows}
    assert verdicts["inventory-001"] == "success", verdicts
    assert verdicts["inventory-002"] == "failure", verdicts
    print("\nOK: a fabricated answer did not pass a state-based case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
