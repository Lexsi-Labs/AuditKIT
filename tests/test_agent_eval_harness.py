"""A4: harness-owned bounded tool loop (AG-15/AG-16/AG-17), fully offline.

AuditKit drives the loop against a scripted policy Model and a caller-supplied
tool-env double; no network. Covers a completed loop with real state
verification, the max_steps bound, and the no-tool-env seam.
"""

from __future__ import annotations

from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    FinalStateAssertion,
)
from auditkit.model import Generated, Model, Result_


class _ShipEnv:
    """A resettable test double: `ship` flips state; `snapshot()` verifies it."""

    def __init__(self) -> None:
        self._state: dict = {}

    def __call__(self, name, arguments):
        if name == "ship":
            self._state["shipped"] = True
            return "shipped ok"
        return "unknown tool"

    def snapshot(self) -> dict:
        return dict(self._state)


class _ShipThenAnswer(Model):
    """Step 1: call ship. Step 2: no calls, final text -> loop stops."""

    name = "scripted"

    def __init__(self) -> None:
        self.step = 0

    def generate(self, requests):
        self.step += 1
        if self.step == 1:
            return [Result_(completions=[Generated(
                text=None, trace={"tool_calls": [[{"name": "ship", "arguments": {"sku": "A"}}]]})])]
        return [Result_(completions=[Generated(text="Shipped A.", trace={"tool_calls": []})])]


class _AlwaysCalls(Model):
    name = "loopy"

    def generate(self, requests):
        return [Result_(completions=[Generated(
            text=None, trace={"tool_calls": [[{"name": "ship", "arguments": {}}]]})])]


def _case():
    return AgentCase(id="ship-1", task="Ship item A.", allowed_tools=["ship"],
                     outcome=FinalStateAssertion("shipped", equals=True))


def test_harness_completes_and_verifies_state():
    spec = AgentEvalSpec(cases=[_case()], mode="harness", agent=_ShipThenAnswer(),
                         agent_opts={"tool_env": _ShipEnv(), "max_steps": 5},
                         scorers=["tool_call_validity"])
    result = AgentEvalRunner().run(spec)
    row = result.rows[0]
    assert row.status == "completed"
    assert row.outcome["verdict"] == "success"          # state oracle verified a real effect
    assert row.coverage["final_state"] == "observed"
    assert row.stop_reason == "stop"
    ep = result.episodes[0]
    assert ep.mode == "harness"
    assert [e.payload.get("name") or e.payload.get("function", {}).get("name")
            for e in ep.tool_call_events()] == ["ship"]
    assert ep.coverage_label() == "full"                 # arguments recorded across steps


def test_harness_max_steps_is_budget_exhausted():
    spec = AgentEvalSpec(cases=[_case()], mode="harness", agent=_AlwaysCalls(),
                         agent_opts={"tool_env": _ShipEnv(), "max_steps": 3}, scorers=[])
    row = AgentEvalRunner().run(spec).rows[0]
    assert row.stop_reason == "max_steps"
    assert row.status == "budget_exhausted"
    assert row.resource_use["n_tool_calls"] == 3          # one call per bounded step


def test_harness_without_tool_env_documents_the_seam():
    spec = AgentEvalSpec(cases=[_case()], mode="harness", agent=_AlwaysCalls(),
                         agent_opts={"max_steps": 4}, scorers=[])
    row = AgentEvalRunner().run(spec).rows[0]
    # No double -> cannot feed results back, loop stops after one step (the seam).
    assert row.stop_reason == "no_tool_env"


# -- G1 / G2 (fix plan #15): text tool calls and the generation budget -----------------------------------

import pytest  # noqa: E402

from auditkit.runspec import RunConfig  # noqa: E402


class _TextPolicy(Model):
    """A text-output policy (like hf:/vllm:/Cohere): calls are written into the reply."""

    name = "text-policy"

    def __init__(self, first_reply) -> None:
        self.step, self.first_reply, self.params = 0, first_reply, []

    def generate(self, requests):
        self.step += 1
        self.params.append(dict(requests[0].params))
        text = self.first_reply if self.step == 1 else "Shipped A."
        return [Result_(completions=[Generated(text=text)])]


@pytest.mark.parametrize("first_reply", [
    '<think>ship it</think>\n<tool_call>\n{"name": "ship", "arguments": {"sku": "A"}}\n</tool_call>',   # Hermes / Qwen
    '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "ship", "parameters": {"sku": "A"}}]<|END_ACTION|>',  # Cohere
    '{"name": "ship", "arguments": {"sku": "A"}}',                                                          # bare JSON
])
def test_g1_the_harness_executes_text_tool_calls(first_reply):
    env = _ShipEnv()
    spec = AgentEvalSpec(cases=[_case()], mode="harness", agent=_TextPolicy(first_reply),
                         agent_opts={"tool_env": env, "max_steps": 5}, scorers=[])
    result = AgentEvalRunner().run(spec)
    assert env.snapshot() == {"shipped": True}
    assert result.rows[0].outcome["verdict"] == "success"
    assert result.episodes[0].metadata.get("calls_from_text") is True


def test_g1_an_explicit_empty_structured_trace_is_not_reparsed():
    class Structured(Model):
        name = "structured"

        def generate(self, requests):
            return [Result_(completions=[Generated(text='<tool_call>{"name": "ship", "arguments": {}}</tool_call>',
                                                   trace={"tool_calls": []})])]
    env = _ShipEnv()
    AgentEvalRunner().run(AgentEvalSpec(cases=[_case()], mode="harness", agent=Structured(),
                                        agent_opts={"tool_env": env, "max_steps": 3}, scorers=[]))
    assert env.snapshot() == {}                     # the backend said "no calls": trusted


def test_g2_the_generation_config_reaches_every_step_and_the_identity():
    policy = _TextPolicy('<tool_call>{"name": "ship", "arguments": {"sku": "A"}}</tool_call>')
    spec = AgentEvalSpec(cases=[_case()], mode="harness", agent=policy,
                         agent_opts={"tool_env": _ShipEnv(), "max_steps": 5}, scorers=[],
                         config=RunConfig(max_tokens=512, temperature=0.0, seed=7))
    AgentEvalRunner().run(spec)
    assert len(policy.params) == 2
    assert all(p["max_tokens"] == 512 and p["seed"] == 7 and "messages" in p for p in policy.params)
    assert spec.identity()["generation"] == {"max_tokens": 512, "temperature": 0.0, "seed": 7}


def test_g2_no_config_keeps_the_backend_defaults():
    policy = _TextPolicy("done")
    AgentEvalRunner().run(AgentEvalSpec(cases=[_case()], mode="harness", agent=policy,
                                        agent_opts={"tool_env": _ShipEnv(), "max_steps": 2}, scorers=[]))
    assert "max_tokens" not in policy.params[0]


@pytest.mark.parametrize("first_reply", [
    '<think>ship it</think>\n<tool_call>\n{"name": "ship", "arguments": {"sku": "A"}}\n</tool_call>',
    'I will ship it.<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "ship", "parameters": {"sku": "A"}}]<|END_ACTION|>',
    'Action: ```json\n[{"tool_name": "ship", "parameters": {"sku": "A"}}]\n```',
    '{"name": "ship", "arguments": {"sku": "A"}}',
])
def test_g1_text_calls_are_not_left_in_the_assistant_content(first_reply):
    from auditkit.trace import parse_tool_calls
    policy = _TextPolicy(first_reply)
    AgentEvalRunner().run(AgentEvalSpec(cases=[_case()], mode="harness", agent=policy,
                                        agent_opts={"tool_env": _ShipEnv(), "max_steps": 5}, scorers=[]))
    turn = policy.params[1]["messages"][1]          # the assistant step fed back to the policy
    assert turn["tool_calls"][0]["function"]["name"] == "ship"
    assert parse_tool_calls(turn["content"]) == [] and "sku" not in turn["content"]   # rendered once


@pytest.mark.parametrize("first_reply,executed", [
    # the second call was cut off (max_tokens): the whole action list is not executed
    ('<|START_ACTION|>[{"tool_name": "ship", "parameters": {"sku": "A"}}, '
     '{"tool_name": "ship", "parameters": {"sku": "B', {}),
    ('<|START_ACTION|>[{"tool_name": "ship", "param', {}),             # nothing parseable at all
])
def test_g1_a_call_with_a_parse_error_is_recorded_not_dropped(first_reply, executed):
    env = _ShipEnv()
    result = AgentEvalRunner().run(AgentEvalSpec(cases=[_case()], mode="harness",
                                                 agent=_TextPolicy(first_reply),
                                                 agent_opts={"tool_env": env, "max_steps": 5}, scorers=[]))
    ep, row = result.episodes[0], result.rows[0]
    assert env.snapshot() == executed
    assert ep.stop_reason == "parse_error" and ep.final_answer is None     # not the final answer
    assert any("truncated" in e for e in row.errors)
