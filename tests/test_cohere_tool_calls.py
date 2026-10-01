"""Cohere native tool calls (Command R7B, Command-R / Aya Expanse)
are read by the tool-call parser, so tool metrics score them. Stdlib only.

The reply strings are the forms AgentTune's rollout parser handles
(``rollout_factory._extract_tool_calls``).
"""

from __future__ import annotations

import auditkit as ak
from auditkit.agent_eval.types import AgentEpisode, AgentEvent
from auditkit.trace import parse_tool_calls

R7B = ('<|START_THINKING|>I will add the numbers.<|END_THINKING|><|START_ACTION|>[\n'
       '    {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}\n'
       ']<|END_ACTION|>')
# What a skip_special_tokens=True decode leaves of R7B.
STRIPPED = 'I will add the numbers.[\n    {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}\n]'
COMMAND_R = ('Action: ```json\n[\n    {\n        "tool_name": "add",\n'
             '        "parameters": {"a": 2, "b": 3}\n    }\n]\n```')
PARALLEL = ('<|START_THINKING|>Two [sums].<|END_THINKING|><|START_ACTION|>['
            '{"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}, '
            '{"tool_call_id": "1", "tool_name": "add", "parameters": {"a": 1, "b": 1}}'
            ']<|END_ACTION|>')
# Stripped, with a bracket in the plan: the first "[" is prose, not a call list.
PARALLEL_STRIPPED = ('Two [sums].[{"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}, '
                     '{"tool_call_id": "1", "tool_name": "add", "parameters": {"a": 1, "b": 1}}]')

ADD = {"name": "add", "arguments": {"a": 2, "b": 3}}


def _calls(text):
    return [(c.name, c.arguments, c.parse_error) for c in parse_tool_calls(text)]


def test_single_call_every_surface_form():
    for text in (R7B, STRIPPED, COMMAND_R):
        assert _calls(text) == [("add", {"a": 2, "b": 3}, None)], text


def test_parallel_calls_and_brackets_in_the_plan():
    for text in (PARALLEL, PARALLEL_STRIPPED):
        assert _calls(text) == [("add", {"a": 2, "b": 3}, None), ("add", {"a": 1, "b": 1}, None)], text


def test_plain_answers_are_not_calls():
    assert _calls("The answer is [5].") == []
    assert _calls('I would use tool_name "add" but [the list] is empty.') == []
    assert _calls("<|START_RESPONSE|>5<|END_RESPONSE|>") == []


def test_tool_call_f1_scores_cohere_output(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = [ak.Sample(input="2+3?", actual_output=t, expected_tool_calls=[ADD])
               for t in (R7B, STRIPPED, COMMAND_R)]
    samples.append(ak.Sample(input="2+3?", actual_output="5", expected_tool_calls=[ADD]))
    r = ak.evaluate(samples, "precomputed", [ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == r.headline["tool_call_exact"] == 0.75


def test_recorded_cohere_call_dicts_become_turns():
    """A trajectory that recorded Cohere's raw call dict is still a call."""
    call = {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}
    ep = AgentEpisode(events=[AgentEvent(index=0, role="assistant", type="tool_call",
                                         payload=call, turn_id=0)])
    assert ep.turns() == [[call]]
    assert ep.to_trace()["messages"][0]["tool_calls"] == [call]
