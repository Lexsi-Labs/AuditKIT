"""Cohere tool calling (#10): every surface form the parser reads, scoring through the
tool metrics, recorded Cohere calls in AgentTune event logs, and the real Cohere chat
templates (Command R7B, Aya Expanse, Tiny Aya) from tests/fixtures/chat_templates.

Offline. The template tests need the gpt2 tokenizer in the local HF cache (only its
vocabulary is used; the template is the Cohere one).
"""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.trace import normalize_call, parse_tool_calls

from .conftest import cached_tokenizer

pytestmark = pytest.mark.skipif(not hasattr(ak, "episodes_from_agenttune"), reason="needs #10")


def calls(text):
    return [(c.name, c.arguments) for c in parse_tool_calls(text)]


def act(*items, think=None, respond=None):
    body = json.dumps(list(items))
    out = f"<|START_THINKING|>{think}<|END_THINKING|>" if think is not None else ""
    out += f"<|START_ACTION|>{body}<|END_ACTION|>"
    if respond is not None:
        out += f"<|START_RESPONSE|>{respond}<|END_RESPONSE|>"
    return out


def c(name, i="0", **params):
    return {"tool_call_id": i, "tool_name": name, "parameters": params}


# -- parsing -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    (act(c("add", a=2, b=3)), [("add", {"a": 2, "b": 3})]),
    (act(c("add", a=2, b=3), think="I will add [2, 3]."), [("add", {"a": 2, "b": 3})]),   # bracket in the plan
    ("I will add." + json.dumps([c("add", a=2)]), [("add", {"a": 2})]),                     # special tokens stripped
    ("Action: ```json\n" + json.dumps([c("add", a=2)], indent=4) + "\n```", [("add", {"a": 2})]),   # Command-R
    (act(c("add", "0", a=1), c("add", "1", a=2)), [("add", {"a": 1}), ("add", {"a": 2})]),  # parallel
    (act(c("add", a=2), respond="done"), [("add", {"a": 2})]),                              # action then response
    (act({"tool_name": "add", "parameters": json.dumps({"a": 2})}), [("add", {"a": 2})]),   # parameters as a string
    (act({"tool_name": "get_time"}), [("get_time", {})]),                                   # no parameters
    (act(c("search", q="naïve café 東京", filters={"k": [1, 2]})),
     [("search", {"q": "naïve café 東京", "filters": {"k": [1, 2]}})]),                     # unicode, nesting
])
def test_every_cohere_surface_form(text, want):
    assert calls(text) == want


@pytest.mark.parametrize("text", [
    "The answer is [5].",
    "<|START_RESPONSE|>5<|END_RESPONSE|>",
    act() + "<|START_RESPONSE|>no tool needed<|END_RESPONSE|>",                            # empty action list
    'I considered tool_name "add" but answered directly.',
])
def test_plain_answers_are_not_calls(text):
    assert calls(text) == []


def test_a_list_hidden_in_the_thinking_block_is_ignored():
    text = act(c("add", a=1), think='maybe [{"tool_name": "rm", "parameters": {}}]')
    assert calls(text) == [("add", {"a": 1})]


def test_the_cohere_call_id_is_kept():
    assert normalize_call(c("add", "7", a=1)).id == "7"


def test_command_r_directly_answer_is_not_a_tool_call():
    text = "Action: ```json\n" + json.dumps([{"tool_name": "directly-answer", "parameters": {}}]) + "\n```"
    assert calls(text) == []


def test_a_truncated_action_list_is_reported_not_dropped():
    text = '<|START_ACTION|>[{"tool_name": "a", "parameters": {}}, {"tool_name": "b", "parameters": {"x": 1'
    got = parse_tool_calls(text)
    assert got and got[0].name == "a"


# -- scoring through evaluate() ---------------------------------------------------------------------------------

ADD = {"name": "add", "arguments": {"a": 2, "b": 3}}
ADD11 = {"name": "add", "arguments": {"a": 1, "b": 1}}


@pytest.mark.parametrize("output,expected,metric,want", [
    (act(c("add", a=2, b=3)), [ADD], "tool_call_f1", 1.0),
    (act(c("add", a=2, b=4)), [ADD], "tool_call_f1", 0.0),                                    # wrong argument
    (act(c("add", "0", a=2, b=3), c("add", "1", a=1, b=1)), [[ADD, ADD11]], "parallel_detection", 1.0),
    ("5", [ADD], "tool_call_f1", 0.0),                                                         # answered instead
    ("5", [], "tool_call_f1", 1.0),                                                            # correct irrelevance
])
def test_cohere_output_scores_through_the_tool_metrics(output, expected, metric, want):
    s = ak.Sample(input="2+3?", actual_output=output, expected_tool_calls=expected)
    r = ak.evaluate([s], "precomputed", [ak.ToolCallF1(), ak.ParallelToolCalls()])
    assert r.errors == [] and r.headline[metric] == want


def test_cohere_output_is_schema_validated():
    tools = [{"type": "function", "function": {"name": "add", "parameters": {
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
    good = ak.Sample(id="g", input="q", tools=tools, actual_output=act(c("add", a=2, b=3)))
    bad = ak.Sample(id="b", input="q", tools=tools, actual_output=act(c("add", a="two")))
    r = ak.evaluate([good, bad], "precomputed", [ak.ToolCallValidity()])
    assert r.headline["tool_call_validity"] == 0.5


# -- recorded Cohere calls (AgentTune event logs) ------------------------------------------------------------------

def eventlog(*events, id="e"):
    return {"id": id, "tier": "full", "events": list(events)}


def test_cohere_calls_in_an_event_log_become_scored_turns():
    from auditkit.agent_eval.importers import episode_from_eventlog
    e = episode_from_eventlog(eventlog(
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [c("add", "0", a=2, b=3), c("add", "1", a=1, b=1)]}}},
        {"kind": "tool_result", "payload": {"output": "5"}},
        {"kind": "text", "payload": {"text": "5 and 2"}}))
    assert e.coverage["tool_call_arguments"] == "observed"
    s = ak.Sample(input="q", actual_output=e.final_answer, actual_trace=e.to_trace(), expected_tool_calls=[[ADD, ADD11]])
    assert ak.evaluate([s], "precomputed", [ak.ToolCallF1()]).headline["tool_call_f1"] == 1.0


def test_a_parameterless_cohere_call_reaches_the_trace():
    from auditkit.agent_eval.importers import episode_from_eventlog
    e = episode_from_eventlog(eventlog(
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [{"tool_call_id": "0", "tool_name": "get_time"}]}}},
        {"kind": "text", "payload": {"text": "noon"}}))
    assert e.counters["n_tool_calls"] == 1 and e.to_trace()["tool_calls"] != []


def test_cohere_call_result_pairing_is_observed():
    from auditkit.agent_eval.importers import episode_from_eventlog
    e = episode_from_eventlog(eventlog(
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [c("add", "7", a=1)]}}},
        {"kind": "tool_result", "payload": {"tool_call_id": "7", "output": "2"}},
        {"kind": "text", "payload": {"text": "2"}}))
    assert e.coverage["call_result_pairing"] == "observed"


# -- the real Cohere chat templates --------------------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "add", "description": "add two ints", "parameters": {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]


def render(template, messages, **kw):
    tok = cached_tokenizer("gpt2", template)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def test_command_r7b_renders_tools_and_a_full_tool_turn():
    out = render("command_r7b_tool_use.jinja", [
        {"role": "user", "content": "2+3?"},
        {"role": "assistant", "tool_plan": "I will add.",
         "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "add", "arguments": {"a": 2, "b": 3}}}]},
        {"role": "tool", "tool_call_id": "call_9", "content": "5"}], tools=TOOLS)
    assert '"add"' in out and "<|START_ACTION|>" in out and "<|START_TOOL_RESULT|>" in out
    # what the template renders for the history is itself parseable as the same call
    history = out.rsplit("<|START_ACTION|>", 1)[1].split("<|END_ACTION|>")[0]   # the last one is the turn
    assert calls(history) == [("add", {"a": 2, "b": 3})]
    assert out.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")


@pytest.mark.parametrize("template", ["aya_expanse.jinja", "tiny_aya_default.jinja"])
def test_aya_templates_do_not_render_tools(template):
    out = render(template, [{"role": "user", "content": "2+3?"}], tools=TOOLS)
    assert '"add"' not in out                       # so hf: must refuse native tools for these (test_hf.py)


def test_openai_string_arguments_render_as_an_object_for_command_r7b():
    from auditkit.model import Request

    from .conftest import stub_hf
    m = stub_hf(cached_tokenizer("gpt2", "command_r7b_tool_use.jinja"))
    history = [{"role": "user", "content": "2+3?"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "call_9", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
               {"role": "tool", "tool_call_id": "call_9", "content": "5"}]
    m.generate([Request(prompt="2+3?", params={"messages": history, "tools": TOOLS})])
    prompt = m._pipeline.calls[-1][0][0]
    assert '"parameters": {"a": 2, "b": 3}' in prompt
