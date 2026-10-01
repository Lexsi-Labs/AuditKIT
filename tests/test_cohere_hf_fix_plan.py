"""Regression tests for G7-G12 (fix plan #15): Cohere tool-call parsing and import, and
the hf: backend's chat-template arguments and BOS handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from auditkit.agent_eval.importers import episode_from_eventlog
from auditkit.model import Request, template_messages
from auditkit.trace import is_call_payload, normalize_call, parse_tool_calls

TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"


def calls(text):
    return [(c.name, c.arguments, c.id, c.parse_error) for c in parse_tool_calls(text)]


# -- G7: Cohere's tool_call_id is kept, and results pair with it ----------------------------------------------

def test_g7_the_cohere_call_id_is_kept():
    assert normalize_call({"tool_call_id": "7", "tool_name": "add", "parameters": {}}).id == "7"
    assert normalize_call({"id": "a", "tool_call_id": "7", "name": "add"}).id == "a"      # an explicit id wins


def test_g7_cohere_results_pair_with_their_calls():
    e = episode_from_eventlog({"id": "e", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [
            {"tool_call_id": "7", "tool_name": "add", "parameters": {"a": 1}}]}}},
        {"kind": "tool_result", "payload": {"tool_call_id": "7", "output": "2"}},
        {"kind": "text", "payload": {"text": "2"}}]})
    assert e.coverage["call_result_pairing"] == "observed"
    assert [x.call_id for x in e.events if x.type in ("tool_call", "tool_result")] == ["7", "7"]


def test_g7_a_result_for_an_unknown_id_is_not_observed_pairing():
    e = episode_from_eventlog({"id": "e", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [
            {"tool_call_id": "7", "tool_name": "add", "parameters": {}}]}}},
        {"kind": "tool_result", "payload": {"tool_call_id": "9", "output": "2"}}]})
    assert e.coverage["call_result_pairing"] != "observed"


# -- G8: a parameterless Cohere call is a call ------------------------------------------------------------------

def test_g8_a_parameterless_cohere_call_reaches_the_trace():
    e = episode_from_eventlog({"id": "e", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [{"tool_call_id": "0", "tool_name": "get_time"}]}}},
        {"kind": "text", "payload": {"text": "noon"}}]})
    assert e.to_trace()["tool_calls"] == [[{"tool_call_id": "0", "tool_name": "get_time"}]]


def test_g8_a_flat_parameterless_cohere_action_is_a_call(tmp_path):
    import json

    from auditkit import load_agenttune
    from auditkit.agent_eval import episodes_from_agenttune
    e = episode_from_eventlog({"id": "e", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_name": "get_time"}}},
        {"kind": "text", "payload": {"text": "noon"}}]})
    assert e.to_trace()["tool_calls"] == [[{"tool_name": "get_time"}]] and e.counters["n_tool_calls"] == 1
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"trajectory_id": "t", "task": "time?", "final_response": "noon",
                             "steps": [{"action": {"tool_name": "get_time"}}, {"action": {}}],
                             "metadata": {}}) + "\n")
    [ep] = episodes_from_agenttune(str(p))
    assert [[c.get("tool_name") for c in t] for t in ep.turns()] == [["get_time"]]
    [sample] = load_agenttune(str(p))
    assert sample.actual_trace["tool_calls"] == [[{"tool_name": "get_time"}]]


def test_g8_a_store_export_row_is_still_not_a_call():
    assert not is_call_payload({"tool_name": "search", "query": "{'q': 'x'}", "result": "..."})
    assert is_call_payload({"tool_name": "get_time"}) and is_call_payload({"tool_name": "f", "parameters": {}})


# -- G9: Command-R's directly-answer is not a call ----------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ('Action: ```json\n[{"tool_name": "directly-answer", "parameters": {}}]\n```', []),
    ('<|START_ACTION|>[{"tool_name": "directly_answer", "parameters": {}}]<|END_ACTION|>', []),
    ('<|START_ACTION|>[{"tool_name": "directly-answer", "parameters": {}}, '
     '{"tool_name": "add", "parameters": {"a": 1}}]<|END_ACTION|>', [("add", {"a": 1}, None, None)]),
])
def test_g9_directly_answer_is_filtered(text, want):
    assert calls(text) == want


@pytest.mark.parametrize("text", [
    '[{"tool_name": "directly-answer", "parameters": {}}]',                              # bare JSON list
    '{"tool_name": "directly-answer", "parameters": {}}',                                # bare JSON object
    'Here you go:\n```json\n{"tool_name": "directly_answer", "parameters": {}}\n```',    # fenced block
    '<tool_call>{"name": "directly-answer", "arguments": {}}</tool_call>',
])
def test_g9_directly_answer_is_filtered_on_every_text_path(text):
    assert calls(text) == []


def test_g9_directly_answer_is_filtered_from_a_structured_trace():
    from auditkit.trace import predicted_turns
    da = {"tool_name": "directly-answer", "parameters": {}}
    add = {"tool_name": "add", "parameters": {"a": 1}}
    assert predicted_turns("", {"trace": {"tool_calls": [[da]]}}) == []
    got = predicted_turns("", {"trace": {"tool_calls": [[da, add]]}})
    assert [[c.name for c in t] for t in got] == [["add"]]
    msgs = [{"role": "assistant", "tool_calls": [{"type": "function", "function": {
        "name": "directly_answer", "arguments": "{}"}}]}]
    assert predicted_turns("", {"trace": {"messages": msgs}}) == []


# -- G10: a truncated action list is reported, not dropped ----------------------------------------------------------

@pytest.mark.parametrize("text", [
    '<|START_ACTION|>[{"tool_name": "a", "parameters": {}}, {"tool_name": "b", "parameters": {"x": 1',
    'Action: ```json\n[{"tool_name": "a", "parameters": {}}, {"tool_name": "b", "param',
])
def test_g10_a_truncated_action_list_keeps_its_complete_calls(text):
    got = calls(text)
    assert got[0] == ("a", {}, None, None) and got[1][0] == "b" and got[1][3] == "truncated"


@pytest.mark.parametrize("text", ["The answer is [5].", 'I considered tool_name "add" but answered directly.',
                                  "<|START_RESPONSE|>5<|END_RESPONSE|>"])
def test_g10_plain_answers_are_still_not_calls(text):
    assert calls(text) == []


# -- G11: JSON-string arguments are decoded before the chat template --------------------------------------------------

HISTORY = [{"role": "user", "content": "2+3?"},
           {"role": "assistant", "content": "", "tool_calls": [
               {"id": "call_9", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
           {"role": "tool", "tool_call_id": "call_9", "content": "5"}]


def test_g11_template_messages_decodes_string_arguments_without_mutating():
    out = template_messages(HISTORY)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"a": 2, "b": 3}
    assert HISTORY[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'   # input untouched
    assert template_messages([{"role": "assistant", "tool_calls": [
        {"function": {"name": "f", "arguments": "not json"}}]}])[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def _tokenizer(name, template=None):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"{name} tokenizer not cached")
    if template:
        tok.chat_template = (TEMPLATES / template).read_text()
    return tok


class _Stub:
    def __init__(self, tok):
        self.tokenizer, self.calls = tok, []

    def __call__(self, prompts, **kw):
        self.calls.append((list(prompts), kw))
        return [[{"generated_text": "ok"}] for _ in prompts]


def _hf(tok):
    from auditkit.model.hf_gen import HFGenModel
    m = HFGenModel("stub")
    m._pipeline = _Stub(tok)
    return m


def test_g11_command_r7b_renders_openai_history_parameters_as_an_object():
    m = _hf(_tokenizer("gpt2", "command_r7b_tool_use.jinja"))
    tools = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]
    m.generate([Request(prompt="2+3?", params={"messages": HISTORY, "tools": tools})])
    prompt = m._pipeline.calls[-1][0][0]
    assert '"parameters": {"a": 2, "b": 3}' in prompt and '\\"a\\"' not in prompt


# -- G12: every prompt of a mixed batch gets exactly one BOS ------------------------------------------------------------

def test_g12_a_mixed_batch_gives_every_prompt_one_bos():
    tok = _tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja")
    m = _hf(tok)
    out = m.generate([Request(prompt="raw", params={"apply_chat_template": False}), Request(prompt="chat"),
                      Request(prompt="raw2", params={"apply_chat_template": False})])
    counts = [tok(p, add_special_tokens=kw["add_special_tokens"])["input_ids"].count(tok.bos_token_id)
              for prompts, kw in m._pipeline.calls for p in prompts]
    assert len(out) == 3 and all(r.completions[0].text == "ok" for r in out) and counts == [1, 1, 1]
    assert len(m._pipeline.calls) == 2                     # one call per group, results in request order
