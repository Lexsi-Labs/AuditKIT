"""How the agent's calls reach the metrics: every trace shape and text format.

Each text case is a raw model reply scored with no structured trace, so the
metrics fall back to parse_tool_calls() on the output. The expectation is the
turns that should come out (always one turn for text).
"""

from __future__ import annotations

import json

import pytest

from auditkit.metrics.agent import ParallelToolCalls, ToolCallF1
from auditkit.sample import Sample
from auditkit.trace import predicted_turns, to_turns

from .catalog import T, W, c

P, R = W("Paris"), W("Rome")


def names(turns):
    """Comparable form of turns given as ToolCall objects or raw call dicts."""
    return [[(x.name, x.arguments) for x in t] for t in to_turns(turns)]


def hermes(*calls):
    return "\n".join(f"<tool_call>{json.dumps(x)}</tool_call>" for x in calls)


TEXT_CASES = [
    ("hermes-single", hermes(P), [[P]]),
    ("hermes-two-blocks-one-turn", hermes(P, R), [[P, R]]),
    ("hermes-with-prose", "Let me check.\n" + hermes(P) + "\nOne moment.", [[P]]),
    ("think-then-call", "<think>maybe Rome?</think>" + hermes(P), [[P]]),
    ("call-only-inside-think", "<think>" + hermes(P) + "</think>I'll answer directly: sunny.", []),
    ("lone-closing-think", "draft " + hermes(R) + "</think>" + hermes(P), [[P]]),
    ("unclosed-tool-call", '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}', [[P]]),
    ("bare-json-object", json.dumps(P), [[P]]),
    ("bare-json-list", json.dumps([P, R]), [[P, R]]),
    ("concatenated-bare-json", json.dumps(P) + "\n" + json.dumps(R), [[P, R]]),
    ("concatenated-no-separator", json.dumps(P) + json.dumps(R), [[P, R]]),
    ("fenced-json", "Here you go:\n```json\n" + json.dumps(P) + "\n```", [[P]]),
    ("tool_calls-wrapper", json.dumps({"tool_calls": [P, R]}), [[P, R]]),
    ("openai-entry-string-args", json.dumps({"type": "function", "function": {"name": "get_weather",
                                                                            "arguments": '{"city": "Paris"}'}}), [[P]]),
    ("alias-key-args", json.dumps({"name": "get_weather", "args": {"city": "Paris"}}), [[P]]),
    ("alias-key-parameters", json.dumps({"name": "get_weather", "parameters": {"city": "Paris"}}), [[P]]),
    ("bfcl-shape", json.dumps({"get_weather": {"city": "Paris"}}), [[P]]),
    ("plain-prose", "It is sunny in Paris.", []),
    ("json-answer-not-a-call", json.dumps({"answer": "Paris"}), []),
    ("json-with-name-key-not-a-call", json.dumps({"name": "Paris", "population": 2161000}), []),
    ("prose-starting-with-brace", "{this is not json} but text", []),
    ("empty-output", "", []),
]


@pytest.mark.parametrize("text,want", [x[1:] for x in TEXT_CASES], ids=[x[0] for x in TEXT_CASES])
def test_text_output_parsing(text, want):
    assert names(predicted_turns(text, {})) == names(to_turns(want))


def test_bad_block_becomes_a_parse_error_call():
    turns = predicted_turns("<tool_call>{not json}</tool_call>", {})
    assert len(turns) == 1 and turns[0][0].parse_error


def test_many_unclosed_tags_stay_fast():
    import time
    t0 = time.time()
    predicted_turns("<tool_call>{" * 5000, {})
    assert time.time() - t0 < 2.0


# -- structured trace shapes --------------------------------------------------------

def oa(i, call):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}


def test_messages_transcript_one_turn_per_assistant_message():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [oa(1, P), oa(2, R)]},
            {"role": "tool", "tool_call_id": "c1", "content": "14C"},
            {"role": "tool", "tool_call_id": "c2", "content": "23C"},
            {"role": "assistant", "content": None, "tool_calls": [oa(3, T("Rome"))]},
            {"role": "tool", "tool_call_id": "c3", "content": "12:00"},
            {"role": "assistant", "content": "Rome, 12:00"}]
    assert names(predicted_turns("Rome, 12:00", {"trace": {"messages": msgs}})) == names([[P, R], [T("Rome")]])


def test_tool_calls_wins_over_messages():
    ctx = {"trace": {"tool_calls": [[P]], "messages": [{"role": "assistant", "tool_calls": [oa(1, R)]}]}}
    assert names(predicted_turns("", ctx)) == names([[P]])


def test_structured_trace_wins_over_text():
    assert names(predicted_turns(hermes(R), {"trace": {"tool_calls": [[P]]}})) == names([[P]])


def test_empty_structured_trace_means_no_calls_not_text_fallback():
    # tool_calls=[] is an explicit "no calls", so text is not parsed
    assert predicted_turns(hermes(P), {"trace": {"tool_calls": []}}) == []


REF_SHAPES = [
    ("turns", [[P, R], [T("Rome")]], [[P, R], [T("Rome")]]),
    ("flat-list-is-one-turn", [P, R], [[P, R]]),
    ("single-dict", P, [[P]]),
    ("json-string", json.dumps([[P, R]]), [[P, R]]),
    ("empty-list", [], []),
    ("none", None, []),
    ("empty-inner-turns-dropped", [[P], [], [R]], [[P], [R]]),
    ("wrapper-dict", {"tool_calls": [[P]]}, [[P]]),
]


@pytest.mark.parametrize("ref,want", [x[1:] for x in REF_SHAPES], ids=[x[0] for x in REF_SHAPES])
def test_reference_shapes(ref, want):
    assert names(to_turns(ref)) == names(to_turns(want))


def test_unparseable_reference_string_raises():
    with pytest.raises(ValueError):
        to_turns("not json at all")


def test_same_behaviour_scored_identically_across_formats():
    """One parallel answer, four wire formats -> identical scores."""
    ref = [[P, R]]
    s = Sample(input="t", expected_tool_calls=ref)
    variants = {
        "native": ("", {"trace": {"tool_calls": [[oa(1, P), oa(2, R)]]}}),
        "transcript": ("", {"trace": {"messages": [{"role": "assistant", "tool_calls": [oa(1, P), oa(2, R)]}]}}),
        "hermes-text": (hermes(P, R), {}),
        "bare-json-text": (json.dumps(P) + json.dumps(R), {}),
    }
    results = {}
    for k, (out, ctx) in variants.items():
        got = [x for m in (ToolCallF1(), ParallelToolCalls()) for x in m.score(s, out, ctx)]
        results[k] = {x.name: x.value for x in got}
    first = next(iter(results.values()))
    assert all(v == first for v in results.values()), results
    assert first["parallel_recall"] == 1.0 and first["tool_call_exact"] == 1.0


def test_parse_errors_never_match_even_in_name_mode():
    s = Sample(input="t", expected_tool_calls=[[P]])
    bad = {"trace": {"tool_calls": [[{"type": "function", "function": {"name": "get_weather", "arguments": "{x"}}]]}}
    assert {x.name: x.value for x in ToolCallF1(arg_mode="name").score(s, "", bad)}["tool_call_recall_name"] == 0.0


def test_reference_with_parse_error_is_visible():
    turns = to_turns([[{"name": "f", "arguments": "{broken"}]])
    assert turns[0][0].parse_error


def test_bfcl_single_key_answer_not_confused_with_json_answer():
    assert names(predicted_turns(json.dumps({"answer": "x"}), {})) == []
    assert names(predicted_turns(json.dumps({"f": {"a": 1}}), {})) == names([[c("f", a=1)]])
