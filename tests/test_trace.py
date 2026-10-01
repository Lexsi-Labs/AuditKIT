"""Tests for auditkit.trace: call shapes, turn grouping, text parsing, matching."""

from __future__ import annotations

import json

import pytest

from auditkit.trace import (
    ToolCall, args_match, call_matches, max_matching, normalize_call, parse_tool_calls,
    predicted_turns, render_trace, to_turns,
)


def call(name, **args):
    return {"name": name, "arguments": args}


def names(turns):
    return [[c.name for c in t] for t in turns]


# -- normalize_call -----------------------------------------------------------

def test_openai_entry_with_json_string_arguments():
    c = normalize_call({"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}})
    assert (c.name, c.arguments, c.id, c.parse_error) == ("get_weather", {"city": "Paris"}, "call_1", None)


def test_openai_entry_with_dict_and_empty_arguments():
    assert normalize_call({"function": {"name": "f", "arguments": {"a": 1}}}).arguments == {"a": 1}
    assert normalize_call({"function": {"name": "f", "arguments": ""}}).arguments == {}
    assert normalize_call({"function": {"name": "f"}}).arguments == {}


@pytest.mark.parametrize("key", ["arguments", "args", "parameters", "input"])
def test_flat_call_argument_keys(key):
    c = normalize_call({"name": "f", key: {"x": 1}})
    assert (c.name, c.arguments) == ("f", {"x": 1})


def test_flat_call_with_json_string_arguments():
    assert normalize_call({"name": "f", "arguments": '{"x": [1, 2]}'}).arguments == {"x": [1, 2]}


def test_bfcl_ground_truth_shape():
    c = normalize_call({"calc_area": {"base": [10], "height": [5]}})
    assert (c.name, c.arguments) == ("calc_area", {"base": [10], "height": [5]})


def test_toolcall_passthrough():
    c = ToolCall("f", {"a": 1})
    assert normalize_call(c) is c


def test_malformed_json_arguments_set_parse_error():
    c = normalize_call({"function": {"name": "f", "arguments": '{"city": "Par'}})
    assert c.name == "f" and c.arguments == {} and "not valid JSON" in c.parse_error
    c = normalize_call({"name": "f", "arguments": "[1, 2]"})
    assert "list" in c.parse_error
    c = normalize_call({"name": "f", "arguments": 7})
    assert "unsupported type" in c.parse_error
    assert c.to_dict()["parse_error"] == c.parse_error


@pytest.mark.parametrize("bad", ["f", 3, None, ["f"], {"a": 1, "b": 2}])
def test_unreadable_shapes_raise(bad):
    with pytest.raises(ValueError):
        normalize_call(bad)


@pytest.mark.parametrize("bad", [{"answer": "Paris"}, {"answer": 42}, {"ok": True}, {"items": [1, 2]}])
def test_single_key_non_call_is_not_a_bfcl_call(bad):
    # A one-key JSON *answer* must not be read as a BFCL call {"fn": {args}}.
    with pytest.raises(ValueError):
        normalize_call(bad)


def test_parse_error_call_never_matches_even_in_name_mode():
    bad = normalize_call({"name": "f", "arguments": "{oops"})
    ref = ToolCall("f", {})
    for mode in ("exact", "subset", "name"):
        assert not call_matches(bad, ref, mode)
    assert not call_matches(bad, ref, arg_match=lambda n, p, r: True)
    assert max_matching([bad], [ref], "name") == {}


# -- to_turns -----------------------------------------------------------------

def test_to_turns_empty_shapes():
    assert to_turns(None) == [] and to_turns([]) == [] and to_turns({"tool_calls": []}) == []


def test_to_turns_list_of_turns_kept_and_empty_turns_dropped():
    turns = to_turns([[call("a"), call("b")], [], [call("c")]])
    assert names(turns) == [["a", "b"], ["c"]]


def test_to_turns_flat_list_is_one_turn():
    assert names(to_turns([call("a"), call("b"), call("c")])) == [["a", "b", "c"]]


def test_to_turns_single_call_dict():
    assert names(to_turns(call("a", x=1))) == [["a"]]


def test_to_turns_openai_messages_grouped_by_assistant_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather in Paris and Rome, then book"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}},
            {"id": "2", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Rome"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "2", "content": "rain"},
        {"role": "assistant", "content": "thinking", "tool_calls": None},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "3", "type": "function", "function": {"name": "book", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "3", "content": "ok"},
        {"role": "assistant", "content": "Booked Paris."},
    ]
    turns = to_turns(msgs)
    assert names(turns) == [["weather", "weather"], ["book"]]
    assert [c.arguments["city"] for c in turns[0]] == ["Paris", "Rome"]
    assert names(to_turns({"messages": msgs})) == names(turns)
    assert names(to_turns(msgs[2])) == [["weather", "weather"]]  # one message dict


def test_to_turns_tool_calls_wrapper():
    assert names(to_turns({"tool_calls": [[call("a")], [call("b")]]})) == [["a"], ["b"]]


def test_to_turns_rejects_mixed_list():
    with pytest.raises(ValueError):
        to_turns([[call("a")], call("b")])


# -- parse_tool_calls ---------------------------------------------------------

def test_multiple_hermes_blocks_are_parallel_calls():
    text = ('<tool_call>\n{"name": "weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "weather", "arguments": {"city": "Rome"}}\n</tool_call>')
    calls = parse_tool_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [("weather", {"city": "Paris"}), ("weather", {"city": "Rome"})]
    assert predicted_turns(text, None) == [calls]  # one turn, two calls


def test_hermes_block_holding_a_list():
    text = '<tool_call>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]</tool_call>'
    assert [c.name for c in parse_tool_calls(text)] == ["a", "b"]


def test_unparseable_block_among_valid_ones_is_kept_as_error_call():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {broken</tool_call>'
            '<tool_call>{"name": "c", "arguments": {"x": 1}}</tool_call>')
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "", "c"]
    assert calls[1].parse_error and "unparseable" in calls[1].parse_error
    assert calls[0].parse_error is None and calls[2].arguments == {"x": 1}


def test_hermes_block_with_string_arguments():
    calls = parse_tool_calls('<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call>')
    assert calls[0].arguments == {"x": 1}


@pytest.mark.parametrize("text", ["", "The capital of France is Paris.",
                                  "I can't help with that. [citation needed]",
                                  "{not json at all", "[1, 2, 3]", '["a", "b"]',
                                  '{"answer": "Paris"}', '{"answer": 42}',
                                  '{"answer": "Paris", "confidence": 0.9}',
                                  "```json\n{\"answer\": \"Paris\"}\n```",
                                  "```python\nprint('hi')\n```"])
def test_text_without_calls_parses_to_nothing(text):
    assert parse_tool_calls(text) == []
    assert predicted_turns(text, None) == []


def test_bare_json_object_and_list():
    assert [c.name for c in parse_tool_calls('  {"name": "f", "arguments": {"a": 1}}  ')] == ["f"]
    assert [c.name for c in parse_tool_calls('[{"name": "f"}, {"name": "g", "args": {"b": 2}}]')] == ["f", "g"]
    assert [c.name for c in parse_tool_calls('{"tool_calls": [{"name": "f"}, {"name": "g"}]}')] == ["f", "g"]


def test_fenced_json_after_prose():
    text = 'I will look it up.\n```json\n{"name": "search", "arguments": {"q": "x"}}\n```\nDone.'
    assert [(c.name, c.arguments) for c in parse_tool_calls(text)] == [("search", {"q": "x"})]


def test_think_block_is_stripped():
    text = ('<think>maybe <tool_call>{"name": "wrong", "arguments": {}}</tool_call></think>\n'
            '<tool_call>{"name": "right", "arguments": {}}</tool_call>')
    assert [c.name for c in parse_tool_calls(text)] == ["right"]
    # thinking-only reply with a draft call inside -> no calls
    assert parse_tool_calls('<think><tool_call>{"name": "x"}</tool_call></think>The answer is 4.') == []
    # bare JSON after a think block
    assert [c.name for c in parse_tool_calls('<think>hmm</think>\n{"name": "f", "arguments": {}}')] == ["f"]


# -- predicted_turns ----------------------------------------------------------

def test_trace_tool_calls_win_over_text():
    text = '<tool_call>{"name": "from_text", "arguments": {}}</tool_call>'
    ctx = {"trace": {"tool_calls": [[call("a")], [call("b")]]}}
    assert names(predicted_turns(text, ctx)) == [["a"], ["b"]]


def test_trace_with_empty_tool_calls_means_no_calls_not_text_fallback():
    text = '<tool_call>{"name": "from_text", "arguments": {}}</tool_call>'
    assert predicted_turns(text, {"trace": {"tool_calls": []}}) == []


def test_trace_messages_used_when_no_tool_calls_key():
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": "{}"}}]},
            {"role": "tool", "content": "r"}]
    assert names(predicted_turns("", {"trace": {"messages": msgs}})) == [["a"]]


def test_trace_without_calls_falls_back_to_text():
    text = '<tool_call>{"name": "t", "arguments": {}}</tool_call>'
    assert names(predicted_turns(text, {"trace": {"retrieved_contexts": ["c"]}})) == [["t"]]
    assert names(predicted_turns(text, {"trace": "not a dict"})) == [["t"]]
    assert names(predicted_turns(text, None)) == [["t"]]


# -- matching -----------------------------------------------------------------

def _greedy(preds, refs, **kw):
    used, n = set(), 0
    for r in refs:
        for j, p in enumerate(preds):
            if j not in used and call_matches(p, r, **kw):
                used.add(j)
                n += 1
                break
    return n


def test_max_matching_beats_greedy_on_bfcl_allowed_values():
    # BFCL-style ground truth lists allowed values per argument.
    allowed = lambda name, pred, ref: all(pred.get(k) in v for k, v in ref.items())  # noqa: E731
    refs = [ToolCall("A", {"x": [1, 2]}), ToolCall("A", {"x": [1]})]
    preds = [ToolCall("A", {"x": 1}), ToolCall("A", {"x": 2})]
    assert _greedy(preds, refs, arg_match=allowed) == 1  # greedy pairs ref0 with A(1) and strands ref1
    m = max_matching(preds, refs, arg_match=allowed)
    assert m == {0: 1, 1: 0}


def test_max_matching_beats_greedy_in_subset_mode():
    refs = [ToolCall("A", {"x": 1}), ToolCall("A", {"x": 1, "y": 2})]
    preds = [ToolCall("A", {"x": 1, "y": 2}), ToolCall("A", {"x": 1})]  # pred0 fits both refs
    assert _greedy(preds, refs, mode="subset") == 1
    assert max_matching(preds, refs, "subset") == {0: 1, 1: 0}


def test_max_matching_is_one_to_one_with_duplicates():
    a = ToolCall("A", {"x": 1})
    assert len(max_matching([a, a, a], [a])) == 1
    assert len(max_matching([a], [a, a, a])) == 1
    assert max_matching([], [a]) == {} and max_matching([a], []) == {}


def test_args_match_modes():
    assert args_match({"x": 1}, {"x": 1.0})
    assert not args_match({"x": 1, "y": 2}, {"x": 1})
    assert args_match({"x": 1, "y": 2}, {"x": 1}, "subset")
    assert not args_match({"y": 2}, {"x": 1}, "subset")
    assert not args_match({"x": 2}, {"x": 1}, "subset")
    assert args_match({"z": 9}, {"x": 1}, "name")
    assert not call_matches(ToolCall("B", {}), ToolCall("A", {}), "name")
    with pytest.raises(ValueError):
        args_match({}, {}, "fuzzy")


# -- render_trace -------------------------------------------------------------

def test_render_trace_turns():
    text = render_trace([[ToolCall("a", {"x": 1}), ToolCall("b")], [ToolCall("c")]])
    lines = text.splitlines()
    assert lines[0].startswith("[turn 1]") and '"name": "a"' in lines[0] and '"x": 1' in lines[0]
    assert '"name": "b"' in lines[0] and lines[1].startswith("[turn 2]")
    assert render_trace([]) == "(no tool calls)"


def test_render_trace_prefers_messages_and_includes_tool_results():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "lookup", "arguments": '{"id": 7}'}}]},
            {"role": "tool", "content": "RESULT-42"},
            {"role": "assistant", "content": [{"type": "text", "text": "final"}]}]
    text = render_trace([], {"messages": msgs})
    assert "[assistant] tool_calls:" in text and '"lookup"' in text and '"id": 7' in text
    assert "[tool] RESULT-42" in text and "[assistant] final" in text


def test_render_trace_truncates_keeping_head_and_tail():
    # A long single turn: head-only truncation would drop the tail; keep both.
    text = render_trace([[ToolCall("f", {"head": "H" * 300, "tail": "T" * 300})]], limit=100)
    assert len(text) < 130 and "...(truncated)..." in text
    assert "H" in text and "T" in text  # both ends survive


def test_render_trace_messages_with_no_tool_calls_gets_marker():
    msgs = [{"role": "user", "content": "book it"},
            {"role": "assistant", "content": "Done! booked, confirmation XK42PZ"}]
    text = render_trace([], {"messages": msgs})
    assert "(no tool calls)" in text and "XK42PZ" in text


def test_to_dict_roundtrip_is_json():
    c = ToolCall("f", {"a": [1, {"b": None}]}, id="c1")
    assert json.loads(json.dumps(c.to_dict())) == {"name": "f", "arguments": {"a": [1, {"b": None}]}, "id": "c1"}


# -- follow-up fixes (lead review) ---------------------------------------------

def test_bool_is_not_int_in_argument_matching():
    from auditkit.trace import args_match
    assert not args_match({"x": True}, {"x": 1})
    assert not args_match({"x": [True]}, {"x": [1]}, "subset")
    assert args_match({"x": 1.0, "y": {"z": [1, 2]}}, {"x": 1, "y": {"z": [1, 2]}})


def test_json_string_reference_is_decoded_not_treated_as_no_calls():
    from auditkit.trace import to_turns
    turns = to_turns('[{"name": "a", "arguments": {"x": 1}}]')
    assert [[c.name for c in t] for t in turns] == [["a"]]
    with pytest.raises(ValueError):
        to_turns("not json")


def test_lone_closing_think_tag_drops_draft_calls():
    from auditkit.trace import parse_tool_calls
    text = 'draft: {"name": "wrong", "arguments": {}}</think>{"name": "right", "arguments": {}}'
    assert [c.name for c in parse_tool_calls(text)] == ["right"]


def test_strip_thinking_helper():
    from auditkit.trace import strip_thinking
    assert strip_thinking("<think>reason</think>answer") == "answer"
    assert strip_thinking("reason</think>answer") == "answer"  # template-opened block
    assert strip_thinking("no tags here") == "no tags here"


def test_unclosed_tool_call_block_still_parses_its_call():
    # command-r7b and similar emit an opening <tool_call> with no close tag.
    text = '<tool_call>{"name": "get_time", "arguments": {"timezone": "Asia/Kolkata"}}'
    calls = parse_tool_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [("get_time", {"timezone": "Asia/Kolkata"})]


def test_mixed_closed_then_unclosed_blocks():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}')  # second block unclosed
    assert [c.name for c in parse_tool_calls(text)] == ["a", "b"]


def test_prose_mentioning_tool_call_tag_without_json_stays_empty():
    assert parse_tool_calls("Use the <tool_call> block to call a tool.") == []


def test_unclosed_block_with_whitespace_flood_is_fast():
    import time
    # The old \s*(.*?)\s* regex backtracked cubically here (minutes at this size).
    text = '<tool_call>{"name": "f", "arguments": {"q": "x"}}\n' + "\n" * 20000
    t0 = time.perf_counter()
    calls = parse_tool_calls(text)
    assert time.perf_counter() - t0 < 2.0
    assert [c.name for c in calls] == ["f"]


@pytest.mark.parametrize("text", [
    '{"name": "Marie Curie", "born": 1867}',                      # structured answer
    '[{"name": "Mercury", "order": 1}, {"name": "Venus", "order": 2}]',
    '```json\n{"name": "Paris", "population": 2161000}\n```',
])
def test_structured_json_answer_with_name_is_not_a_call(text):
    assert parse_tool_calls(text) == []


# -- concatenated bare-JSON parallel calls (live qwen2.5-coder gap) -------------

def test_concatenated_bare_json_objects_parse_as_parallel_calls():
    from auditkit.trace import parse_tool_calls
    for text in (
        '{"name":"get_weather","arguments":{"city":"Paris"}}\n{"name":"get_weather","arguments":{"city":"Tokyo"}}',
        '{"name":"a","arguments":{}}{"name":"b","arguments":{}}',
        '{"name":"a","arguments":{}}, {"name":"b","arguments":{}}, {"name":"c","arguments":{}}',
    ):
        names = [c.name for c in parse_tool_calls(text)]
        assert len(names) >= 2 and names[0] and names[-1]
    # single object and JSON array unchanged
    assert [c.name for c in parse_tool_calls('{"name":"only","arguments":{"x":1}}')] == ["only"]
    assert [c.name for c in parse_tool_calls('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]')] == ["a", "b"]
    # no false positives on non-call JSON or prose
    assert parse_tool_calls('{"answer":"Paris"}\n{"answer":"Rome"}') == []
    assert parse_tool_calls('The weather in Paris is sunny.') == []
    assert parse_tool_calls('{this is prose, not json}') == []


# -- edge cases: model-controlled / malformed input must not hang or crash -----

def test_repeated_unclosed_tool_call_tags_are_fast():
    import time
    # A repetition-loop model output: K opens, no close. The old lazy .*? restarted
    # a scan-to-end at each open -> O(K^2) (~23s at K=20000). Tempered dot -> linear.
    t0 = time.perf_counter()
    assert parse_tool_calls("<tool_call>" * 20000) == []
    assert time.perf_counter() - t0 < 1.0
    # one real call then the loop still recovers the call (an early-out would miss it)
    t0 = time.perf_counter()
    calls = parse_tool_calls('<tool_call>{"name": "f", "arguments": {}}</tool_call>' + "<tool_call>" * 20000)
    assert [c.name for c in calls] == ["f"] and time.perf_counter() - t0 < 1.0


def test_repeated_unclosed_think_tags_are_fast():
    import time
    from auditkit.trace import strip_thinking
    # <think>*K with no close was O(K^2) in re.sub (~6s at K=20000).
    t0 = time.perf_counter()
    assert strip_thinking("<think>" * 20000) == "<think>" * 20000  # nothing closed -> unchanged
    assert time.perf_counter() - t0 < 1.0


def test_unclosed_fence_with_whitespace_flood_is_fast():
    import time
    # _FENCE_RE kept a \s* the sibling regex dropped -> quadratic backtracking on an
    # unclosed fence + long whitespace run (~15s at 80k). Content is .strip()ed
    # downstream, so dropping \s* is output-preserving.
    t0 = time.perf_counter()
    assert parse_tool_calls("```" + " " * 80000) == []
    assert time.perf_counter() - t0 < 1.0
    # a normal fenced block (with whitespace after ```json) still parses
    assert [c.name for c in parse_tool_calls('```json\n  \n{"name": "s", "arguments": {"q": "x"}}\n```')] == ["s"]


def test_deeply_nested_args_compare_without_recursion_error():
    # json.loads accepts far deeper nesting than a recursive compare survived, so a
    # trace that parses (parse_error=None) must stay comparable. _json_eq is iterative.
    deep = lambda leaf: json.loads('{"a":' * 600 + leaf + "}" * 600)  # noqa: E731
    assert args_match(deep("null"), deep("null"))
    assert args_match(deep("null"), deep("null"), "subset")
    assert not args_match(deep("1"), deep("2"))  # a mismatch at the bottom is still found


def test_max_matching_large_reference_no_recursion_error():
    # ~1000+ same-named refs chain the augmenting path through all of them; the
    # recursive augment overflowed the stack. Iterative _augment handles it.
    n = 1000
    m = max_matching([ToolCall("f") for _ in range(n)], [ToolCall("f") for _ in range(n)])
    assert len(m) == n


# -- stress regressions: model-controlled text must parse, never crash ----------

def test_deep_nesting_in_model_text_never_raises_recursion_error():
    # '[' * ~1000 raised RecursionError (a RuntimeError), escaping every
    # `except ValueError`, so the metric crashed and the sample left its stat.
    deep = "[" * 5000
    for text in (deep, '{"a":' * 5000, "<tool_call>" + deep, "```json\n" + deep + "\n```",
                 '{"name": "f"}\n' + deep):
        assert parse_tool_calls(text) == []
    # a closed garbage block stays an error call, as before
    bad = parse_tool_calls("<tool_call>" + deep + "</tool_call>")
    assert len(bad) == 1 and bad[0].parse_error
    # a native arguments string: a parse_error, not an exception
    tc = normalize_call({"function": {"name": "f", "arguments": '{"x": ' + deep + "}"}})
    assert tc.parse_error and tc.arguments == {}
    # reference side: ValueError (the documented error), not RecursionError
    with pytest.raises(ValueError):
        to_turns(deep)
    # valid JSON just under the interpreter limit parsed, then crashed json.dumps /
    # the saved run downstream: nesting past the cap is a parse error up front
    ok = "[" * 150 + "]" * 150
    tc = normalize_call({"name": "f", "arguments": '{"x": %s}' % ok})
    assert tc.parse_error and "nested" in tc.parse_error
    calls = parse_tool_calls('<tool_call>{"name": "f", "arguments": {"x": %s}}</tool_call>' % ok)
    assert len(calls) == 1 and calls[0].parse_error
    shallow = "[" * 20 + "]" * 20  # ordinary nesting still reads
    assert parse_tool_calls('{"name": "f", "arguments": {"x": %s}}' % shallow)[0].parse_error is None


@pytest.mark.parametrize("arg", ["see </think> tag", "<think>x</think>y", "use <tool_call> here",
                                 "end </tool_call> mark", 'a <tool_call>{"name": "g"}</tool_call> b'])
def test_tag_literals_inside_argument_strings_are_just_text(arg):
    # The regex parser cut blocks at tag literals inside JSON strings, and
    # strip_thinking ran before parsing: calls were dropped, turned into error
    # calls, or had their arguments silently rewritten.
    call = {"name": "echo", "arguments": {"text": arg}}
    for text in ("<tool_call>" + json.dumps(call) + "</tool_call>",
                 "<tool_call>" + json.dumps(call),  # unclosed
                 json.dumps(call),  # bare JSON
                 "<think>plan</think>\n<tool_call>" + json.dumps(call) + "</tool_call>"):
        got = parse_tool_calls(text)
        assert [(c.name, c.arguments, c.parse_error) for c in got] == [("echo", {"text": arg}, None)], text


def test_tag_scanner_keeps_reasoning_and_block_semantics():
    f = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    g = '<tool_call>{"name": "g", "arguments": {}}</tool_call>'
    assert [c.name for c in parse_tool_calls("<think>draft " + g + "</think>" + f)] == ["f"]
    assert [c.name for c in parse_tool_calls("draft " + g + "</think>" + f)] == ["f"]  # template-opened
    assert [c.name for c in parse_tool_calls("<think>truncated " + g)] == ["g"]  # never closed: kept
    assert [c.name for c in parse_tool_calls(f + g)] == ["f", "g"]
    assert parse_tool_calls("I would use a <tool_call> tag here.") == []
    bad = parse_tool_calls("<tool_call>not json</tool_call>")
    assert len(bad) == 1 and bad[0].name == "" and "not json" in bad[0].parse_error


def test_nan_arguments_equal_themselves():
    # nan != nan made a call built from '{"x": NaN}' never match itself, while
    # RedundantToolCalls (json.dumps -> "NaN") counted it as a duplicate.
    nan = float("nan")
    assert args_match({"x": nan}, {"x": nan}) and args_match({"x": [nan]}, {"x": [nan]}, "subset")
    assert not args_match({"x": nan}, {"x": 1.0})
    p = parse_tool_calls('<tool_call>{"name": "f", "arguments": {"x": NaN}}</tool_call>')[0]
    assert len(max_matching([p], [ToolCall("f", {"x": nan})])) == 1


def test_to_dict_arguments_are_strict_json():
    # NaN / 1e999 (inf) are accepted from model text; stored in score metadata they
    # made the saved run.json unreadable to strict JSON parsers.
    tc = parse_tool_calls('<tool_call>{"name": "f", "arguments": {"x": 1e999, "y": NaN, "z": [1]}}</tool_call>')[0]
    d = tc.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["arguments"] == {"x": None, "y": None, "z": [1]}


def test_max_matching_identical_calls_not_cubic():
    import time
    # Plain Kuhn chained through every earlier owner: 2000 identical calls took ~24s.
    n = 2000
    t0 = time.perf_counter()
    assert len(max_matching([ToolCall("f") for _ in range(n)], [ToolCall("f") for _ in range(n)])) == n
    # more refs than preds: every leftover ref fails its augment; identical ones skip it
    assert len(max_matching([ToolCall("f") for _ in range(n // 2)], [ToolCall("f") for _ in range(n)])) == n // 2
    # name mode, same tool with different args
    preds = [ToolCall("f", {"i": i}) for i in range(n)]
    refs = [ToolCall("f", {"i": -i}) for i in range(n)]
    assert len(max_matching(preds, refs, "name")) == n
    assert time.perf_counter() - t0 < 10.0  # ~3s; was minutes
