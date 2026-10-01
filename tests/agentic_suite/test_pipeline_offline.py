"""ak.evaluate() end to end for agentic metrics, through every backend path.

- model="precomputed" with recorded traces
- a text callable with ToolCallAdapter(mode="prompt")
- the real api: backend against a mock OpenAI-compatible HTTP server (native tools)
- the real agent: backend against a mock deployed agent (multi-turn transcripts)
No external network; the servers run on 127.0.0.1.
"""

from __future__ import annotations

import json
from statistics import mean

import pytest

import auditkit as ak
from auditkit.errors import CapabilityError
from auditkit.sample import Sample

from .catalog import BOOK, SEARCH, TOOLS, T, W, c

P, R = W("Paris"), W("Rome")
AGENT_METRICS = lambda: [ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls(),  # noqa: E731
                         ak.ToolCallValidity(), ak.RedundantToolCalls()]


def per_sample(result):
    return {p.sample_id: {s["name"]: s["value"] for s in p.metadata.get("scores", [])} for p in result.predictions}


def oa(i, call):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}


# -- precomputed: recorded traces ---------------------------------------------------

RECORDED = [
    # id, reference, recorded turns
    ("perfect-parallel", [[P, R], [BOOK]], [[P, R], [BOOK]]),
    ("serialized", [[P, R], [BOOK]], [[P], [R], [BOOK]]),
    ("dependent-batched", [[P, R], [BOOK]], [[P, R, BOOK]]),
    ("irrelevance-ok", [], []),
    ("irrelevance-violated", [], [[P]]),
    ("invalid-and-looping", [[SEARCH]], [[c("search_flights", origin="DEL")], [c("search_flights", origin="DEL")]]),
    ("single-ok", [[T("Rome")]], [[T("Rome")]]),
]


def recorded_samples():
    return [Sample(id=i, input=f"task {i}", tools=TOOLS, expected_tool_calls=ref, actual_output="done",
                   actual_trace={"tool_calls": turns}) for i, ref, turns in RECORDED]


def test_precomputed_scores_every_sample_without_errors():
    r = ak.evaluate(recorded_samples(), model="precomputed", scorers=AGENT_METRICS())
    assert r.errors == [] and r.failed_count == 0
    got = per_sample(r)
    assert got["perfect-parallel"]["trajectory_strict"] == 1.0
    assert got["serialized"]["parallel_recall"] == 0.0 and "parallel_precision" not in got["serialized"]
    assert got["dependent-batched"]["parallel_precision"] == 0.0
    assert got["irrelevance-violated"]["tool_call_f1"] == 0.0
    assert got["invalid-and-looping"]["tool_call_validity"] == 0.0
    assert got["invalid-and-looping"]["redundant_tool_calls"] == 0.5
    # no-call samples produce no validity/redundancy score at all
    assert "tool_call_validity" not in got["irrelevance-ok"] and "redundant_tool_calls" not in got["irrelevance-ok"]


def test_headline_is_mean_over_samples_that_have_the_score():
    r = ak.evaluate(recorded_samples(), model="precomputed", scorers=AGENT_METRICS())
    got = per_sample(r)
    for name, value in r.headline.items():
        vals = [s[name] for s in got.values() if name in s]
        assert vals, name
        assert value == pytest.approx(mean(vals)), name
    # parallel_precision only exists where the model batched >=2 matched calls
    assert sum("parallel_precision" in s for s in got.values()) == 2


def test_registry_names_work_as_scorers():
    names = ["tool_call_f1", "trajectory_match", "parallel_tool_calls", "tool_call_validity", "redundant_tool_calls"]
    r = ak.evaluate(recorded_samples(), model="precomputed", scorers=names)
    assert {"tool_call_f1", "trajectory_strict", "parallel_detection", "tool_call_validity",
            "redundant_tool_calls"} <= r.headline.keys()


def test_precomputed_transcript_messages():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [oa(1, P), oa(2, R)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "14"},
            {"role": "tool", "tool_call_id": "call_2", "content": "23"},
            {"role": "assistant", "content": None, "tool_calls": [oa(3, BOOK)]},
            {"role": "assistant", "content": "Booked."}]
    s = Sample(id="t", input="q", expected_tool_calls=[[P, R], [BOOK]], actual_output="Booked.",
               actual_trace={"messages": msgs})
    got = per_sample(ak.evaluate([s], model="precomputed", scorers=AGENT_METRICS()))["t"]
    assert got["trajectory_strict"] == 1.0 and got["parallel_recall"] == 1.0


def test_precomputed_text_only_falls_back_to_parsing():
    out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>' \
          '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>'
    s = Sample(id="t", input="q", expected_tool_calls=[[P, R]], actual_output=out)
    got = per_sample(ak.evaluate([s], model="precomputed", scorers=AGENT_METRICS()))["t"]
    assert got["parallel_recall"] == 1.0 and got["tool_call_exact"] == 1.0


def test_fingerprint_depends_on_reference():
    a = ak.evaluate([Sample(id="x", input="q", expected_tool_calls=[[P]], actual_output="")],
                    model="precomputed", scorers=[ak.ToolCallF1()])
    b = ak.evaluate([Sample(id="x", input="q", expected_tool_calls=[[R]], actual_output="")],
                    model="precomputed", scorers=[ak.ToolCallF1()])
    assert a.fingerprint != b.fingerprint


# -- prompt mode through a text callable -------------------------------------------------

def test_prompt_mode_writes_schemas_and_parses_reply():
    seen = []

    def model(prompts):
        seen.extend(prompts)
        return ['<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
                '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>'] * len(prompts)

    s = Sample(id="p", input="Compare Paris and Rome weather", tools=TOOLS, expected_tool_calls=[[P, R]])
    r = ak.evaluate([s], model=model, adapter=ak.ToolCallAdapter(mode="prompt"), scorers=AGENT_METRICS())
    got = per_sample(r)["p"]
    assert got["tool_call_exact"] == 1.0 and got["parallel_recall"] == 1.0 and got["tool_call_validity"] == 1.0
    assert "get_weather" in seen[0] and "<tool_call>" in seen[0]


def test_native_tools_refused_on_text_backend():
    s = Sample(id="p", input="q", tools=TOOLS, expected_tool_calls=[[P]])
    with pytest.raises(CapabilityError):
        ak.evaluate([s], model=lambda ps: ["x"] * len(ps), adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])


# -- api: backend against a mock OpenAI-compatible server ------------------------------

def chat_reply(content=None, calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
        finish = "tool_calls"
    return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}


API_SCRIPT = {
    # user prompt -> (reference, server reply)
    "weather Paris and Rome": ([[P, R]], chat_reply(calls=[oa(1, P), oa(2, R)])),
    "weather Paris only": ([[P]], chat_reply(calls=[oa(1, P)])),
    "serialize please": ([[P, R]], chat_reply(calls=[oa(1, P)])),
    "what is 2+2": ([], chat_reply(content="4")),
    "call when not needed": ([], chat_reply(calls=[oa(1, P)])),
    "broken arguments": ([[P]], chat_reply(calls=[{"id": "b", "type": "function",
                                                   "function": {"name": "get_weather", "arguments": "{oops"}}])),
    "hallucinated tool": ([[P]], chat_reply(calls=[oa(1, c("teleport", to="Paris"))])),
    "text-form tool call": ([[P]], chat_reply(content='<tool_call>{"name": "get_weather", '
                                                     '"arguments": {"city": "Paris"}}</tool_call>')),
}


def _api_samples():
    return [Sample(id=k, input=k, tools=TOOLS, expected_tool_calls=ref) for k, (ref, _) in API_SCRIPT.items()]


def test_api_backend_native_tools_end_to_end(openai_server):
    openai_server.respond = lambda body: (200, API_SCRIPT[body["messages"][-1]["content"]][1])
    r = ak.evaluate(_api_samples(), model="api:mock-model", api_base=openai_server.url, api_key="EMPTY",
                    adapter=ak.ToolCallAdapter(), scorers=AGENT_METRICS())
    assert r.errors == []
    # the request really carried the tool schemas
    body = openai_server.received[0]["body"]
    assert body["tools"] == TOOLS and openai_server.received[0]["path"].endswith("/chat/completions")
    got = per_sample(r)
    assert got["weather Paris and Rome"]["parallel_recall"] == 1.0
    assert got["weather Paris and Rome"]["tool_call_exact"] == 1.0
    assert got["serialize please"]["tool_call_recall"] == 0.5 and got["serialize please"]["parallel_recall"] == 0.0
    assert got["what is 2+2"]["tool_call_f1"] == 1.0
    assert got["call when not needed"]["tool_call_f1"] == 0.0
    assert got["broken arguments"]["tool_call_validity"] == 0.0 and got["broken arguments"]["tool_call_f1"] == 0.0
    assert got["hallucinated tool"]["tool_call_validity"] == 0.0
    # no native tool_calls -> the text reply is parsed instead
    assert got["text-form tool call"]["tool_call_exact"] == 1.0
    # the trace the metrics used is kept on the prediction
    pred = next(p for p in r.predictions if p.sample_id == "weather Paris and Rome")
    assert len(pred.context["trace"]["tool_calls"][0]) == 2


def test_api_backend_forwards_tool_choice_and_parallel_flag(openai_server):
    openai_server.respond = lambda body: (200, chat_reply(calls=[oa(1, P)]))
    ak.evaluate([Sample(id="a", input="q", tools=TOOLS, expected_tool_calls=[[P]])], model="api:mock-model",
                api_base=openai_server.url, api_key="EMPTY",
                adapter=ak.ToolCallAdapter(tool_choice="required", parallel_tool_calls=False),
                scorers=[ak.ToolCallF1()])
    body = openai_server.received[0]["body"]
    assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False


def test_api_backend_server_error_is_recorded_not_scored(openai_server):
    """A 5xx on one sample fails only that sample; the run finishes and no request is replayed."""
    openai_server.respond = lambda body: (500, {"error": "boom"}) if body["messages"][-1]["content"] == "bad" \
        else (200, chat_reply(calls=[oa(1, P)]))
    samples = [Sample(id=i, input=i, tools=TOOLS, expected_tool_calls=[[P]]) for i in ("ok1", "bad", "ok2")]
    r = ak.evaluate(samples, model="api:mock-model", api_base=openai_server.url, api_key="EMPTY",
                    adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()], config=ak.RunConfig(max_retries=0))
    got = per_sample(r)
    assert r.errors and "tool_call_f1" not in got.get("bad", {})
    assert got["ok1"]["tool_call_f1"] == 1.0 and got["ok2"]["tool_call_f1"] == 1.0
    bad_posts = [x for x in openai_server.received if x["body"]["messages"][-1]["content"] == "bad"]
    assert len(bad_posts) == 1                      # max_retries=0: no retry, and no batch replay
    assert sum(x["body"]["messages"][-1]["content"] == "ok1" for x in openai_server.received) == 1


def test_api_backend_transient_error_recovers_on_retry(openai_server):
    calls = {"n": 0}

    def respond(body):
        calls["n"] += 1
        return (503, {"error": "busy"}) if calls["n"] == 1 else (200, chat_reply(calls=[oa(1, P)]))
    openai_server.respond = respond
    r = ak.evaluate([Sample(id="a", input="q", tools=TOOLS, expected_tool_calls=[[P]])], model="api:mock-model",
                    api_base=openai_server.url, api_key="EMPTY", adapter=ak.ToolCallAdapter(),
                    scorers=[ak.ToolCallF1()], config=ak.RunConfig(max_retries=1, retry_delay=0.0))
    assert r.errors == [] and per_sample(r)["a"]["tool_call_f1"] == 1.0
    assert calls["n"] == 2                          # retried once, per request


def test_api_backend_client_error_is_recorded_per_sample(openai_server):
    """Any failing request, 4xx included, fails only its own sample (the run is not aborted)."""
    openai_server.respond = lambda body: (400, {"error": "unknown field"})
    r = ak.evaluate([Sample(id="a", input="q", tools=TOOLS, expected_tool_calls=[[P]])], model="api:mock-model",
                    api_base=openai_server.url, api_key="EMPTY", adapter=ak.ToolCallAdapter(),
                    scorers=[ak.ToolCallF1()], config=ak.RunConfig(max_retries=0))
    assert r.errors and "tool_call_f1" not in r.headline


def test_agent_backend_one_bad_sample_does_not_sink_the_run(agent_server):
    """The agent: side of FINDING-1 -- the behaviour api: should match."""
    agent_server.respond = lambda body: (500, {"error": "boom"}) if body["input"] == "bad" \
        else (200, {"output": "ok", "tool_calls": [[P]]})
    samples = [Sample(id=i, input=i, tools=TOOLS, expected_tool_calls=[[P]]) for i in ("ok1", "bad", "ok2")]
    r = ak.evaluate(samples, model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(),
                    scorers=[ak.ToolCallF1()], config=ak.RunConfig(max_retries=0))
    got = per_sample(r)
    assert r.errors and "tool_call_f1" not in got.get("bad", {})
    assert got["ok1"]["tool_call_f1"] == 1.0 and got["ok2"]["tool_call_f1"] == 1.0


# -- agent: backend against a mock deployed agent ---------------------------------------

def transcript(turns, answer="done"):
    msgs, i = [{"role": "user", "content": "q"}], 0
    for turn in turns:
        entries = []
        for call in turn:
            i += 1
            entries.append(oa(i, call))
        msgs.append({"role": "assistant", "content": None, "tool_calls": entries})
        msgs += [{"role": "tool", "tool_call_id": e["id"], "content": "ok"} for e in entries]
    msgs.append({"role": "assistant", "content": answer})
    return msgs


AGENT_SCRIPT = {
    "good agent": ([[P, R], [T("Rome")]], {"output": "Rome, 12:00", "messages": transcript([[P, R], [T("Rome")]])}),
    "serial agent": ([[P, R], [T("Rome")]], {"messages": transcript([[P], [R], [T("Rome")]], "Rome")}),
    "looping agent": ([[SEARCH], [BOOK]], {"messages": transcript([[SEARCH], [SEARCH], [SEARCH], [BOOK]])}),
    "turns-only agent": ([[P, R]], {"output": "ok", "tool_calls": [[P, R]]}),
    "answer-only agent": ([[P]], {"output": "It is sunny in Paris."}),
}


def test_agent_backend_multi_turn_end_to_end(agent_server):
    agent_server.respond = lambda body: (200, AGENT_SCRIPT[body["input"]][1])
    samples = [Sample(id=k, input=k, tools=TOOLS, expected_tool_calls=ref) for k, (ref, _) in AGENT_SCRIPT.items()]
    r = ak.evaluate(samples, model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=AGENT_METRICS())
    assert r.errors == []
    got = per_sample(r)
    assert got["good agent"]["trajectory_strict"] == 1.0 and got["good agent"]["parallel_recall"] == 1.0
    assert got["serial agent"]["trajectory_strict"] == 0.0 and got["serial agent"]["trajectory_in_order"] == 1.0
    assert got["serial agent"]["parallel_recall"] == 0.0
    assert got["looping agent"]["redundant_tool_calls"] == pytest.approx(0.5)
    assert got["looping agent"]["trajectory_in_order"] == 1.0
    assert got["turns-only agent"]["parallel_recall"] == 1.0
    # An answer with no transcript/tool_calls: tool use unobservable -> tool metrics skip (not "no calls")
    assert "tool_call_recall" not in got["answer-only agent"]
    # the agent received the tools and the prompt
    assert agent_server.received[0]["body"]["tools"] == TOOLS


def test_agent_backend_http_error_recorded(agent_server):
    agent_server.respond = lambda body: (502, {"error": "upstream"})
    r = ak.evaluate([Sample(id="x", input="q", tools=TOOLS, expected_tool_calls=[[P]])],
                    model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()],
                    config=ak.RunConfig(max_retries=0))
    assert r.failed_count == 1 or r.errors
    assert "tool_call_f1" not in per_sample(r).get("x", {})


def test_agent_backend_malformed_json_recorded(agent_server):
    agent_server.respond = lambda body: (200, b"not json{")
    r = ak.evaluate([Sample(id="x", input="q", tools=TOOLS, expected_tool_calls=[[P]])],
                    model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()],
                    config=ak.RunConfig(max_retries=0))
    assert r.failed_count == 1 or r.errors


# -- consistency findings across metrics ------------------------------------------------

def test_task_completion_parse_failure_is_not_averaged_as_zero():
    """Regression for FINDING-3."""
    from auditkit.model import CallableModel
    replies = iter(["CHOICE: complete", "no verdict here"])
    judge = CallableModel(lambda ps: [next(replies) for _ in ps])
    ss = [Sample(id=f"s{i}", input="q", actual_output="done", actual_trace={"tool_calls": [[P]]}) for i in range(2)]
    r = ak.evaluate(ss, model="precomputed", scorers=[ak.TaskCompletion(judge_model=judge)])
    assert r.headline["task_completion"] == 1.0 and len(r.errors) == 1


def test_rag_judge_parse_failure_is_an_error_not_a_zero():
    """The behaviour FINDING-3 compares against."""
    from auditkit.model import CallableModel
    s = Sample(id="r", input="q", actual_output="Paris is in France.",
               actual_trace={"retrieved_contexts": ["Paris is in France."]})
    r = ak.evaluate([s], model="precomputed",
                    scorers=[ak.Faithfulness(judge_model=CallableModel(lambda ps: ["garbage"] * len(ps)))])
    assert "faithfulness" not in r.headline and len(r.errors) == 1


def test_validity_checks_nested_types():
    """Regression for LIMIT-1: array items and nested object properties are validated."""
    s = Sample(id="v", input="q", tools=TOOLS, actual_output="",
               actual_trace={"tool_calls": [[c("create_event", title="t", date="d", attendees=["a", 3],
                                               reminder={"minutes_before": "soon"})]]})
    [p] = ak.evaluate([s], model="precomputed", scorers=[ak.ToolCallValidity()]).predictions
    sc = p.metadata["scores"][0]
    assert sc["value"] == 0.0
    assert "attendees[1]" in sc["reason"] and "reminder.minutes_before" in sc["reason"]


@pytest.mark.parametrize("args,valid", [
    ({"title": "t", "date": "d", "attendees": ["a", "b"], "reminder": {"minutes_before": 5}}, True),
    ({"title": "t", "date": "d", "attendees": []}, True),
    ({"title": "t", "date": "d", "attendees": [["nested"]]}, False),
    ({"title": "t", "date": "d", "reminder": {"minutes_before": 5.5}}, False),
    ({"title": "t", "date": "d", "reminder": {"extra": 1}}, False),       # an object that declares properties is closed
])
def test_validity_nested_cases(args, valid):
    s = Sample(id="v", input="q", tools=TOOLS, actual_output="", actual_trace={"tool_calls": [[c("create_event", **args)]]})
    h = ak.evaluate([s], model="precomputed", scorers=[ak.ToolCallValidity()]).headline
    assert h["tool_call_validity"] == (1.0 if valid else 0.0)


def test_validity_nested_required_and_additional_properties():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "o": {"type": "object", "required": ["id"], "additionalProperties": False,
              "properties": {"id": {"type": "integer"}}}}}}}]
    for args, want in (({"o": {"id": 1}}, 1.0), ({"o": {}}, 0.0), ({"o": {"id": 1, "x": 2}}, 0.0)):
        s = Sample(id="v", input="q", tools=tools, actual_output="", actual_trace={"tool_calls": [[c("f", **args)]]})
        assert ak.evaluate([s], model="precomputed", scorers=[ak.ToolCallValidity()]).headline["tool_call_validity"] == want


def test_missing_trace_is_scored_as_no_calls():
    """GAP-1 (current behaviour): a run that reports no trace is indistinguishable from
    one that made no calls -- recall 0, not 'unknown'. agent_eval marks this ineligible."""
    s = Sample(id="a", input="q", expected_tool_calls=[[P]], actual_output="It's sunny in Paris.")
    assert ak.evaluate([s], model="precomputed", scorers=[ak.ToolCallF1()]).headline["tool_call_recall"] == 0.0
