"""Tests for the agent/RAG loaders: load_jsonl and load_agenttune."""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.adapter import ToolCallAdapter
from auditkit.loaders import _flat_ids, load_agenttune, load_jsonl
from auditkit.runspec import RunConfig
from auditkit.types import TaskKind


def _write_jsonl(path, rows, blank_after=None):
    lines = [json.dumps(r) for r in rows]
    if blank_after is not None:
        lines.insert(blank_after, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


WEATHER_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "book_flight", "parameters": {
        "type": "object", "properties": {"destination": {"type": "string"}}, "required": ["destination"]}}},
]

# Reference: two independent weather lookups (one parallel turn), then a
# booking that depends on their results.
EXPECTED = [
    [{"name": "get_weather", "arguments": {"city": "Paris"}},
     {"name": "get_weather", "arguments": {"city": "London"}}],
    [{"name": "book_flight", "arguments": {"destination": "Paris"}}],
]

TASK = "Check the weather in Paris and London, then book a flight to whichever is warmer."

# Parallel agent, recorded with a full OpenAI transcript and retrieved chunk ids.
TRAJ_PARALLEL = {
    "task": TASK,
    "trajectory_id": "traj-001",
    "reward": 1.0,
    "final_response": "Paris is warmer (21C vs 14C); booked flight PA-221 to Paris.",
    "steps": [
        {"step_number": 0, "state": TASK, "thought": "Both lookups are independent.",
         "action": {"tool_calls": [_call("c1", "get_weather", city="Paris"),
                                   _call("c2", "get_weather", city="London")]},
         "observation": "Paris 21C; London 14C", "reward": 0.0},
        {"step_number": 1, "state": "", "thought": "Paris is warmer.",
         "action": {"tool_calls": [_call("c3", "book_flight", destination="Paris")]},
         "observation": "PA-221 confirmed", "reward": 0.0},
        {"step_number": 2, "state": "", "thought": "Done.", "action": {}, "observation": "", "reward": 1.0},
    ],
    "metadata": {
        "conversation": [
            {"role": "system", "content": "You are a travel agent."},
            {"role": "user", "content": TASK},
            {"role": "assistant", "content": None,
             "tool_calls": [_call("c1", "get_weather", city="Paris"), _call("c2", "get_weather", city="London")]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 21}"},
            {"role": "tool", "tool_call_id": "c2", "content": "{\"temp_c\": 14}"},
            {"role": "assistant", "content": None,
             "tool_calls": [_call("c3", "book_flight", destination="Paris")]},
            {"role": "tool", "tool_call_id": "c3", "content": "{\"confirmation\": \"PA-221\"}"},
            {"role": "assistant", "content": "Paris is warmer (21C vs 14C); booked flight PA-221 to Paris."},
        ],
        "retrieved_chunk_ids": ["doc-1", "doc-2", 3],
        "episode": 7,
    },
}

# Serial agent, recorded without a conversation: turns come from steps.
TRAJ_SERIAL = {
    "task": TASK,
    "trajectory_id": 2,
    "reward": 0.5,
    "final_response": "Booked a flight to Paris.",
    "steps": [
        {"step_number": 0, "action": {"tool_calls": [_call("a", "get_weather", city="Paris")]},
         "observation": "21C", "thought": "", "state": "", "reward": 0.0},
        {"step_number": 1, "action": {"tool_calls": [_call("b", "get_weather", city="London")]},
         "observation": "14C", "thought": "", "state": "", "reward": 0.0},
        {"step_number": 2, "action": {"tool_calls": [_call("c", "book_flight", destination="Paris")]},
         "observation": "ok", "thought": "", "state": "", "reward": 0.0},
        {"step_number": 3, "action": {}, "observation": "", "thought": "", "state": "", "reward": 0.5},
    ],
    "metadata": {},
}


# -- load_jsonl ---------------------------------------------------------------

def test_load_jsonl_fields_metadata_and_kind(tmp_path):
    path = _write_jsonl(tmp_path / "data.jsonl", [
        {"input": "Weather in Paris?", "tools": WEATHER_TOOLS, "expected_tool_calls": [EXPECTED[0][0]],
         "id": 17, "difficulty": "easy"},
        {"input": "Tell me a joke", "expected_tool_calls": [], "tags": ["irrelevance"]},
        {"input": "Who wrote it?", "target": "Ada", "retrieval_context": ["Ada wrote it."],
         "reference_contexts": ["c9"], "metadata": {"source": "wiki"}, "split": "dev"},
        {"input": "2+2?", "target": "4", "actual_output": "4"},
        {"input": "Weather in Oslo?", "actual_output": "4C",
         "actual_trace": {"tool_calls": [[EXPECTED[0][0]]], "retrieved_contexts": ["c1"]}},
    ], blank_after=2)
    s = load_jsonl(path)
    assert [x.kind for x in s][:4] == [TaskKind.AGENT, TaskKind.AGENT, TaskKind.RAG, TaskKind.GENERATIVE]
    assert s[0].id == "17" and s[0].tools == WEATHER_TOOLS and s[0].metadata == {"difficulty": "easy"}
    assert s[1].expected_tool_calls == [] and s[1].tags == ["irrelevance"]
    assert s[2].metadata == {"split": "dev", "source": "wiki"}
    assert s[2].retrieval_context == ["Ada wrote it."] and s[2].reference_contexts == ["c9"]
    assert s[3].actual_output == "4" and s[3].target == "4"
    assert s[4].actual_trace == {"tool_calls": [[EXPECTED[0][0]]], "retrieved_contexts": ["c1"]}
    assert s[4].kind == TaskKind.GENERATIVE and s[4].metadata == {}


def test_load_jsonl_field_map(tmp_path):
    path = _write_jsonl(tmp_path / "qa.jsonl", [
        {"question": "Capital of France?", "answer": "Paris", "ctx": ["Paris is the capital."], "qid": "q1"},
    ])
    (s,) = load_jsonl(path, field_map={"question": "input", "answer": "target",
                                       "ctx": "retrieval_context", "qid": "id"})
    assert (s.input, s.target, s.id) == ("Capital of France?", "Paris", "q1")
    assert s.retrieval_context == ["Paris is the capital."] and s.kind == TaskKind.RAG
    assert s.metadata == {}


def test_load_jsonl_errors_name_the_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "ok"}\n\n{"input": "broken"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3: invalid JSON"):
        load_jsonl(str(bad))
    missing = _write_jsonl(tmp_path / "missing.jsonl", [{"input": "a"}, {"question": "b"}])
    with pytest.raises(ValueError, match="line 2: missing 'input'"):
        load_jsonl(missing)


# -- load_agenttune: run_eval / RAG rows ---------------------------------------

def test_agenttune_run_eval_rows(tmp_path):
    prompt = [{"role": "system", "content": "Answer from the user's mailbox. Use the tools."},
              {"role": "user", "content": "What is the confirmation number for my Lisbon hotel?"}]
    path = _write_jsonl(tmp_path / "eval.jsonl", [
        {"prompt": prompt, "answer": ["251832", "#251832"], "message_ids": [["m-101"], ["m-104", "m-105"]],
         "sample_dir": "mail/0001"},
        {"prompt": "Who sent the invoice?", "answer": "Dana"},
        {"prompt": [{"role": "user", "content": "When is the audit due?"}], "gold_answer": "March 3",
         "question_id": 42, "gold_chunk_ids": ["ch-7", "ch-9"]},
    ])
    a, b, c = load_agenttune(path)

    assert a.input == "What is the confirmation number for my Lisbon hotel?"
    assert a.target == "251832" and a.id is None
    assert a.reference_contexts == ["m-101", "m-104", "m-105"] and a.kind == TaskKind.RAG
    assert a.metadata == {"sample_dir": "mail/0001", "messages": prompt}
    assert a.actual_output is None  # a task to run, not a recorded result

    assert (b.input, b.target, b.reference_contexts, b.metadata) == ("Who sent the invoice?", "Dana", None, {})
    assert b.kind == TaskKind.GENERATIVE

    assert (c.input, c.target, c.id) == ("When is the audit due?", "March 3", "42")
    assert c.reference_contexts == ["ch-7", "ch-9"]

    # ToolCallAdapter sends the original system prompt, not just the question.
    (req,) = ToolCallAdapter().adapt(a, RunConfig())
    assert req.params["messages"] == prompt


def test_agenttune_rejects_unknown_records(tmp_path):
    path = _write_jsonl(tmp_path / "x.jsonl", [{"prompt": "ok", "answer": "1"}, {"question": "?"}])
    with pytest.raises(ValueError, match="line 2: not an AgentTune record"):
        load_agenttune(path)
    no_user = _write_jsonl(tmp_path / "y.jsonl", [{"prompt": [{"role": "system", "content": "hi"}]}])
    with pytest.raises(ValueError, match="line 1: 'prompt' has no user message"):
        load_agenttune(no_user)


# -- load_agenttune: TrajectoryDataset ------------------------------------------

def test_agenttune_trajectories(tmp_path):
    path = _write_jsonl(tmp_path / "traj.jsonl", [TRAJ_PARALLEL, TRAJ_SERIAL])
    par, ser = load_agenttune(path)

    assert par.input == TASK and par.id == "traj-001" and par.kind == TaskKind.AGENT
    assert par.actual_output == TRAJ_PARALLEL["final_response"]
    # Turns come from steps (the rollout truncates/rewrites the conversation);
    # the conversation is kept only as messages.
    assert par.actual_trace == {
        "tool_calls": [s["action"]["tool_calls"] for s in TRAJ_PARALLEL["steps"]
                       if s["action"].get("tool_calls")],
        "messages": TRAJ_PARALLEL["metadata"]["conversation"],
        "retrieved_contexts": ["doc-1", "doc-2", "3"]}
    assert par.metadata == {"episode": 7, "reward": 1.0}

    assert ser.id == "2" and ser.metadata == {"reward": 0.5}
    # One turn per step that made calls; the terminal step is dropped.
    assert ser.actual_trace == {"tool_calls": [s["action"]["tool_calls"] for s in TRAJ_SERIAL["steps"][:3]]}
    assert [len(t) for t in ak.to_turns(ser.actual_trace)] == [1, 1, 1]


def test_agenttune_trajectories_score_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = load_agenttune(_write_jsonl(tmp_path / "traj.jsonl", [TRAJ_PARALLEL, TRAJ_SERIAL]))
    for s in samples:
        s.expected_tool_calls = EXPECTED
    samples[0].reference_contexts = ["doc-1", "3"]  # only the parallel run logged retrieval

    result = ak.evaluate(samples, model="precomputed",
                         scorers=[ak.ToolCallF1(), ak.ParallelToolCalls(), ak.RetrievalMetrics()])
    h, stats = result.headline, result.stats
    assert not result.errors
    # Both agents made exactly the right calls...
    assert h["tool_call_f1"] == 1.0 and h["tool_call_exact"] == 1.0
    # ...but only the first batched the independent weather lookups.
    assert h["parallel_recall"] == 0.5 and stats["parallel_recall"].count == 2
    assert h["parallel_detection"] == 0.5
    assert h["parallel_precision"] == 1.0 and stats["parallel_precision"].count == 1
    # Retrieval: ranked [doc-1, doc-2, 3] against gold {doc-1, 3}.
    assert stats["recall"].count == 1
    assert h["hit_rate"] == 1.0 and h["recall"] == 1.0 and h["mrr"] == 1.0
    assert h["precision"] == pytest.approx(2 / 3)
    assert h["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


# -- load_agenttune: run_eval report JSON ---------------------------------------

def test_agenttune_report_json(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    report = {
        "use_case": "mail_qa", "model": "qwen2.5-7b-agenttune", "n": 2,
        "samples": [
            {"idx": 0, "question": "Confirmation number for the Lisbon hotel?", "gold": "251832",
             "predicted": "251832", "n_tools": 2, "tool_calls": ["search_mail", "read_mail"],
             "scores": {"exact": 1.0}, "error": None},
            {"idx": 1, "question": "Who sent the invoice?", "gold": ["Dana", "Dana K."],
             "predicted": None, "n_tools": 0, "tool_calls": [], "scores": {}, "error": "timeout"},
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    a, b = load_agenttune(str(path))
    assert (a.input, a.target, a.actual_output, a.id) == ("Confirmation number for the Lisbon hotel?",
                                                           "251832", "251832", "0")
    assert a.metadata["tool_calls"] == ["search_mail", "read_mail"] and a.actual_trace is None
    assert (b.target, b.actual_output, b.metadata["error"]) == ("Dana", "", "timeout")

    result = ak.evaluate([a, b], model="precomputed", scorers=["exact_match"])
    assert result.headline["exact_match"] == 0.5


# -- load_agenttune: AgentTune compatibility (findings 22-27) -------------------

def test_agenttune_turns_from_steps_when_conversation_rewritten(tmp_path):
    """[22] The rollout truncates/rewrites the conversation (MEM1 folds it to
    [system, user, assistant(<state>), tool] with no tool_calls). Turns must
    still come from steps; the stub conversation is kept only as messages."""
    rewritten = [
        {"role": "system", "content": "You are a travel agent."},
        {"role": "user", "content": TASK},
        {"role": "assistant", "content": "<state>working</state>"},  # no tool_calls
        {"role": "tool", "tool_call_id": "c3", "content": "{}"},
    ]
    row = {**TRAJ_PARALLEL, "metadata": {"conversation": rewritten}}
    (s,) = load_agenttune(_write_jsonl(tmp_path / "t.jsonl", [row]))
    assert s.actual_trace["messages"] == rewritten
    assert s.actual_trace["tool_calls"] == [st["action"]["tool_calls"]
                                            for st in TRAJ_PARALLEL["steps"] if st["action"].get("tool_calls")]
    # predicted_turns prefers tool_calls, so the two real turns survive.
    assert [len(t) for t in ak.to_turns(s.actual_trace)] == [2, 1]


def test_agenttune_row_gold_path(tmp_path):
    """[23] Real RAG-GRPO rows keep gold chunk ids in gold_path (a JSON string)."""
    path = _write_jsonl(tmp_path / "grpo.jsonl", [{
        "prompt": [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "What was FY24 revenue?"}],
        "gold_answer": "$111.5M", "question_id": "q9",
        "gold_path": json.dumps(["ch-7", "ch-9"]), "hop_count": 2,
    }])
    (s,) = load_agenttune(path)
    assert s.reference_contexts == ["ch-7", "ch-9"] and s.kind == TaskKind.RAG
    assert "gold_path" not in s.metadata and s.metadata["hop_count"] == 2


def test_agenttune_trajectory_flat_actions(tmp_path):
    """[24] The documented flat action shape {"name","arguments"} must load as
    one call per step (search x2 with identical args, then fetch)."""
    traj = {
        "task": "find docs", "trajectory_id": "flat-1", "final_response": "done",
        "steps": [
            {"step_number": 0, "state": "", "thought": "", "observation": "r1",
             "action": {"name": "search_corpus", "arguments": {"q": "x"}}},
            {"step_number": 1, "state": "", "thought": "", "observation": "r2",
             "action": {"name": "search_corpus", "arguments": {"q": "x"}}},
            {"step_number": 2, "state": "", "thought": "", "observation": "r3",
             "action": {"name": "fetch", "arguments": {"id": "doc1"}}},
            {"step_number": 3, "state": "", "thought": "", "observation": "", "action": {}},
        ],
    }
    (s,) = load_agenttune(_write_jsonl(tmp_path / "flat.jsonl", [traj]))
    turns = ak.to_turns(s.actual_trace)
    assert [[c.name for c in t] for t in turns] == [["search_corpus"], ["search_corpus"], ["fetch"]]
    assert turns[0][0].arguments == {"q": "x"}


def test_agenttune_trajectory_input_is_last_user_message(tmp_path):
    """[25] A chat-list task must load as the last user message, not str(list)."""
    task = [{"role": "system", "content": "You are a financial research agent."},
            {"role": "user", "content": "What was the revenue?"},
            {"role": "assistant", "content": "one moment"},
            {"role": "tool", "tool_call_id": "c1", "content": "..."},
            {"role": "user", "content": "Just the number please."}]
    traj = {"task": task, "trajectory_id": "chat-1", "final_response": "ok", "steps": []}
    (s,) = load_agenttune(_write_jsonl(tmp_path / "chat.jsonl", [traj]))
    assert s.input == "Just the number please."
    assert "messages" not in s.metadata  # not duplicated into metadata


def test_agenttune_trace_jsonl(tmp_path):
    """[26] train_grpo's trace.jsonl shape must load (answer + retrieval ids)."""
    rec = {
        "question": "What was FY24 revenue?", "tool_calls": [{"query": {"q": "rev"}, "result": "..."}],
        "n_tool_calls": 1, "final_answer": "<answer>$111.5 million</answer>",
        "has_answer_tag": True, "reward": 1.0, "gold_answer": "$111.5 million",
        "reward_components": {"answer": 1.0}, "retrieved_chunk_ids": ["d::0", "d::1"],
        "gold_chunk_ids": ["d::0"],
    }
    (s,) = load_agenttune(_write_jsonl(tmp_path / "trace.jsonl", [rec]))
    assert s.input == "What was FY24 revenue?" and s.kind == TaskKind.RAG
    assert s.actual_output == "$111.5 million"  # <answer> stripped
    assert s.reference_contexts == ["d::0"]
    assert s.actual_trace == {"retrieved_contexts": ["d::0", "d::1"]}
    assert s.metadata["n_tool_calls"] == 1 and s.metadata["tool_calls"][0]["query"] == {"q": "rev"}


def test_agenttune_answer_tag_stripped_for_exact_match(tmp_path, monkeypatch):
    """[27] <answer>-tagged output must score exact_match 1.0, as AgentTune does."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    traj = {"task": "the question", "trajectory_id": "ans-1",
            "final_response": "<answer>251832</answer>", "steps": []}
    (s,) = load_agenttune(_write_jsonl(tmp_path / "a.jsonl", [traj]))
    s.target = "251832"
    assert s.actual_output == "251832"
    assert s.metadata["raw_output"] == "<answer>251832</answer>"  # raw kept
    result = ak.evaluate([s], model="precomputed", scorers=["exact_match"])
    assert result.headline["exact_match"] == 1.0


def test_agenttune_report_json_strips_answer_tag(tmp_path, monkeypatch):
    """[27] The report-JSON 'predicted' is also <answer>-unwrapped."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    report = {"samples": [
        {"idx": 0, "question": "Q?", "gold": "251832", "predicted": "<answer>251832</answer>"},
    ]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    (a,) = load_agenttune(str(path))
    assert a.actual_output == "251832"
    result = ak.evaluate([a], model="precomputed", scorers=["exact_match"])
    assert result.headline["exact_match"] == 1.0


# -- edge-case regressions (pr9-loaders) ---------------------------------------

def test_flat_ids_preserves_scalars_and_json_null():
    """[fix1] json.loads must not reformat a bare numeric/keyword id string, and
    JSON null is "no gold", not a phantom id."""
    # json.dumps'd lists / single ids and raw ids are preserved as-is.
    assert _flat_ids(json.dumps(["ch-7", "ch-9"])) == ["ch-7", "ch-9"]
    assert _flat_ids(json.dumps("chunk_1")) == ["chunk_1"]
    assert _flat_ids("chunk_1") == ["chunk_1"] and _flat_ids("d::0") == ["d::0"]
    assert _flat_ids(["m-1", "m-2"]) == ["m-1", "m-2"]
    # out-of-contract bare scalars keep their raw string (no 1.50->1.5 mangling).
    assert _flat_ids("1.50") == ["1.50"] and _flat_ids("1e3") == ["1e3"]
    assert _flat_ids("true") == ["true"] and _flat_ids("NaN") == ["NaN"]
    # "null" -> no gold, not the string "None".
    assert _flat_ids("null") is None and _flat_ids(None) is None


def test_load_jsonl_non_dict_metadata_names_the_line(tmp_path):
    """[fix2] A truthy non-dict metadata raises a clear ValueError naming the line."""
    for bad in ("a string", ["a", "list"], 5):
        path = _write_jsonl(tmp_path / "m.jsonl", [{"input": "ok"}, {"input": "q", "metadata": bad}])
        with pytest.raises(ValueError, match="line 2: 'metadata' must be a JSON object"):
            load_jsonl(path)
    # field_map pointing a scalar column at metadata is the same failure.
    fm = _write_jsonl(tmp_path / "fm.jsonl", [{"input": "q", "score": 0.9}])
    with pytest.raises(ValueError, match="line 1: 'metadata' must be a JSON object"):
        load_jsonl(fm, field_map={"score": "metadata"})
    # falsy metadata still coerces to {} (unchanged behavior).
    ok = _write_jsonl(tmp_path / "ok.jsonl", [{"input": "q", "metadata": {}}, {"input": "q2", "extra": 1}])
    a, b = load_jsonl(ok)
    assert a.metadata == {} and b.metadata == {"extra": 1}


def test_agenttune_scalar_id_fields_dont_crash(tmp_path):
    """[fix3] A scalar gold_path/retrieved_chunk_ids/message_ids coerces to one
    id instead of raising a raw TypeError; a lone string id is not char-split."""
    # scalar gold_path on a run_eval row.
    (s,) = load_agenttune(_write_jsonl(tmp_path / "a.jsonl",
                                       [{"prompt": "q", "gold_answer": "x", "gold_path": 7}]))
    assert s.reference_contexts == ["7"]
    # scalar + string retrieved_chunk_ids on a trace record.
    t1, t2 = load_agenttune(_write_jsonl(tmp_path / "b.jsonl", [
        {"question": "q", "final_answer": "a", "retrieved_chunk_ids": 5},
        {"question": "q", "final_answer": "a", "retrieved_chunk_ids": "doc-1"},
    ]))
    assert t1.actual_trace == {"retrieved_contexts": ["5"]}
    assert t2.actual_trace == {"retrieved_contexts": ["doc-1"]}  # not list("doc-1")


def test_agenttune_structural_mismatch_names_the_line(tmp_path):
    """[fix3] steps-as-dict / metadata-as-list / non-dict report sample raise a
    ValueError naming the line, not a raw TypeError/AttributeError."""
    steps_dict = _write_jsonl(tmp_path / "s.jsonl",
                              [{"task": "t", "final_response": "r", "steps": {"oops": 1}}])
    with pytest.raises(ValueError, match="line 1: malformed AgentTune record"):
        load_agenttune(steps_dict)
    meta_list = _write_jsonl(tmp_path / "m.jsonl",
                             [{"trajectory_id": "x", "final_response": "r", "metadata": ["oops"]}])
    with pytest.raises(ValueError, match="line 1: malformed AgentTune record"):
        load_agenttune(meta_list)
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"samples": [{"question": "q", "predicted": "a"}, "not-an-object"]}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match=r"samples\[1\] is not a JSON object"):
        load_agenttune(str(report))


def test_agenttune_numeric_answers_dont_crash(tmp_path):
    """[fix4] A numeric final_answer/final_response/predicted is coerced to str
    instead of crashing the <answer>-tag regex."""
    (trace,) = load_agenttune(_write_jsonl(tmp_path / "t.jsonl",
                                           [{"question": "q", "final_answer": 42}]))
    assert trace.actual_output == "42"
    (traj,) = load_agenttune(_write_jsonl(tmp_path / "j.jsonl",
                                          [{"trajectory_id": "z", "final_response": 3.5, "steps": []}]))
    assert traj.actual_output == "3.5"
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"samples": [{"question": "q", "gold": "g", "predicted": 7}]}),
                      encoding="utf-8")
    (rep,) = load_agenttune(str(report))
    assert rep.actual_output == "7"
