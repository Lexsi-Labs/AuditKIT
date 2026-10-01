"""AgentTune -> AuditKIT (#1 + #10): every AgentTune output shape (run_eval /
GRPO dataset rows, TraceLogger records, TrajectoryStore exports, TrajectoryDataset
runs, run_eval reports, EventLogs), a whole run folder with lexsi_provenance.json,
inspect, rescore, the Sample path, a deployed AgentTune agent behind agent:, and
provenance flowing into results.

Uses the real AgentTune fixtures in tests/fixtures/lexsi_real/agenttune. Offline.
"""

from __future__ import annotations

import json
import shutil

import pytest

import auditkit as ak

from .conftest import AGENTTUNE, cached_tokenizer, stub_hf

if not hasattr(ak, "episodes_from_agenttune"):
    pytest.skip("AgentTune import not present", allow_module_level=True)

from auditkit.agent_eval import AgentCase  # noqa: E402
from auditkit.agent_eval.importers import inspect_agenttune  # noqa: E402
from auditkit.agent_eval.runner import rescore  # noqa: E402

PROV = {"schema": "lexsi.provenance/1", "library": "agenttune", "version": "1.1.0", "git_sha": None,
        "created_at": "2026-09-28T12:00:00Z", "base_model": "CohereLabs/tiny-aya", "method": "grpo",
        "inputs": [], "params": {}}
ADD = {"name": "add", "arguments": {"a": 2, "b": 3}}
OPENAI_ADD = {"id": "c1", "type": "function", "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 3})}}
COHERE_ADD = {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}


def fixture(name):
    return str(AGENTTUNE / name)


def trajectory(tid, action, final="5"):
    return {"trajectory_id": tid, "task": "what is 2 plus 3", "reward": 1.0, "final_response": final,
            "steps": [{"step_number": 0, "action": action, "observation": "5"},
                      {"step_number": 1, "action": {}, "observation": ""}], "metadata": {}}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


# -- the loaders: every shape AgentTune writes -------------------------------------------------------------------------

@pytest.mark.parametrize("name,n,kind,has_output", [
    ("dataset_train_grpo.jsonl", 3, ak.TaskKind.RAG, False),         # tasks to run
    ("trace_finder.jsonl", 2, ak.TaskKind.RAG, True),                # TraceLogger records
    ("trace_hotpotqa.jsonl", 2, ak.TaskKind.GENERATIVE, True),
    ("trajectory_store_export.jsonl", 2, ak.TaskKind.GENERATIVE, True),
])
def test_load_agenttune_reads_every_fixture(name, n, kind, has_output):
    samples = ak.load_agenttune(fixture(name))
    assert len(samples) == n and {s.kind for s in samples} == {kind}
    assert all((s.actual_output is not None) is has_output for s in samples)
    assert all(s.input and s.target for s in samples)


def test_dataset_rows_carry_the_prompt_messages_and_gold_evidence():
    s = ak.load_agenttune(fixture("dataset_train_grpo.jsonl"))[0]
    assert s.metadata["messages"][0]["role"] == "system" and s.reference_contexts == ["doc-1::0"]


def test_trace_records_score_retrieval_directly():
    s = ak.load_agenttune(fixture("trace_finder.jsonl"))
    r = ak.evaluate(s, "precomputed", [ak.RetrievalMetrics(k=2), "exact_match"])
    assert r.errors == [] and r.headline["recall@2"] == pytest.approx(0.5) and r.headline["exact_match"] == 1.0


@pytest.mark.parametrize("final,answer", [
    ("<answer> Paris </answer>", "Paris"), ("reasoning <ANSWER>Rome</ANSWER> tail", "Rome"),
    (42, "42"), (None, ""), ("no tag at all", "no tag at all"),
])
def test_the_answer_tag_is_stripped_like_agenttune_does(tmp_path, final, answer):
    [s] = ak.load_agenttune(jsonl(tmp_path / "t.jsonl", [trajectory("t", {"name": "add", "arguments": {}}, final)]))
    assert s.actual_output == answer


# -- episodes: every shape, with honest coverage ----------------------------------------------------------------------

@pytest.mark.parametrize("name,fmt,n", [
    ("dataset_train_grpo.jsonl", "agenttune_run_eval_row", 3),
    ("trace_finder.jsonl", "agenttune_trace", 2),
    ("trajectory_store_export.jsonl", "agenttune_trace", 2),
    ("report_constructed.json", "agenttune_report", 2),
])
def test_episodes_from_every_fixture(name, fmt, n):
    eps = ak.episodes_from_agenttune(fixture(name))
    assert len(eps) == n and {e.source_format for e in eps} == {fmt}


def test_unrun_tasks_are_marked_and_claim_nothing():
    e = ak.episodes_from_agenttune(fixture("dataset_train_grpo.jsonl"))[0]
    assert e.metadata["is_task"] and e.final_answer is None and set(e.coverage.values()) == {"unavailable"}


@pytest.mark.parametrize("action,expected", [
    ({"tool_calls": [OPENAI_ADD]}, [ADD]),                          # rollout-normalised OpenAI calls
    ({"tool_calls": [COHERE_ADD]}, [ADD]),                          # Cohere dicts kept as-is
    (COHERE_ADD, [ADD]),                                             # a flat Cohere action
    ({"name": "add", "arguments": {"a": 2, "b": 3}}, [ADD]),        # a flat call
    ({"tool_calls": [OPENAI_ADD, COHERE_ADD]}, [[ADD, ADD]]),        # one step, two calls = a parallel group
])
def test_trajectory_actions_become_scored_turns(tmp_path, action, expected):
    [e] = ak.episodes_from_agenttune(jsonl(tmp_path / "t.jsonl", [trajectory("t", action)]))
    s = ak.Sample(input="q", actual_output=e.final_answer, actual_trace=e.to_trace(), expected_tool_calls=expected)
    r = ak.evaluate([s], "precomputed", [ak.ToolCallF1()])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0


def test_source_reward_is_provenance_never_the_outcome(tmp_path):
    row = trajectory("t", {"name": "add", "arguments": {"a": 2, "b": 3}}, final="6")
    row["reward"] = 1.0
    [e] = ak.episodes_from_agenttune(jsonl(tmp_path / "t.jsonl", [row]))
    case = AgentCase(id="t", task="what is 2 plus 3", outcome={"type": "answer_assertion", "reference": "5"})
    [res] = rescore([e], ["tool_call_f1"], cases=[case]).rows
    assert e.source_reward == 1.0 and res.outcome["verdict"] == "failure"   # AuditKIT's oracle decides, not the reward


# -- a whole AgentTune run folder -------------------------------------------------------------------------------------------

def run_folder(tmp_path):
    d = tmp_path / "run"
    (d / "checkpoints").mkdir(parents=True)
    jsonl(d / "b_trajectories.jsonl", [trajectory("t1", {"tool_calls": [COHERE_ADD]})])
    jsonl(d / "a_eventlog.jsonl", [{"id": "e1", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [OPENAI_ADD]}}},
        {"kind": "tool_result", "payload": {"output": "5"}}, {"kind": "text", "payload": {"text": "5"}}]}])
    shutil.copy(fixture("dataset_train_grpo.jsonl"), d / "c_dataset.jsonl")
    jsonl(d / "checkpoints" / "ignored.jsonl", [trajectory("nested", {})])
    (d / "config.json").write_text(json.dumps({"model_type": "cohere2"}))
    (d / "lexsi_provenance.json").write_text(json.dumps(PROV))
    return str(d)


def test_a_run_folder_reads_top_level_jsonl_in_name_order_with_provenance(tmp_path):
    eps = ak.episodes_from_agenttune(run_folder(tmp_path))
    assert [e.source_format for e in eps][:2] == ["eventlog", "agenttune_trajectory"] or \
        [e.source_id for e in eps][:2] == ["e1", "t1"]
    assert "nested" not in {e.source_id for e in eps}                       # sub-folders are not read
    assert all(e.metadata["provenance"] == PROV for e in eps)


def test_rescore_of_a_run_folder_skips_the_unrun_tasks(tmp_path):
    eps = ak.episodes_from_agenttune(run_folder(tmp_path))
    res = rescore(eps, ["tool_call_f1"], cases=[AgentCase(id="e1", task="t", reference_turns=[ADD]),
                                                  AgentCase(id="t1", task="t", reference_turns=[ADD])])
    assert len(eps) == 5 and len(res.rows) == 2
    assert {r.case_id: r.diagnostics["tool_call_f1"]["value"] for r in res.rows} == {"e1": 1.0, "t1": 1.0}


def test_an_empty_folder_is_no_episodes(tmp_path):
    assert ak.episodes_from_agenttune(str(tmp_path)) == []


def test_inspect_is_read_only_and_counts_tasks(tmp_path):
    rep = inspect_agenttune(run_folder(tmp_path))
    assert rep["n_tasks"] == 3 and rep["formats"]["agenttune_run_eval_row"] == 3
    assert "tool_call_f1" in rep["eligible_metrics"]


# -- TraceLogger records: arguments but no tool names -----------------------------------------------------------------------

def test_nameless_trace_calls_are_not_scored_as_no_calls():
    e = ak.episodes_from_agenttune(fixture("trace_finder.jsonl"))[0]
    s = ak.Sample(input="q", actual_output=e.final_answer, actual_trace=e.to_trace(),
                  expected_tool_calls=[{"name": "search_corpus", "arguments": {"query": "<placeholder query a>"}}])
    r = ak.evaluate([s], "precomputed", [ak.ToolCallF1()])
    assert e.counters["n_tool_calls"] == 2 and r.headline.get("tool_call_f1") != 0.0


def test_inspect_does_not_offer_name_based_metrics_without_names():
    rep = inspect_agenttune(fixture("trace_finder.jsonl"))
    assert rep["coverage_sample"]["tool_call_names"] == "unavailable"
    assert not {"tool_call_f1", "trajectory_match", "parallel_tool_calls"} & set(rep["eligible_metrics"])


def test_rescore_says_why_a_metric_was_not_scored():
    eps = ak.episodes_from_agenttune(fixture("trace_finder.jsonl"))
    for i, e in enumerate(eps):
        e.case_id = f"t{i}"                          # TraceLogger records carry no case id: join them
    cases = [AgentCase(id=f"t{i}", task="t", reference_turns=[ADD]) for i in range(2)]
    rows = rescore(eps, ["tool_call_f1"], cases=cases).rows
    assert all(r.ineligible.get("tool_call_f1") == "trace.tool_call_names" and "tool_call_f1" not in r.diagnostics
               for r in rows)


# -- a deployed AgentTune agent behind agent: ------------------------------------------------------------------------------

def run_agent(server, reply, samples, scorers, **kw):
    server.respond = lambda body: (200, reply(body) if callable(reply) else reply)
    return ak.evaluate(samples, model=f"agent:{server.url}", scorers=scorers, **kw)


def test_agent_endpoint_with_cohere_shaped_calls(agent_server):
    s = ak.Sample(input="2+3?", expected_tool_calls=[ADD])
    r = run_agent(agent_server, {"output": "5", "tool_calls": [[COHERE_ADD]]}, [s], [ak.ToolCallF1()])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0
    assert agent_server.received[0]["input"] == "2+3?"


def test_agent_endpoint_with_an_openai_transcript_and_contexts(agent_server):
    msgs = [{"role": "user", "content": "2+3?"},
            {"role": "assistant", "content": None, "tool_calls": [OPENAI_ADD]},
            {"role": "tool", "tool_call_id": "c1", "content": "5"}, {"role": "assistant", "content": "5"}]
    s = ak.Sample(input="2+3?", target="5", expected_tool_calls=[ADD], reference_contexts=["doc-1"])
    r = run_agent(agent_server, {"output": "5", "messages": msgs, "contexts": ["doc-1", "doc-2"]}, [s],
                  [ak.ToolCallF1(), ak.RetrievalMetrics(k=2), "exact_match"])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0 and r.headline["recall@2"] == 1.0
    assert r.headline["exact_match"] == 1.0


ADD_TOOL = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}]


def test_an_answer_only_reply_with_offered_tools_leaves_tool_metrics_unscored(agent_server):
    s = ak.Sample(input="2+3?", tools=ADD_TOOL, expected_tool_calls=[ADD])
    r = run_agent(agent_server, {"output": "5"}, [s], [ak.ToolCallF1()], adapter=ak.ToolCallAdapter())
    assert "tool_call_f1" not in r.headline                          # unobservable, not a measured miss


def test_an_answer_only_reply_from_a_server_side_tool_agent_is_unscored(agent_server):
    s = ak.Sample(input="2+3?", expected_tool_calls=[ADD])
    r = run_agent(agent_server, {"output": "5"}, [s], [ak.ToolCallF1()])
    assert "tool_call_f1" not in r.headline


def test_agent_endpoint_errors_are_per_sample(agent_server):
    s = [ak.Sample(id="ok", input="fine", expected_tool_calls=[ADD]),
         ak.Sample(id="bad", input="boom", expected_tool_calls=[ADD])]
    agent_server.respond = lambda body: (400, {"error": "bad"}) if body["input"] == "boom" else \
        (200, {"output": "5", "tool_calls": [[COHERE_ADD]]})
    r = ak.evaluate(s, model=f"agent:{agent_server.url}", scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0 and {e["sample_id"] for e in r.errors} == {"bad"}


# -- provenance: AgentTune checkpoint -> AuditKIT results --------------------------------------------------------------------

def test_an_agenttune_checkpoint_folders_provenance_reaches_the_results(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "lexsi_provenance.json").write_text(json.dumps(PROV))
    m = stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"), reply="5", name=str(ckpt))
    r = ak.evaluate([ak.Sample(input="2+3?", target="5")], model=m, scorers=["exact_match"])
    [model_input] = [i for i in r.metadata["inputs"] if i["kind"] == "model"]
    assert model_input["provenance"] == PROV
    out = tmp_path / "results"
    out.mkdir()
    r.save(str(out / "run.json"))
    written = json.loads((out / "lexsi_provenance.json").read_text())
    assert written["library"] == "auditkit" and written["inputs"][0]["provenance"]["library"] == "agenttune"
