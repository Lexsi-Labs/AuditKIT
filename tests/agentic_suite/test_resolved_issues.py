"""Regression tests for review items fixed on feat/rag-agent-evals (see FINDINGS.md).

- answer-only agent replies: tool use is unobservable (`tool_calls_unavailable`), so tool metrics skip
- exact matching tolerates extra arguments the schema declares optional
- the api: judge timeout default (LLM judges here; the RAG judges' share is in PR #6)
- OPENAI_API_KEY is only sent to openai.com hosts
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metrics.agent import ParallelToolCalls, ToolCallF1, TrajectoryMatch
from auditkit.model import CallableModel
from auditkit.model.api_gen import APIModel
from auditkit.sample import Sample

from .catalog import TOOLS, T, W


def run(metric, expected, predicted, tools=TOOLS):
    s = Sample(input="t", expected_tool_calls=expected, tools=tools)
    return {x.name: x.value for x in metric.score(s, "", {"trace": {"tool_calls": predicted}})}


# -- LIVE-1: schema-aware exact matching --------------------------------------------------

@pytest.mark.parametrize("pred,want", [
    ([[W("Paris", unit="celsius")]], 1.0),          # declared optional, added unasked -> fine
    ([[W("Paris", unit="kelvin")]], 1.0),           # value of an unconstrained optional isn't judged here
    ([[W("Paris", country="FR")]], 0.0),            # undeclared argument -> still a mismatch
    ([[W("Lyon", unit="celsius")]], 0.0),           # reference argument wrong -> mismatch
    ([[W("Paris")]], 1.0),
])
def test_exact_tolerates_declared_optional_args(pred, want):
    assert run(ToolCallF1(), [[W("Paris")]], pred)["tool_call_f1"] == want


def test_optional_arg_in_the_reference_must_match_exactly():
    assert run(ToolCallF1(), [[W("Chicago", unit="fahrenheit")]], [[W("Chicago", unit="celsius")]])["tool_call_f1"] == 0.0
    assert run(ToolCallF1(), [[W("Chicago", unit="fahrenheit")]], [[W("Chicago")]])["tool_call_f1"] == 0.0


def test_without_schemas_exact_stays_strict():
    assert run(ToolCallF1(), [[W("Paris")]], [[W("Paris", unit="celsius")]], tools=None)["tool_call_f1"] == 0.0


def test_parallel_and_trajectory_use_the_same_tolerance():
    ref = [[W("Paris"), W("Rome")], [T("Rome")]]
    pred = [[W("Paris", unit="celsius"), W("Rome", unit="celsius")], [T("Rome")]]
    assert run(ParallelToolCalls(), ref, pred)["parallel_recall"] == 1.0
    assert run(TrajectoryMatch(), ref, pred) == {"trajectory_strict": 1.0, "trajectory_in_order": 1.0}


def test_explicit_modes_and_matchers_are_unchanged():
    # name/subset and a custom arg_match keep their own semantics
    assert run(ToolCallF1(arg_mode="subset"), [[W("Paris")]], [[W("Paris", country="FR")]])["tool_call_f1_subset"] == 1.0
    fn = lambda name, p, r: p == r  # noqa: E731  strict custom matcher
    got = run(ToolCallF1(arg_match=fn), [[W("Paris")]], [[W("Paris", unit="celsius")]])
    assert next(v for k, v in got.items() if k.startswith("tool_call_f1_custom")) == 0.0


def test_score_names_did_not_change():
    assert set(run(ToolCallF1(), [[W("Paris")]], [[W("Paris")]])) == {
        "tool_call_precision", "tool_call_recall", "tool_call_f1", "tool_call_exact"}


# -- answer-only agent replies: tool use unobservable -----------------------------------------

def _ctx(**trace):
    return {"trace": trace}


@pytest.mark.parametrize("metric", [ToolCallF1(), TrajectoryMatch(), ParallelToolCalls(), ak.ToolCallValidity(),
                                    ak.RedundantToolCalls()])
def test_tool_metrics_skip_runs_whose_tool_use_is_unobservable(metric):
    """Neither a transcript nor tool_calls: "didn't call" and "didn't say" can't be told apart -> no score."""
    s = Sample(input="t", expected_tool_calls=[], tools=TOOLS)
    assert metric.score(s, "an answer", _ctx(tool_calls_unavailable=True)) == []


def test_explicit_empty_tool_calls_is_evidence_of_no_calls():
    s = Sample(input="t", expected_tool_calls=[], tools=TOOLS)
    assert {x.name: x.value for x in ToolCallF1().score(s, "42", _ctx(tool_calls=[]))}["tool_call_f1"] == 1.0


def test_non_tool_metrics_still_score_answer_only_runs():
    judge = CallableModel(lambda ps: ["CHOICE: complete"] * len(ps))
    s = Sample(id="x", input="What is 12+30?", target="42", actual_output="42",
               actual_trace={"tool_calls_unavailable": True, "retrieved_contexts": ["12+30=42"]},
               reference_contexts=["12+30=42"])
    r = ak.evaluate([s], model="precomputed", scorers=[ak.TaskCompletion(judge_model=judge), ak.RetrievalMetrics(),
                                                       ak.LexicalGroundedness(), ak.AnswerOverlap()])
    assert r.errors == []
    assert r.headline["task_completion"] == 1.0 and r.headline["hit_rate"] == 1.0


def test_recorded_text_without_trace_is_still_parsed():
    """Only the agent: backend marks missing evidence; a recorded text output is real evidence."""
    s = Sample(input="t", expected_tool_calls=[[W("Paris")]])
    out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    assert {x.name: x.value for x in ToolCallF1().score(s, out, {})}["tool_call_f1"] == 1.0


def test_agent_answer_only_reply_is_not_scored(agent_server):
    agent_server.respond = lambda body: (200, {"output": "42"})
    r = ak.evaluate([Sample(id="x", input="12+30?", tools=TOOLS, expected_tool_calls=[])],
                    model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=[ToolCallF1()])
    assert r.errors == [] and "tool_call_f1" not in r.headline


def test_agent_explicit_empty_tool_calls_scores_irrelevance(agent_server):
    agent_server.respond = lambda body: (200, {"output": "42", "tool_calls": []})
    r = ak.evaluate([Sample(id="x", input="12+30?", tools=TOOLS, expected_tool_calls=[])],
                    model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=[ToolCallF1()])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0


def test_post_loop_chat_completion_reply_is_unobservable(agent_server):
    """A chat-completion reply with no tool_calls may be the final message of a loop that did call tools."""
    agent_server.respond = lambda body: (200, {"choices": [{"message": {"role": "assistant", "content": "42"},
                                                            "finish_reason": "stop"}]})
    r = ak.evaluate([Sample(id="x", input="12+30?", tools=TOOLS, expected_tool_calls=[])],
                    model=f"agent:{agent_server.url}", adapter=ak.ToolCallAdapter(), scorers=[ToolCallF1()])
    assert r.errors == [] and "tool_call_f1" not in r.headline


# -- JUDGE-TIMEOUT --------------------------------------------------------------------------------

# The RAG judges (faithfulness, context_precision, context_recall) get the same default in PR #6.

def test_api_llm_judges_default_to_a_long_timeout(monkeypatch):
    monkeypatch.delenv("AUDITKIT_JUDGE_TIMEOUT", raising=False)
    m = ak.TaskCompletion(judge_model="api:some-judge", judge_model_args={"api_base": "http://localhost:1/v1",
                                                                           "api_key": "x"})
    assert m._model().timeout == 600


def test_explicit_judge_timeout_wins():
    m = ak.TaskCompletion(judge_model="api:j", judge_model_args={"api_base": "http://localhost:1/v1", "api_key": "x",
                                                                  "timeout": 30})
    assert m._model().timeout == 30


def test_judge_timeout_is_not_part_of_the_fingerprint_identity():
    a = ak.Faithfulness(judge_model="api:j", judge_model_args={"api_base": "http://h/v1"}).identity()
    b = ak.Faithfulness(judge_model="api:j", judge_model_args={"api_base": "http://h/v1", "timeout": 5}).identity()
    assert a == b


# -- KEY-1: OPENAI_API_KEY never leaves for a non-OpenAI host -------------------------------------

@pytest.mark.parametrize("base,expected", [
    ("https://api.openai.com/v1", "sk-real"),
    ("https://eu.api.openai.com/v1", "sk-real"),
    ("http://localhost:11434/v1", None),
    ("http://my-vllm:8000/v1", None),
    ("https://openai.com.attacker.example/v1", None),     # look-alike host
    ("https://api.groq.com/openai/v1", None),              # "openai" in the path is not the host
])
def test_openai_key_only_goes_to_openai(monkeypatch, base, expected):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert APIModel(api_base=base)._api_key == expected


def test_explicit_and_generic_keys_still_work(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert APIModel(api_base="http://localhost:1/v1", api_key="EMPTY")._api_key == "EMPTY"
    monkeypatch.setenv("API_KEY", "generic")
    assert APIModel(api_base="http://localhost:1/v1")._api_key == "generic"
