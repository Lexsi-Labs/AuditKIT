"""PR #1 (feat/rag-agent-evals), parts added after the first review and not covered by
tests/agentic_suite: reference-free agent metrics, reference-free RAG judges, RAG-stress
metrics, agent_eval reliability (A3), the harness loop (A4) and the sidecar (A5).

Judges are scripted (every verdict known), models are stubs; no network.
"""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.model import CallableModel, Generated, Model, Result_
from auditkit.sample import Sample

from .conftest import per_sample

if not hasattr(ak, "AgentLoopDetection"):
    pytest.skip("reference-free agent metrics not present", allow_module_level=True)


def W(city):
    return {"name": "get_weather", "arguments": {"city": city}}


TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def scores(metric, turns, **sample_kw):
    s = Sample(input="task", tools=sample_kw.pop("tools", TOOLS), **sample_kw)
    out = metric.score(s, "", {"trace": {"tool_calls": turns}})
    return {x.name: x.value for x in (out if isinstance(out, list) else [out])}


# -- agent_loop_detection ------------------------------------------------------------------------

@pytest.mark.parametrize("turns,want", [
    ([[W("Paris")], [W("Rome")]], 1.0),                       # no loop
    ([[W("Paris")], [W("Paris")], [W("Paris")]], 0.0),        # same call x3 (threshold 3)
    ([[W("Paris")], [W("Rome")], [W("Paris")]], 0.0),         # revisits an earlier call: call-graph cycle
])
def test_agent_loop_detection(turns, want):
    assert scores(ak.AgentLoopDetection(), turns) == {"agent_loop_detection": want}


def test_loop_detection_skips_runs_with_nothing_to_check_and_validates_threshold():
    assert scores(ak.AgentLoopDetection(), []) == {}
    with pytest.raises(ValueError):
        ak.AgentLoopDetection(repetition_threshold=1)


# -- tool_permission --------------------------------------------------------------------------------

@pytest.mark.parametrize("turns,metric,meta,want", [
    ([[W("Paris")]], ak.ToolPermission(), {}, 1.0),                                   # offered tool
    ([[{"name": "delete_all", "arguments": {}}]], ak.ToolPermission(), {}, 0.0),       # not offered
    ([[W("Paris")]], ak.ToolPermission(denied_tools=["get_weather"]), {}, 0.0),        # denial wins
    ([[W("Paris")]], ak.ToolPermission(), {"allowed_tools": ["get_time"]}, 0.0),       # sample allowlist
    ([[W("Paris"), {"name": "delete_all", "arguments": {}}]], ak.ToolPermission(), {}, 0.5),
])
def test_tool_permission(turns, metric, meta, want):
    assert scores(metric, turns, metadata=meta) == {"tool_permission": want}


def test_tool_permission_skips_when_no_calls():
    assert scores(ak.ToolPermission(), []) == {}


# -- tool_selection (per-call judge) ------------------------------------------------------------------

def _selection_judge(prompts):
    return ["Reason.\nCHOICE: no" if "delete_all" in p.split("<candidate_action>")[1] else "Reason.\nCHOICE: yes"
            for p in prompts]


def test_tool_selection_one_verdict_per_call():
    s = Sample(id="a", input="weather in Paris?", actual_output="",
               actual_trace={"tool_calls": [[W("Paris"), {"name": "delete_all", "arguments": {}}]]})
    r = ak.evaluate([s], model="precomputed", scorers=[ak.ToolSelectionJudge(judge_model=CallableModel(_selection_judge))])
    assert r.errors == [] and r.headline["tool_selection"] == pytest.approx(0.5)


def test_tool_selection_unreadable_verdict_is_an_error_not_a_zero():
    s = Sample(id="a", input="q", actual_output="", actual_trace={"tool_calls": [[W("Paris")]]})
    r = ak.evaluate([s], model="precomputed",
                    scorers=[ak.ToolSelectionJudge(judge_model=CallableModel(lambda ps: ["Unclear."] * len(ps)))])
    assert "tool_selection" not in r.headline and r.errors


# -- reference-free RAG judges -------------------------------------------------------------------------

class RagJudge:
    """Answers each RAG prompt type with a fixed reply."""

    def __call__(self, prompts):
        out = []
        for p in prompts:
            if "Break the ANSWER" in p:
                out.append("1. Laptops are replaced every 3 years.\n2. Laptops are free.")
            elif "Break the RESPONSE" in p:
                out.append("1. Laptops are replaced every 3 years.\n2. The cafeteria opens at 8.")
            elif "numbered CLAIM" in p:
                out.append("1: SUPPORTED\n2: UNSUPPORTED")
            elif "SENTENCE of the response" in p:
                out.append("1: SUPPORTED\n2: UNSUPPORTED\n3: SUPPORTED")
            elif "whether it addresses" in p:
                out.append("1: ADDRESSES\n2: OFF_TOPIC")
            elif "useful for answering" in p:
                out.append("1: RELEVANT\n2: IRRELEVANT\n3: IRRELEVANT\n4: RELEVANT")
            else:
                out.append("??")
        return out


def _rag_sample(output):
    return Sample(id="s", input="How often are laptops replaced?", actual_output=output,
                  actual_trace={"retrieved_contexts": ["Laptops: every 3 years.", "Cafeteria: 8am.", "Wifi: Mondays.",
                                                       "IT portal handles laptop swaps."]})


@pytest.mark.parametrize("metric_cls,output,name,want", [
    (ak.Hallucination, "Laptops are replaced every 3 years. They are free.", "hallucination", 0.5),   # 1 - 1/2
    (ak.AnswerRelevancy, "Laptops every 3 years. Cafeteria at 8.", "answer_relevancy", 0.5),
    (ak.ResponseGroundedness, "One. Two. Three.", "response_groundedness", 2 / 3),                  # 3 sentences split locally
    (ak.ContextRelevance, "anything", "context_relevance", 0.5),                                     # 2 of 4 chunks
])
def test_reference_free_rag_judges(metric_cls, output, name, want):
    r = ak.evaluate([_rag_sample(output)], model="precomputed",
                    scorers=[metric_cls(judge_model=CallableModel(RagJudge()))])
    assert r.errors == [] and r.headline[name] == pytest.approx(want)


def test_reference_free_rag_judges_skip_without_contexts():
    s = Sample(id="s", input="q", actual_output="an answer", actual_trace={"retrieved_contexts": []})
    r = ak.evaluate([s], model="precomputed", scorers=[ak.ResponseGroundedness(judge_model=CallableModel(RagJudge())),
                                                       ak.ContextRelevance(judge_model=CallableModel(RagJudge()))])
    assert r.headline == {} and r.errors == []


# -- RAG stress metrics -----------------------------------------------------------------------------------

rs = pytest.importorskip("auditkit.metrics.rag_stress")


def _stress_sample(sid, case, corpus, trace, output="x"):
    return Sample(id=sid, input=case.question, actual_output=output, actual_trace=trace,
                  metadata={"rag_case": case.to_dict(), "corpus": corpus.to_dict()})


def test_evidence_set_recall_needs_a_complete_sufficient_set():
    corpus, cases = rs.synthetic_bank_kb()
    c1 = cases[0]                                      # sets: [LP-CURRENT, FIN-TABLE] or [SUMMARY]
    full = _stress_sample("full", c1, corpus, {"retrieved": [{"id": "LP-CURRENT"}, {"id": "FIN-TABLE"}]})
    half = _stress_sample("half", c1, corpus, {"retrieved": [{"id": "LP-CURRENT"}]})
    got = per_sample(ak.evaluate([full, half], model="precomputed", scorers=[rs.EvidenceSetRecall()]))
    assert got["full"]["evidence_set_recall"] == 1.0 and got["half"]["evidence_set_recall"] == 0.0


def test_abstention_on_an_unanswerable_case():
    corpus, cases = rs.synthetic_bank_kb()
    c2 = cases[1]                                      # unanswerable
    refused = _stress_sample("refused", c2, corpus, {"abstained": True}, output="I don't have that information.")
    answered = _stress_sample("answered", c2, corpus, {"abstained": False}, output="It is 4.5%.")
    got = per_sample(ak.evaluate([refused, answered], model="precomputed", scorers=[rs.Abstention()]))
    assert got["refused"]["abstention"] == 1.0 and got["answered"]["abstention"] == 0.0


def test_not_tested_results_are_not_averaged_into_the_headline():
    corpus, cases = rs.synthetic_bank_kb()
    c1 = cases[0]
    measured = _stress_sample("m", c1, corpus, {"retrieved": [{"id": "SUMMARY"}]})
    untested = _stress_sample("u", c1, corpus, {})
    r = ak.evaluate([measured, untested], model="precomputed", scorers=[rs.EvidenceSetRecall()])
    assert r.headline["evidence_set_recall"] == 1.0


# -- agent_eval reliability (A3) ---------------------------------------------------------------------------

import importlib  # noqa: E402

# the package re-exports a function named `reliability`, which shadows the module attribute
rel = importlib.import_module("auditkit.agent_eval.reliability")


def test_reliability_by_hand():
    # 4 decided trials, 2 successes, k=2: pass@2 = 1 - C(2,2)/C(4,2) = 5/6 ; all-2 = C(2,2)/C(4,2) = 1/6
    r = rel.reliability(["success", "success", "failure", "failure"], k=2, independent=True)
    assert r["pass_at_k"] == pytest.approx(5 / 6) and r["all_k"] == pytest.approx(1 / 6)
    assert r["success_rate"] == pytest.approx(0.5) and r["status"] == "mixed"


def test_reliability_excludes_unknown_and_flags_unconfirmed_independence():
    r = rel.reliability(["success", "unknown", "success", "error"])
    assert r["n_decided"] == 2 and r["n_unknown"] == 1 and r["n_error"] == 1
    assert r["status"] == "reliable" and any("independence unconfirmed" in f for f in r["flags"])


def test_reliability_refuses_to_estimate_from_too_little():
    assert rel.reliability(["success"])["status"] == "unknown"
    assert rel.reliability(["success", "unknown"])["status"] == "unknown"


def test_duplicate_trial_ids_are_flagged():
    r = rel.reliability(["success", "success"], trial_ids=["t1", "t1"], independent=True)
    assert any("duplicate trial_id" in f for f in r["flags"])


@pytest.mark.parametrize("verdicts,want", [
    (["success", "success", "failure"], "success"),
    (["success", "failure"], "unknown"),                 # a tie is not a coin flip
    (["unknown", "error"], "unknown"),
])
def test_aggregate_verdict(verdicts, want):
    assert rel.aggregate_verdict(verdicts)[0] == want


# -- harness loop (A4) and sidecar (A5) ------------------------------------------------------------------------

from auditkit.agent_eval import (  # noqa: E402
    AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion, read_sidecar, sidecar_records, write_sidecar,
)


class ScriptedPolicy(Model):
    """Calls lookup_inventory once, then answers; or loops forever when `forever`."""

    name = "scripted-policy"

    def __init__(self, forever=False):
        self.forever = forever

    def generate(self, requests):
        out = []
        for r in requests:
            done = any(m.get("role") == "tool" for m in r.params.get("messages", []))
            if done and not self.forever:
                g = Generated(text="Shipment for A is ready.")
            else:
                g = Generated(text="", trace={"tool_calls": [[{"name": "lookup_inventory", "arguments": {"sku": "A"}}]]})
            out.append(Result_(completions=[g]))
        return out


class InventoryDouble:
    def __init__(self):
        self.state = {"shipment_decision": None}

    def __call__(self, name, args):
        self.state["shipment_decision"] = "ready"
        return {"in_stock": True}


def _harness(policy, max_steps=8):
    case = AgentCase(id="inv-1", task="Ship item A if it is in stock.", allowed_tools=["lookup_inventory"],
                     outcome=FinalStateAssertion("shipment_decision", equals="ready"))
    spec = AgentEvalSpec(cases=[case], mode="harness", agent=policy,
                         agent_opts={"tool_env": InventoryDouble(), "max_steps": max_steps}, scorers=[])
    return AgentEvalRunner().run(spec)


def test_harness_runs_the_loop_and_verifies_real_state():
    res = _harness(ScriptedPolicy())
    [row] = res.rows
    assert row.outcome["verdict"] == "success"
    [ep] = res.episodes
    assert ep.stop_reason == "stop" and ep.final_state == {"shipment_decision": "ready"}


def test_harness_stops_at_max_steps():
    res = _harness(ScriptedPolicy(forever=True), max_steps=3)
    assert res.episodes[0].stop_reason == "max_steps"


def test_sidecar_round_trips(tmp_path):
    res = _harness(ScriptedPolicy())
    p = tmp_path / "sidecar.jsonl"
    text = write_sidecar(res, str(p))
    assert read_sidecar(str(p)) == sidecar_records(res)
    assert all(json.loads(line) for line in text.splitlines() if line.strip())
