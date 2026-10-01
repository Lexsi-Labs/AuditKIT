"""The run fingerprint (cache key) for agentic/RAG runs: what must change it, what must not.

A fingerprint collision is a silent wrong answer: the second run is served the
first run's cached predictions. A needless change is only a cache miss.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import auditkit as ak
from auditkit.model import AutoModel, CallableModel
from auditkit.runspec import RunConfig, RunSpec
from auditkit.scenario import ListScenario

from .catalog import TOOLS, W


def samples(**over):
    base = dict(id="s1", input="Weather in Paris and Rome?", tools=TOOLS, expected_tool_calls=[[W("Paris"), W("Rome")]],
                target="sunny", reference_contexts=["d1"], actual_output="ok",
                actual_trace={"tool_calls": [[W("Paris"), W("Rome")]], "retrieved_contexts": ["d1", "d2"]})
    base.update(over)
    return [ak.Sample(**base)]


def fp(samples_=None, model=None, adapter=None, metrics=None, config=None):
    return RunSpec(scenario=ListScenario(samples_ or samples()), model=model or AutoModel.resolve("precomputed"),
                   adapter=adapter or ak.ToolCallAdapter(),
                   metrics=metrics if metrics is not None else [ak.ToolCallF1(), ak.RetrievalMetrics(k=3)],
                   config=config or RunConfig()).fingerprint()


def api(**kw):
    return AutoModel.resolve("api:qwen3:8b", **{"api_base": "http://a:1/v1", "api_key": "k1", **kw})


def faith(**args):
    return [ak.Faithfulness(judge_model="api:qwen3:8b", judge_model_args=args)]


def tc(**args):
    return [ak.TaskCompletion(judge_model="api:qwen3:8b", judge_model_args=args)]


BASE = None

MUST_CHANGE = {
    "input": lambda: fp(samples(input="Weather in Oslo?")),
    "expected_tool_calls": lambda: fp(samples(expected_tool_calls=[[W("Paris")]])),
    "tools": lambda: fp(samples(tools=[])),
    "reference_contexts": lambda: fp(samples(reference_contexts=["d2"])),
    "actual_trace": lambda: fp(samples(actual_trace={"tool_calls": [[W("Paris")], [W("Rome")]]})),
    "actual_output": lambda: fp(samples(actual_output="other")),
    "target": lambda: fp(samples(target="rainy")),
    "metadata": lambda: fp(samples(metadata={"messages": [{"role": "system", "content": "x"}]})),
    "user sample id": lambda: fp(samples(id="s2")),
    "adapter mode": lambda: fp(adapter=ak.ToolCallAdapter(mode="prompt")),
    "adapter tool_choice": lambda: fp(adapter=ak.ToolCallAdapter(tool_choice="required")),
    "arg_mode": lambda: fp(metrics=[ak.ToolCallF1(arg_mode="subset"), ak.RetrievalMetrics(k=3)]),
    "retrieval k": lambda: fp(metrics=[ak.ToolCallF1(), ak.RetrievalMetrics(k=5)]),
    "extra metric": lambda: fp(metrics=[ak.ToolCallF1(), ak.RetrievalMetrics(k=3), ak.ParallelToolCalls()]),
    "temperature": lambda: fp(config=RunConfig(temperature=0.7)),
    "max_tokens": lambda: fp(config=RunConfig(max_tokens=99)),
}


@pytest.mark.parametrize("name", list(MUST_CHANGE))
def test_field_changes_fingerprint(name):
    assert MUST_CHANGE[name]() != fp()


@pytest.mark.parametrize("a,b", [
    (lambda: fp(model=api()), lambda: fp(model=api(api_base="http://b:2/v1"))),
    (lambda: fp(metrics=faith(api_base="http://a:1/v1")), lambda: fp(metrics=faith(api_base="http://b:2/v1"))),
    (lambda: fp(metrics=tc(api_base="http://a:1/v1")), lambda: fp(metrics=tc(api_base="http://b:2/v1"))),
    (lambda: fp(metrics=[ak.Faithfulness(judge_model="api:qwen3:8b")]),
     lambda: fp(metrics=[ak.Faithfulness(judge_model="api:gemma3:12b")])),
    (lambda: fp(metrics=[ak.TaskCompletion(judge_model=CallableModel(lambda ps: ["CHOICE: complete"] * len(ps)))]),
     lambda: fp(metrics=[ak.TaskCompletion(judge_model=CallableModel(lambda ps: ["CHOICE: failed"] * len(ps)))])),
    (lambda: fp(metrics=[ak.ToolCallF1(arg_match=lambda n, p, r: True)]),
     lambda: fp(metrics=[ak.ToolCallF1(arg_match=lambda n, p, r: p == r)])),
    (lambda: fp(samples(id="a") + samples(id="b", input="x")), lambda: fp(samples(id="b", input="x") + samples(id="a"))),
], ids=["model server", "RAG judge server", "task_completion judge server", "judge model", "judge callable",
        "arg_match fn", "sample order"])
def test_pairs_that_must_differ(a, b):
    assert a() != b()


@pytest.mark.parametrize("a,b", [
    (lambda: fp(model=api()), lambda: fp(model=api(api_key="k2"))),
    (lambda: fp(metrics=faith(api_base="x", timeout=60, api_key="a")),
     lambda: fp(metrics=faith(api_base="x", timeout=600, api_key="b"))),
    (lambda: fp(metrics=[ak.RetrievalMetrics(k=3), ak.ToolCallF1()]), lambda: fp()),
    (lambda: fp(samples(actual_trace={"retrieved_contexts": ["d1", "d2"], "tool_calls": [[W("Paris"), W("Rome")]]})),
     lambda: fp()),
    (lambda: fp(samples(id=None)), lambda: fp(samples(id="0"))),   # Runner's auto id is not identity
], ids=["api_key", "judge timeout/api_key", "metric order", "dict key order", "auto id"])
def test_pairs_that_must_match(a, b):
    assert a() == b()


def test_auditkit_version_is_part_of_identity(monkeypatch):
    base = fp()
    monkeypatch.setattr(ak, "__version__", "9.9.9")
    assert fp() != base


def test_fingerprint_is_stable_across_processes():
    code = ("import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.');"
            "from tests.agentic_suite.test_fingerprint_offline import fp; print(fp())")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip() for seed in ("0", "1", "4242")}
    assert len(outs) == 1 and len(next(iter(outs))) == 16, outs


def test_cached_run_never_reports_another_datasets_ids():
    """Regression for FP-1: same content, different ids -> the second run must carry its own ids."""
    mk = lambda i: [ak.Sample(id=i, input="q", expected_tool_calls=[[W("x")]], actual_output="",
                              actual_trace={"tool_calls": [[W("x")]]})]
    a = ak.evaluate(mk("customer-A-row-1"), model="precomputed", scorers=[ak.ToolCallF1()])
    b = ak.evaluate(mk("customer-B-row-9"), model="precomputed", scorers=[ak.ToolCallF1()])
    assert a.fingerprint != b.fingerprint
    assert b.predictions[0].sample_id == "customer-B-row-9"


# task / kind in the fingerprint: covered by the fix in PR #6 (tests/test_fingerprint_stability.py).


def test_plain_datasets_keep_their_fingerprint():
    """Defaults (no task, generative kind, no id) add nothing to the hash."""
    from auditkit.scenario import ListScenario
    plain = ListScenario([ak.Sample(input="hi", target="hi")]).name
    assert plain == ListScenario([ak.Sample(input="hi", target="hi", task="", kind=ak.TaskKind.GENERATIVE)]).name


def test_id_less_samples_still_hit_the_cache_on_rerun():
    calls = []
    model = lambda prompts: (calls.append(1), ["hi" for _ in prompts])[1]
    s = [ak.Sample(input="hi", target="hi")]
    r1 = ak.evaluate(s, model=model, config=ak.RunConfig(concurrency=1))
    n = len(calls)
    r2 = ak.evaluate(s, model=model, config=ak.RunConfig(concurrency=1))   # Runner set s[0].id = "0" in run 1
    assert r1.fingerprint == r2.fingerprint and len(calls) == n
