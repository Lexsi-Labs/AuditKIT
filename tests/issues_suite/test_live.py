"""Live, opt-in (AK_LIVE_HF=1): a real Qwen3 behind servers that behave like SGLang, vLLM, a
gateway and a bare server, driven through api: the way a user drives a deployment.
"""

from __future__ import annotations

import csv
import json
import logging

import pytest

import auditkit as ak
from auditkit.model.api_gen import APIModel
from auditkit.runspec import RunConfig

from .conftest import served_requests

pytestmark = pytest.mark.live

QA = [ak.Sample(id="fr", input="What is the capital of France? Answer with one word.", target="Paris"),
      ak.Sample(id="jp", input="What is the capital of Japan? Answer with one word.", target="Tokyo")]
NO_THINK = RunConfig(chat_template_kwargs={"enable_thinking": False}, max_tokens=64)


def api(live, persona, **kw):
    return APIModel(live["model"], api_base=live[persona], api_key="EMPTY", name=f"api:{persona}", **kw)


# -- #43: fields a server may ignore -------------------------------------------------------------------

def test_sglang_probed_answers_but_the_note_stays_until_declared(live):
    r = ak.evaluate(QA, model=api(live, "sglang"), scorers=["quasi_exact_match"], config=NO_THINK)
    assert r.headline["quasi_exact_match"] == 1.0 and not r.errors          # the server did apply it
    assert r.metadata["api_server"] == {"kind": "sglang", "source": "probed", "owned_by": "sglang"}
    assert "server='sglang'" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]
    declared = ak.evaluate(QA, model=api(live, "sglang", server="sglang"), scorers=["quasi_exact_match"], config=NO_THINK)
    assert "unverified_request_fields" not in declared.metadata
    assert declared.headline == r.headline


def test_a_gateway_that_drops_the_field_is_flagged_and_it_shows(live, caplog):
    # the gateway forwards without chat_template_kwargs, so Qwen3 thinks and runs out of tokens
    with caplog.at_level(logging.WARNING, logger="auditkit.model.api_gen"):
        r = ak.evaluate(QA, model=api(live, "proxy"), scorers=["quasi_exact_match"], config=NO_THINK)
    assert "owned_by='openai'" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]
    assert "chat_template_kwargs" in r.summary()
    assert r.errors or r.headline.get("quasi_exact_match", 0.0) < 1.0, "the dropped field has a visible effect"
    assert sum("may have been ignored" in m for m in caplog.messages) == 1


def test_a_bare_server_is_unknown(live):
    r = ak.evaluate(QA[:1], model=api(live, "bare"), scorers=["quasi_exact_match"], config=NO_THINK)
    assert r.metadata["api_server"]["source"] == "unknown"
    assert "no GET /models answer" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]


def test_a_user_who_declares_the_wrong_server_keeps_the_note(live):
    r = ak.evaluate(QA[:1], model=api(live, "proxy", server="tgi"), scorers=["quasi_exact_match"], config=NO_THINK)
    assert "declared server 'tgi'" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]


def parallel_samples(bfcl_root, n=4):
    return ak.load_bfcl(str(bfcl_root / "BFCL_v3_parallel.json"), limit=n)


def test_the_one_call_cap_on_sglang_auto_is_marked_unverified(live, bfcl_root):
    r = ak.evaluate(parallel_samples(bfcl_root), model=api(live, "sglang"), config=NO_THINK,
                    adapter=ak.ToolCallAdapter(parallel_tool_calls=False),
                    scorers=[ak.ParallelToolCalls(arg_match=ak.bfcl_arg_match)])
    docs = [d for p in r.predictions for d in p.metadata.get("scores", []) if d["name"].startswith("parallel_")]
    assert docs and all(d["metadata"]["cap_unverified"] for d in docs)
    # and the cap really was not applied: the model still batched
    assert any(len(t) > 1 for p in r.predictions for t in ((p.context or {}).get("trace") or {}).get("tool_calls", []))


def test_the_cap_on_declared_vllm_is_applied_and_verified(live, bfcl_root):
    r = ak.evaluate(parallel_samples(bfcl_root), model=api(live, "vllm", server="vllm"), config=NO_THINK,
                    adapter=ak.ToolCallAdapter(parallel_tool_calls=False),
                    scorers=[ak.ParallelToolCalls(arg_match=ak.bfcl_arg_match)])
    assert "unverified_request_fields" not in r.metadata
    assert all(len(t) <= 1 for p in r.predictions for t in ((p.context or {}).get("trace") or {}).get("tool_calls", []))


# -- #45: BFCL through a real model --------------------------------------------------------------------

SLICE = {"simple": 6, "multiple": 3, "parallel": 3, "live_simple": 3, "sql": 2, "java": 2, "irrelevance": 3}


def test_bfcl_single_turn_through_a_served_model(live, bfcl_root):
    samples = [s for c, n in SLICE.items()
               for s in ak.load_bfcl(str(bfcl_root / f"BFCL_v3_{c}.json"), limit=n, revision="61fc060")]
    r = ak.evaluate(samples, model=api(live, "vllm", server="vllm"), config=RunConfig(
        chat_template_kwargs={"enable_thinking": False}, max_tokens=384),
        adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1(arg_match=ak.bfcl_arg_match), ak.ToolCallValidity()])
    per = {}
    for s, p in zip(samples, r.predictions):
        f1 = next(d["value"] for d in p.metadata["scores"] if d["name"].startswith("tool_call_f1"))
        per.setdefault(s.metadata["bfcl_category"], []).append(f1)
    print("\nBFCL tool_call_f1 by category:", {c: round(sum(v) / len(v), 3) for c, v in per.items()})
    assert not r.errors
    assert sum(sum(v) for v in per.values()) / len(samples) > 0.5
    assert min(per["irrelevance"]) >= 0.0 and len(per) == len(SLICE)


# -- #44: a reloaded dataset is not re-billed ------------------------------------------------------------

def test_a_csv_copy_of_the_dataset_hits_the_cache_against_a_real_server(live, bfcl_root, tmp_path):
    samples = ak.load_bfcl(str(bfcl_root / "BFCL_v3_simple.json"), limit=3)
    m = api(live, "vllm", server="vllm")
    kw = dict(adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1(arg_match=ak.bfcl_arg_match)],
              config=RunConfig(chat_template_kwargs={"enable_thinking": False}, max_tokens=256))
    first = ak.evaluate(samples, model=m, **kw)
    before = served_requests(live["vllm"])["vllm"]
    path = tmp_path / "bfcl_simple.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "input", "tools", "expected_tool_calls", "metadata"])
        for s in samples:
            w.writerow([s.id, s.input, json.dumps(s.tools), json.dumps(s.expected_tool_calls), json.dumps(s.metadata)])
    with path.open(newline="") as f:
        again = [ak.Sample(id=r["id"], input=r["input"], tools=json.loads(r["tools"]),
                           expected_tool_calls=r["expected_tool_calls"], metadata=json.loads(r["metadata"]),
                           task=f"bfcl_simple") for r in csv.DictReader(f)]
    second = ak.evaluate(again, model=m, **kw)
    assert served_requests(live["vllm"])["vllm"] == before, "the reloaded dataset must not call the server"
    assert second.headline == first.headline
