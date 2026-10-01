"""Live RAG judge metrics with real judge models. Opt-in: ``AK_LIVE=1``.

Every case has a known right answer (e.g. a faithful answer must beat a
contradicted one), so a judge model can be *graded*:

  PIPELINE (hard): every sample either scores in range or is a recorded error;
                   a judge reply that can't be parsed is never a 0.
  ORDERING (quality, AK_LIVE_QUALITY=1, default on): faithful > hallucinated > contradicted,
                   relevant-first > relevant-last, full coverage > half coverage.
Per-judge parse-failure rates and every score go to live_reports/live_rag_report_<tag>.{json,md}.
Env: same as test_live.py (AK_LIVE_BASE, AK_LIVE_MODELS, AK_LIVE_TAG); every listed model is used as a judge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import auditkit as ak
from auditkit.sample import Sample

from .conftest import OLLAMA, live_enabled, live_models

pytestmark = [pytest.mark.live, pytest.mark.skipif(not live_enabled(), reason="set AK_LIVE=1 to run live tests")]

JUDGES = list(live_models()) if live_enabled() else []
QUALITY = os.environ.get("AK_LIVE_QUALITY", "1") == "1"
REPORT_DIR = Path(os.environ.get("AK_LIVE_REPORT_DIR", Path(__file__).parent / "live_reports"))
REPORT: dict = {}

Q = "What is the company's laptop policy?"
CTX = ["Employee laptops are replaced every 3 years through the IT portal.",
       "Laptops must use full-disk encryption.",
       "The cafeteria is open from 8am to 3pm."]
GOLD = "Laptops are replaced every 3 years through the IT portal. Laptops must use full-disk encryption."

FAITH = {  # id: (answer, expected band)
    # Stay as close to the context as a real faithful answer would: qwen3:8b rated "Laptops are replaced..."
    # UNSUPPORTED against "Employee laptops are replaced..." (a generalisation) -- see FINDINGS LIVE-3.
    "faithful": "Employee laptops are replaced every 3 years through the IT portal, and laptops must use "
                "full-disk encryption.",
    "hallucinated": "Laptops are replaced every 3 years through the IT portal, and each employee gets a free phone.",
    "contradicted": "Laptops are replaced every year, and encryption is optional.",
    "refusal": "I don't know.",
}
PRECISION = {"relevant-first": CTX, "relevant-last": [CTX[2], CTX[1], CTX[0]]}
RECALL = {"full": CTX[:2], "half": [CTX[0], CTX[2]]}


def judge_kwargs(model):
    return dict(judge_model=f"api:{model}", max_tokens=4096,
                judge_model_args={"api_base": f"{OLLAMA}/v1", "api_key": "EMPTY",
                                  # reasoning judges on a laptop can exceed the 120 s default per call
                                  "timeout": float(os.environ.get("AK_LIVE_JUDGE_TIMEOUT", "600"))})


def per_sample(r):
    return {p.sample_id: {s["name"]: s["value"] for s in p.metadata.get("scores", [])} for p in r.predictions}


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    if not REPORT:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("AK_LIVE_TAG", "local")
    (REPORT_DIR / f"live_rag_report_{tag}.json").write_text(json.dumps(REPORT, indent=1, default=str))
    lines = ["# RAG judge metrics: live report", ""]
    for judge, d in REPORT.items():
        lines += [f"## judge: {judge}", "", f"parse errors: {d['n_errors']} / {d['n_calls']} scored items", "",
                  "| metric | case | score |", "|---|---|---|"]
        errored = {(e.get("sample_id"), e.get("metric")) for e in d.get("errors", [])}
        for metric, cases in d["scores"].items():
            for case, v in cases.items():
                shown = f"{v:.2f}" if v is not None else ("error (recorded)" if (case, metric) in errored else "skipped")
                lines.append(f"| {metric} | {case} | {shown} |")
        mark = {True: "✅", False: "❌", None: "n/a (a score was missing)"}
        lines += ["", "ordering checks: " + ", ".join(f"{k}={mark[v]}" for k, v in d["ordering"].items()), ""]
    (REPORT_DIR / f"live_rag_report_{tag}.md").write_text("\n".join(lines))


@pytest.mark.parametrize("judge", JUDGES)
def test_live_rag_judges(judge):
    kw = judge_kwargs(judge)
    faith = [Sample(id=k, input=Q, target=GOLD, actual_output=a, actual_trace={"retrieved_contexts": CTX})
             for k, a in FAITH.items()]
    prec = [Sample(id=k, input=Q, target=GOLD, actual_output="", actual_trace={"retrieved_contexts": c})
            for k, c in PRECISION.items()]
    rec = [Sample(id=k, input=Q, target=GOLD, actual_output="", actual_trace={"retrieved_contexts": c})
           for k, c in RECALL.items()]
    runs = {"faithfulness": ak.evaluate(faith, model="precomputed", scorers=[ak.Faithfulness(**kw)]),
            "context_precision": ak.evaluate(prec, model="precomputed", scorers=[ak.ContextPrecision(**kw)]),
            "context_recall": ak.evaluate(rec, model="precomputed", scorers=[ak.ContextRecall(**kw)])}
    scores = {m: {sid: s.get(m) for sid, s in per_sample(r).items()} for m, r in runs.items()}
    n_err = sum(len(r.errors) for r in runs.values())
    n_calls = len(faith) + len(prec) + len(rec)

    f, p, c = scores["faithfulness"], scores["context_precision"], scores["context_recall"]
    ordering = {
        "faithful>hallucinated": _gt(f.get("faithful"), f.get("hallucinated")),
        "hallucinated>=contradicted": _ge(f.get("hallucinated"), f.get("contradicted")),
        "refusal-skipped": f.get("refusal") is None,
        "relevant-first>relevant-last": _gt(p.get("relevant-first"), p.get("relevant-last")),
        "full>half-recall": _gt(c.get("full"), c.get("half")),
    }
    REPORT[judge] = {"scores": scores, "n_errors": n_err, "n_calls": n_calls, "ordering": ordering,
                     "errors": [e for r in runs.values() for e in r.errors]}

    # PIPELINE: every value in range; a missing value must be an error or the refusal skip
    for m, cases in scores.items():
        for case, v in cases.items():
            assert v is None or 0.0 <= v <= 1.0, (m, case, v)
    recorded = {(e.get("sample_id"), e.get("metric")) for r in runs.values() for e in r.errors}
    for m, cases in scores.items():
        for case, v in cases.items():
            if v is None and not (m == "faithfulness" and case == "refusal"):
                assert any(sid == case for sid, _ in recorded), f"{m}/{case} missing without a recorded error"
    if QUALITY:
        failed = [k for k, ok in ordering.items() if ok is False]
        assert not failed, f"judge {judge} got the ordering wrong on: {failed} -- scores {scores}"


def _gt(a, b):
    return None if a is None or b is None else a > b


def _ge(a, b):
    return None if a is None or b is None else a >= b
