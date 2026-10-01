"""RAG metrics against hand-computed expectations (no model, no network).

- retrieval(k): hit_rate / precision / recall / mrr / average_precision / ndcg
- LLM-judged: faithfulness, context_precision (+ context_relevance), context_recall
  (a scripted judge stands in for the model so every verdict is known)
- lexical: lexical_groundedness, context_coverage, context_overlap, answer_overlap
"""

from __future__ import annotations

import math

import pytest

import auditkit as ak
from auditkit.metrics.rag import AnswerOverlap, ContextCoverage, ContextOverlap, LexicalGroundedness
from auditkit.metrics.rag_judge import ContextPrecision, ContextRecall, Faithfulness, parse_verdicts
from auditkit.metrics.retrieval import RetrievalMetrics
from auditkit.model import CallableModel
from auditkit.sample import Sample

L2 = math.log2


def ret(ranked, ref, k=None, via_trace=True):
    s = Sample(input="q", reference_contexts=ref, retrieval_context=None if via_trace else ranked)
    ctx = {"trace": {"retrieved_contexts": ranked}} if via_trace else {}
    return {x.name: x.value for x in RetrievalMetrics(k=k).score(s, "", ctx)}


def R(hit, p, r, mrr, ap, ndcg, k=None):
    at = f"@{k}" if k else ""
    return {f"hit_rate{at}": hit, f"precision{at}": p, f"recall{at}": r, f"mrr{at}": mrr,
            f"average_precision{at}": ap, f"ndcg{at}": ndcg}


# -- retrieval(k) -------------------------------------------------------------------------

RET_CASES = [
    ("basic", ["a", "b", "c", "d"], ["b", "d"], None,
     R(1, .5, 1, .5, .5, (1 / L2(3) + 1 / L2(5)) / (1 + 1 / L2(3)))),
    ("basic-at-2", ["a", "b", "c", "d"], ["b", "d"], 2,
     R(1, .5, .5, .5, .25, (1 / L2(3)) / (1 + 1 / L2(3)), k=2)),
    ("perfect-order", ["b", "d", "a"], ["b", "d"], None, R(1, 2 / 3, 1, 1, 1, 1)),
    ("nothing-relevant", ["a", "c"], ["b"], None, R(0, 0, 0, 0, 0, 0)),
    ("empty-retrieval", [], ["b"], None, R(0, 0, 0, 0, 0, 0)),
    ("graded-nDCG", ["b", "a"], {"a": 3, "b": 1}, None,
     R(1, 1, 1, 1, 1, (1 + 3 / L2(3)) / (3 + 1 / L2(3)))),
    ("zero-grade-ignored", ["z", "a"], {"a": 1, "z": 0}, None, R(1, .5, 1, .5, .5, 1 / L2(3))),
    ("k-beyond-list", ["a"], ["a"], 5, R(1, .2, 1, 1, 1, 1, k=5)),
    ("whitespace-normalised", [" a "], ["a"], None, R(1, 1, 1, 1, 1, 1)),
    ("reference-as-string", ["x", "a"], "a", None, R(1, .5, 1, .5, .5, 1 / L2(3))),
    # A duplicate keeps its rank slot at zero relevance: 'a' is at rank 3, not 2,
    # and the repeat counts as a (non-relevant) slot for precision.
    ("duplicate-before-relevant", ["x", "x", "a"], ["a"], None, R(1, 1 / 3, 1, 1 / 3, 1 / 3, 1 / L2(4))),
    ("duplicate-relevant", ["a", "a", "b"], ["a", "b"], None,
     R(1, 2 / 3, 1, 1, (1 + 2 / 3) / 2, (1 + 1 / L2(4)) / (1 + 1 / L2(3)))),
    ("duplicate-uses-k-slot", ["x", "x", "a"], ["a"], 2, R(0, 0, 0, 0, 0, 0, k=2)),
]


@pytest.mark.parametrize("ranked,ref,k,want", [x[1:] for x in RET_CASES], ids=[x[0] for x in RET_CASES])
def test_retrieval_metrics(ranked, ref, k, want):
    assert ret(ranked, ref, k) == pytest.approx(want)


def test_retrieval_trace_wins_over_dataset_contexts():
    s = Sample(input="q", reference_contexts=["gold"], retrieval_context=["stale"])
    got = {x.name: x.value for x in RetrievalMetrics().score(s, "", {"trace": {"retrieved_contexts": ["gold"]}})}
    assert got["hit_rate"] == 1.0


def test_retrieval_falls_back_to_dataset_contexts():
    assert ret(["x", "gold"], ["gold"], via_trace=False)["mrr"] == .5


def test_retrieval_skipped_without_gold_and_rejects_bad_k():
    assert not RetrievalMetrics().applicable(Sample(input="q"))
    assert RetrievalMetrics().score(Sample(input="q", reference_contexts={"a": 0}), "", {}) == []
    with pytest.raises(ValueError):
        RetrievalMetrics(k=0)


# -- scripted judge --------------------------------------------------------------------------

class Judge:
    """Answers each RAG prompt type with a fixed reply; records every prompt."""

    def __init__(self, claims="", verdicts="", precision="", recall=""):
        self.replies = {"claims": claims, "verdicts": verdicts, "precision": precision, "recall": recall}
        self.prompts: list[str] = []

    def __call__(self, prompts):
        out = []
        for p in prompts:
            self.prompts.append(p)
            if "Break the ANSWER" in p:
                out.append(self.replies["claims"])
            elif "whether the CONTEXT supports" in p:
                out.append(self.replies["verdicts"])
            elif "useful for answering" in p:
                out.append(self.replies["precision"])
            else:
                out.append(self.replies["recall"])
        return out


CTX = ["Laptops are replaced every 3 years.", "Guest wifi rotates every Monday.", "Phones are replaced every 2 years."]


def rag_sample(output="Laptops are replaced every 3 years.", target="Laptops are replaced every 3 years.",
               contexts=CTX, **kw):
    return Sample(id="s", input="How often are laptops replaced?", target=target, actual_output=output,
                  actual_trace={"retrieved_contexts": contexts}, **kw)


def run(metric, sample):
    r = ak.evaluate([sample], model="precomputed", scorers=[metric])
    return {s["name"]: s for s in r.predictions[0].metadata.get("scores", [])}, r.errors


# faithfulness

def test_faithfulness_supported_unsupported_contradicted():
    j = Judge(claims="1. Laptops are replaced every 3 years.\n2. Laptops are free.\n3. Laptops are replaced yearly.",
              verdicts="1: SUPPORTED\n2: UNSUPPORTED\n3: CONTRADICTED")
    got, err = run(Faithfulness(judge_model=CallableModel(j)), rag_sample())
    assert err == [] and got["faithfulness"]["value"] == pytest.approx(1 / 3)
    assert got["faithfulness"]["metadata"]["n_contradicted"] == 1


@pytest.mark.parametrize("verdicts", [
    "**1**: SUPPORTED\n**2**: SUPPORTED",
    "1) supported - stated in chunk 1\n2) Supported",
    "Here are my verdicts:\n1: SUPPORTED\nsome commentary\n2: SUPPORTED",
    "1 - `SUPPORTED`\n2 - `SUPPORTED`",
])
def test_faithfulness_tolerates_decorated_verdicts(verdicts):
    j = Judge(claims="1. a\n2. b", verdicts=verdicts)
    got, err = run(Faithfulness(judge_model=CallableModel(j)), rag_sample())
    assert err == [] and got["faithfulness"]["value"] == 1.0


def test_unsupported_is_not_read_as_supported():
    assert parse_verdicts("1: UNSUPPORTED", 1, ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")) == ["UNSUPPORTED"]
    assert parse_verdicts("1: NOT_ATTRIBUTED", 1, ("ATTRIBUTED", "NOT_ATTRIBUTED")) == ["NOT_ATTRIBUTED"]
    assert parse_verdicts("1: not attributed", 1, ("ATTRIBUTED", "NOT_ATTRIBUTED")) == ["NOT_ATTRIBUTED"]


@pytest.mark.parametrize("verdicts", ["1: SUPPORTED", "1: SUPPORTED\n2: SUPPORTED\n3: SUPPORTED", "no idea"])
def test_faithfulness_wrong_verdict_count_is_an_error_not_a_zero(verdicts):
    j = Judge(claims="1. a\n2. b", verdicts=verdicts)
    got, err = run(Faithfulness(judge_model=CallableModel(j)), rag_sample())
    assert "faithfulness" not in got and len(err) == 1


def test_faithfulness_conflicting_duplicate_verdict_is_an_error():
    j = Judge(claims="1. a\n2. b", verdicts="1: SUPPORTED\n2: SUPPORTED\n1: CONTRADICTED")
    got, err = run(Faithfulness(judge_model=CallableModel(j)), rag_sample())
    assert "faithfulness" not in got and len(err) == 1


@pytest.mark.parametrize("claims", ["NONE", "1. NONE", "**NONE**"])
def test_faithfulness_no_claims_is_skipped(claims):
    got, err = run(Faithfulness(judge_model=CallableModel(Judge(claims=claims))), rag_sample(output="I can't say."))
    assert got == {} and err == []


def test_faithfulness_skips_without_contexts_or_answer():
    j = Judge()
    assert run(Faithfulness(judge_model=CallableModel(j)), rag_sample(contexts=[]))[0] == {}
    assert run(Faithfulness(judge_model=CallableModel(j)), rag_sample(output="   "))[0] == {}
    assert j.prompts == []


def test_faithfulness_strips_judge_reasoning():
    j = Judge(claims="<think>1. draft claim\n2. another</think>1. Laptops are replaced every 3 years.",
              verdicts="<think>1: CONTRADICTED</think>1: SUPPORTED")
    got, err = run(Faithfulness(judge_model=CallableModel(j)), rag_sample())
    assert got["faithfulness"]["value"] == 1.0 and got["faithfulness"]["metadata"]["claims"] == \
        ["Laptops are replaced every 3 years."]


def test_faithfulness_does_not_judge_the_answers_own_reasoning():
    """A reasoning model's answer: only the text after </think> is the answer."""
    j = Judge(claims="1. Laptops are replaced every 3 years.", verdicts="1: SUPPORTED")
    run(Faithfulness(judge_model=CallableModel(j)),
        rag_sample(output="<think>Maybe every 5 years? Or yearly?</think>Laptops are replaced every 3 years."))
    assert "every 5 years" not in j.prompts[0]


def test_faithfulness_uses_retrieved_contexts_and_truncates():
    j = Judge(claims="1. x", verdicts="1: SUPPORTED")
    run(Faithfulness(judge_model=CallableModel(j), max_context_chars=40), rag_sample(contexts=["LIVE-CHUNK " * 20]))
    assert "LIVE-CHUNK" in j.prompts[1] and "(truncated)" in j.prompts[1]


# context_precision

@pytest.mark.parametrize("verdicts,precision,relevance", [
    ("1: RELEVANT\n2: IRRELEVANT\n3: RELEVANT", (1 + 2 / 3) / 2, 2 / 3),
    ("1: IRRELEVANT\n2: RELEVANT\n3: IRRELEVANT", .5, 1 / 3),
    ("1: RELEVANT\n2: RELEVANT\n3: IRRELEVANT", 1.0, 2 / 3),
    ("1: IRRELEVANT\n2: IRRELEVANT\n3: IRRELEVANT", 0.0, 0.0),
])
def test_context_precision(verdicts, precision, relevance):
    got, err = run(ContextPrecision(judge_model=CallableModel(Judge(precision=verdicts))), rag_sample())
    assert err == []
    assert got["context_precision"]["value"] == pytest.approx(precision)
    assert got["context_relevance"]["value"] == pytest.approx(relevance)


def test_context_precision_includes_reference_only_when_set():
    j = Judge(precision="1: RELEVANT\n2: IRRELEVANT\n3: IRRELEVANT")
    run(ContextPrecision(judge_model=CallableModel(j)), rag_sample(target="Every 3 years."))
    run(ContextPrecision(judge_model=CallableModel(j)), rag_sample(target=None))
    assert "<reference_answer>" in j.prompts[0] and "<reference_answer>" not in j.prompts[1]


def test_context_precision_count_mismatch_is_error():
    got, err = run(ContextPrecision(judge_model=CallableModel(Judge(precision="1: RELEVANT"))), rag_sample())
    assert got == {} and len(err) == 1


# context_recall

def test_context_recall_per_sentence():
    j = Judge(recall="1: ATTRIBUTED\n2: NOT_ATTRIBUTED\n3: ATTRIBUTED")
    got, err = run(ContextRecall(judge_model=CallableModel(j)),
                   rag_sample(target="Laptops are replaced every 3 years. Monitors every 6. Phones every 2 years."))
    assert got["context_recall"]["value"] == pytest.approx(2 / 3)
    assert "Monitors every 6." in got["context_recall"]["reason"]


def test_context_recall_sentence_split_keeps_decimals():
    j = Judge(recall="1: ATTRIBUTED")
    got, _ = run(ContextRecall(judge_model=CallableModel(j)), rag_sample(target="Laptops last 3.5 years on average."))
    assert got["context_recall"]["metadata"]["sentences"] == ["Laptops last 3.5 years on average."]


def test_context_recall_needs_target():
    assert not ContextRecall(judge_model="x").applicable(Sample(input="q"))


# -- lexical RAG metrics -----------------------------------------------------------------------

def lex(metric, output, contexts=None, target=None, trace=None):
    s = Sample(input="q", target=target, retrieval_context=contexts)
    out = metric.score(s, output, {"trace": {"retrieved_contexts": trace}} if trace else {})
    out = out if isinstance(out, list) else [out]
    return {x.name: x.value for x in out}


LEX_CASES = [
    ("groundedness-exact", LexicalGroundedness(), "Paris is in France", ["Paris is in France"], None, 1.0),
    ("groundedness-trailing-punctuation", LexicalGroundedness(), "Paris is in France.", ["Paris is in France"], None, 1.0),
    ("groundedness-case-insensitive", LexicalGroundedness(), "paris is in france", ["Paris is in France"], None, 1.0),
    ("groundedness-unsupported-word", LexicalGroundedness(), "Paris is in Spain", ["Paris is in France"], None, .75),
    ("coverage-punctuation", ContextCoverage(), "", ["Paris is in France"], "Paris.", 1.0),
    ("coverage-case", ContextCoverage(), "", ["paris is in france"], "Paris", 1.0),
    ("coverage-partial", ContextCoverage(), "", ["Paris is big"], "Paris is in France", .5),
    ("overlap-case", ContextOverlap(), "PARIS", ["Paris is in France", "Rome is in Italy"], None, .5),
    ("answer-overlap-case", AnswerOverlap(), "paris", None, "Paris", 1.0),
    ("answer-overlap-partial", AnswerOverlap(), "Paris is nice", None, "Paris is in France", .5),
]


@pytest.mark.parametrize("metric,output,contexts,target,want", [x[1:] for x in LEX_CASES], ids=[x[0] for x in LEX_CASES])
def test_lexical_metrics(metric, output, contexts, target, want):
    assert lex(metric, output, contexts, target)[metric.name] == pytest.approx(want)


@pytest.mark.parametrize("metric", [LexicalGroundedness(), ContextOverlap()])
def test_lexical_metrics_read_live_retrieved_contexts(metric):
    """A live RAG endpoint returns its contexts in the trace; the lexical metrics must use them."""
    s = Sample(id="x", input="q", actual_output="Paris is in France",
               actual_trace={"retrieved_contexts": ["Paris is in France"]})
    r = ak.evaluate([s], model="precomputed", scorers=[metric])
    assert r.headline.get(metric.name) == 1.0


def test_lexical_trace_wins_over_dataset_contexts():
    got = lex(LexicalGroundedness(), "Paris", contexts=["Rome"], trace=["Paris"])
    assert got["lexical_groundedness"] == 1.0


def test_coverage_reads_live_contexts():
    s = Sample(id="x", input="q", target="Paris", actual_output="",
               actual_trace={"retrieved_contexts": ["Paris is in France"]})
    assert ak.evaluate([s], model="precomputed", scorers=[ContextCoverage()]).headline.get("context_coverage") == 1.0


# -- the RAG metrics together through evaluate() ----------------------------------------------

def test_rag_pipeline_end_to_end():
    j = Judge(claims="1. Laptops are replaced every 3 years.", verdicts="1: SUPPORTED",
              precision="1: RELEVANT\n2: IRRELEVANT\n3: IRRELEVANT", recall="1: ATTRIBUTED")
    judge = CallableModel(j)
    s = rag_sample(reference_contexts=["Laptops are replaced every 3 years."])
    r = ak.evaluate([s], model="precomputed", scorers=[
        RetrievalMetrics(k=3), Faithfulness(judge_model=judge), ContextPrecision(judge_model=judge),
        ContextRecall(judge_model=judge), LexicalGroundedness(), AnswerOverlap()])
    h = r.headline
    assert r.errors == []
    assert h["mrr@3"] == 1.0 and h["precision@3"] == pytest.approx(1 / 3)
    assert h["faithfulness"] == 1.0 and h["context_precision"] == 1.0 and h["context_relevance"] == pytest.approx(1 / 3)
    assert h["context_recall"] == 1.0 and h["lexical_groundedness"] == 1.0 and h["answer_overlap"] == 1.0
