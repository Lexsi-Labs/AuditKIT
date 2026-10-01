"""Tests for auditkit.metrics.retrieval: hand-computed ranking metrics."""

from __future__ import annotations

import math

import pytest

import auditkit as ak
from auditkit.metrics.retrieval import RetrievalMetrics, ranking_scores, retrieved_contexts
from auditkit.sample import Sample
from auditkit.types import ScoreKind

L3 = math.log2(3)


def scores(metric, reference, retrieved=None, trace=None):
    s = Sample(input="q", reference_contexts=reference, retrieval_context=retrieved)
    ctx = {"trace": trace} if trace is not None else None
    return {x.name: x.value for x in metric.score(s, "", ctx)}


def test_binary_relevance_hand_computed():
    got = ranking_scores(["a", "x", "b"], {"a": 1.0, "b": 1.0, "c": 1.0})
    assert got == pytest.approx({
        "hit_rate": 1.0, "precision": 2 / 3, "recall": 2 / 3, "mrr": 1.0,
        "average_precision": (1 + 2 / 3) / 3,             # divided by ALL 3 relevant
        "ndcg": (1 + 1 / 2) / (1 + 1 / L3 + 1 / 2),
    })


def test_cutoff_at_k():
    got = ranking_scores(["a", "x", "b"], {"a": 1.0, "b": 1.0, "c": 1.0}, k=2)
    assert got == pytest.approx({
        "hit_rate": 1.0, "precision": 1 / 2, "recall": 1 / 3, "mrr": 1.0,
        "average_precision": 1 / 2,                        # / min(k, n_relevant)
        "ndcg": 1 / (1 + 1 / L3),
    })


def test_first_hit_late():
    got = ranking_scores(["x", "y", "a"], {"a": 1.0})
    assert got["mrr"] == pytest.approx(1 / 3) and got["average_precision"] == pytest.approx(1 / 3)
    assert got["ndcg"] == pytest.approx(1 / 2)  # 1/log2(4)
    assert ranking_scores(["x", "y", "a"], {"a": 1.0}, k=2)["hit_rate"] == 0.0


def test_graded_relevance_ndcg():
    got = ranking_scores(["b", "a"], {"a": 3.0, "b": 1.0})
    assert got["ndcg"] == pytest.approx((1 + 3 / L3) / (3 + 1 / L3))
    assert got["recall"] == 1.0 and got["precision"] == 1.0
    assert ranking_scores(["a", "b"], {"a": 3.0, "b": 1.0})["ndcg"] == pytest.approx(1.0)


def test_k_larger_than_list_divides_precision_by_k():
    got = ranking_scores(["a"], {"a": 1.0, "b": 1.0}, k=5)
    assert got == pytest.approx({
        "hit_rate": 1.0, "precision": 1 / 5, "recall": 1 / 2, "mrr": 1.0,
        "average_precision": 1 / 2, "ndcg": 1 / (1 + 1 / L3),
    })


def test_duplicates_count_once_but_hold_their_rank():
    # A duplicate is counted once but keeps its slot, so 'b' stays at rank 3 --
    # it is not lifted to rank 2 (finding 11): mrr 1/3, not 1/2.
    got = ranking_scores(["x", "x", "b"], {"b": 1.0})
    assert got["mrr"] == pytest.approx(1 / 3) and got["precision"] == pytest.approx(1 / 3)
    # rank-based metrics must reflect 'b' at rank 3, not an inflated rank 2.
    assert got["average_precision"] == pytest.approx(1 / 3)
    assert got["ndcg"] == pytest.approx(1 / 2)  # 1/log2(4), not 1/log2(3)
    # 'a' relevant once; the repeat is a wasted (zero-relevance) slot.
    got = ranking_scores(["a", "a", "b"], {"a": 1.0})
    assert got["recall"] == 1.0 and got["precision"] == pytest.approx(1 / 3)


def test_duplicates_do_not_pull_items_past_k_into_the_top_k():
    # The retriever's top-2 slots hold [a, a]; b sits at rank 4.
    got = ranking_scores(["a", "a", "a", "b"], {"a": 1.0, "b": 1.0}, k=2)
    assert got["recall"] == 0.5 and got["precision"] == 0.5
    assert got["average_precision"] == 0.5
    assert got["ndcg"] == pytest.approx(1 / (1 + 1 / L3))


def test_empty_retrieval_scores_zero():
    for k in (None, 3):
        assert set(ranking_scores([], {"a": 1.0}, k=k).values()) == {0.0}
    at = scores(RetrievalMetrics(k=3), ["a"], [])
    assert set(at.values()) == {0.0} and "recall@3" in at


def test_empty_reference_is_skipped():
    m = RetrievalMetrics()
    assert scores(m, [], ["a"]) == {}
    assert scores(m, {}, ["a"]) == {}
    assert scores(m, {"a": 0}, ["a"]) == {}  # zero grade = not relevant
    assert not m.applicable(Sample(input="q", retrieval_context=["a"]))


def test_zero_grades_excluded_from_relevant_set():
    got = scores(RetrievalMetrics(), {"a": 0, "b": 2}, ["a", "b"])
    assert got["recall"] == 1.0 and got["mrr"] == 0.5


def test_whitespace_is_stripped_on_both_sides():
    assert scores(RetrievalMetrics(), ["doc 1 "], [" doc 1\n"])["hit_rate"] == 1.0


def test_trace_contexts_override_sample_retrieval_context():
    m = RetrievalMetrics()
    assert scores(m, ["a"], ["x"], trace={"retrieved_contexts": ["a"]})["hit_rate"] == 1.0
    # an explicit empty retrieval in the trace wins too
    assert scores(m, ["a"], ["a"], trace={"retrieved_contexts": []})["hit_rate"] == 0.0
    # a trace without retrieved_contexts falls back to the sample
    assert scores(m, ["a"], ["a"], trace={"tool_calls": []})["hit_rate"] == 1.0
    assert retrieved_contexts(Sample(input="q", retrieval_context=[1, 2]), None) == ["1", "2"]


def test_names_kind_metadata_and_identity():
    s = Sample(input="q", reference_contexts=["a", "b"], retrieval_context=["a", "a", "c"])
    out = RetrievalMetrics(k=5).score(s, "")
    assert [x.name for x in out] == ["hit_rate@5", "precision@5", "recall@5", "mrr@5",
                                     "average_precision@5", "ndcg@5"]
    assert all(x.kind == ScoreKind.RAG for x in out)
    assert out[0].metadata == {"n_retrieved": 3, "n_relevant": 2}
    assert [x.name for x in RetrievalMetrics().score(s, "")][0] == "hit_rate"
    assert RetrievalMetrics(k=5).name == "retrieval@5" and RetrievalMetrics().name == "retrieval"
    assert RetrievalMetrics(k=5).identity() != RetrievalMetrics(k=10).identity()
    for bad in (0, -1):
        with pytest.raises(ValueError):
            RetrievalMetrics(k=bad)


def test_evaluate_precomputed_with_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = [
        Sample(input="q1", actual_output="a1", reference_contexts=["d1", "d2"],
               retrieval_context=["zzz"], actual_trace={"retrieved_contexts": ["d1", "x", "d2"]}),
        Sample(input="q2", actual_output="a2", reference_contexts={"d3": 2, "d4": 1},
               retrieval_context=["d4", "d3"]),
        Sample(input="q3", actual_output="a3", retrieval_context=["d1"]),  # no gold: skipped
    ]
    r = ak.evaluate(samples, model="precomputed", scorers=[RetrievalMetrics(k=2)])
    assert r.errors == []
    assert r.stats["recall@2"].count == 2
    assert r.headline["recall@2"] == pytest.approx((1 / 2 + 1) / 2)
    assert r.headline["mrr@2"] == 1.0
    assert r.headline["ndcg@2"] == pytest.approx((1 / (1 + 1 / L3) + (1 + 2 / L3) / (2 + 1 / L3)) / 2)


def test_plain_string_contexts_are_one_chunk_not_characters():
    from auditkit.metrics.retrieval import RetrievalMetrics
    from auditkit.sample import Sample
    scores = RetrievalMetrics().score(Sample(input="q", retrieval_context="abc", reference_contexts="abc"), "")
    assert {s.name: s.value for s in scores}["recall"] == 1.0


def test_non_finite_grade_errors_the_sample_and_output_stays_strict_json(monkeypatch, tmp_path):
    import json
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = [Sample(input="q0", id="r0", actual_output="", reference_contexts={"a": math.inf, "b": 1.0},
                      retrieval_context=["a", "b"]),
               Sample(input="q1", id="r1", actual_output="", reference_contexts={"a": 1.0},
                      retrieval_context=["a"])]
    r = ak.evaluate(samples, model="precomputed", scorers=[RetrievalMetrics()])
    assert len(r.errors) == 1 and "finite" in r.errors[0]["error"]
    assert r.headline["ndcg"] == 1.0
    json.dumps(r.to_dict(), allow_nan=False)
    with pytest.raises(ValueError):
        scores(RetrievalMetrics(), {"a": float("nan")}, ["a"])
    with pytest.raises(ValueError):
        ranking_scores(["a"], {"a": math.inf})
