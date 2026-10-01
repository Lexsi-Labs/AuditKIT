"""Tests for the reference-free RAG metrics in auditkit.metrics.rag_judge.

Each metric needs no gold answer and no labelled relevant chunks; a scripted
stub judge (deterministic, no network) drives them. We assert the score, the
reason, that a parse failure is a recorded error (never a silent 0), and that
each runs with ``target=None``.
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metrics.rag_judge import (
    AnswerRelevancy, ContextRelevance, Hallucination, ResponseGroundedness,
)
from auditkit.model import CallableModel
from auditkit.sample import Sample


def stub(*, claims="", faith="", statements="", relevancy="", grounded="", relevance=""):
    """A judge that answers by prompt type; returns (model, prompts seen).

    Routing keys match the prompt tags/openers in rag_judge.py so each metric's
    two-step (extract then verdict) flow dispatches unambiguously.
    """
    prompts: list[str] = []

    def fn(ps):
        out = []
        for p in ps:
            prompts.append(p)
            if "<claims>" in p:
                out.append(faith)
            elif "<statements>" in p:
                out.append(relevancy)
            elif "<response_sentences>" in p:
                out.append(grounded)
            elif p.startswith("Break the ANSWER"):
                out.append(claims)
            elif p.startswith("Break the RESPONSE"):
                out.append(statements)
            else:  # the RELEVANT/IRRELEVANT context prompt
                out.append(relevance)
        return out

    return CallableModel(fn), prompts


def rag_sample(**kw):
    kw.setdefault("input", "What is the capital of France?")
    kw.setdefault("retrieval_context", ["Paris is the capital of France.", "France is in Europe."])
    return Sample(**kw)


# -- AnswerRelevancy (question + answer only) ---------------------------------

def test_answer_relevancy_scores_fraction_addressing():
    judge, prompts = stub(statements="1. Paris is the capital.\n2. I like croissants.",
                          relevancy="1: ADDRESSES\n2: OFF_TOPIC")
    m = AnswerRelevancy(judge_model=judge)
    # No context and no target: reference-free.
    s = m.score(Sample(input="What is the capital of France?"), "Paris is the capital. I like croissants.")
    assert len(s) == 1 and s[0].value == 0.5
    assert s[0].reason == "off-topic: I like croissants."
    assert "<statements>" in prompts[1]  # verdict step ran off the extracted statements


def test_answer_relevancy_no_target_field_required():
    assert AnswerRelevancy.required_fields == frozenset()
    assert AnswerRelevancy(judge_model=stub()[0]).applicable(Sample(input="q"))


def test_answer_relevancy_parse_failure_is_error_not_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Judge returns two verdicts for three statements -> parse_verdicts raises.
    judge, _ = stub(statements="1. A.\n2. B.\n3. C.", relevancy="1: ADDRESSES\n2: ADDRESSES")
    samples = [Sample(input="q", actual_output="A. B. C.")]
    r = ak.evaluate(samples, model="precomputed", scorers=[AnswerRelevancy(judge_model=judge)])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "answer_relevancy"
    assert "missing [3]" in r.errors[0]["error"]
    assert "answer_relevancy" not in r.stats and "answer_relevancy" not in r.headline


# -- ResponseGroundedness (contexts + answer, no gold) ------------------------

def test_response_groundedness_scores_supported_sentences():
    judge, _ = stub(grounded="1: SUPPORTED\n2: UNSUPPORTED")
    m = ResponseGroundedness(judge_model=judge)
    s = m.score(rag_sample(), "Paris is the capital. It has ten million people.")
    assert len(s) == 1 and s[0].value == 0.5
    assert s[0].reason == "not grounded: It has ten million people."


def test_response_groundedness_runs_without_target():
    judge, _ = stub(grounded="1: SUPPORTED")
    s = ResponseGroundedness(judge_model=judge).score(rag_sample(target=None), "Paris is the capital.")
    assert s[0].value == 1.0
    assert ResponseGroundedness.required_fields == frozenset()


def test_response_groundedness_no_context_is_skipped():
    judge, prompts = stub(grounded="1: SUPPORTED")
    assert ResponseGroundedness(judge_model=judge).score(rag_sample(retrieval_context=None), "x.") == []
    assert prompts == []


def test_response_groundedness_parse_failure_is_error_not_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    judge, _ = stub(grounded="1: SUPPORTED")  # one verdict, two sentences
    samples = [rag_sample(actual_output="A. B.", actual_trace={"retrieved_contexts": ["c1"]})]
    r = ak.evaluate(samples, model="precomputed", scorers=[ResponseGroundedness(judge_model=judge)])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "response_groundedness"
    assert "response_groundedness" not in r.headline


# -- Hallucination (contexts + answer, no gold; lower is better) --------------

def test_hallucination_is_one_minus_supported_fraction():
    judge, _ = stub(claims="1. Paris is the capital.\n2. Paris is in Asia.",
                    faith="1: SUPPORTED\n2: CONTRADICTED")
    s = Hallucination(judge_model=judge).score(rag_sample(), "Paris is the capital. Paris is in Asia.")
    assert len(s) == 1 and s[0].value == 0.5
    assert "CONTRADICTED: Paris is in Asia." in s[0].reason
    assert s[0].metadata["n_contradicted"] == 1


def test_hallucination_direction_is_minimize():
    from auditkit.types import Direction
    assert Hallucination.direction == Direction.MINIMIZE


def test_hallucination_refusal_is_skipped():
    judge, prompts = stub(claims="NONE")
    assert Hallucination(judge_model=judge).score(rag_sample(), "I cannot answer that.") == []
    assert len(prompts) == 1  # no verdict call once there are no claims


def test_hallucination_runs_without_target():
    judge, _ = stub(claims="1. Paris is the capital.", faith="1: SUPPORTED")
    s = Hallucination(judge_model=judge).score(rag_sample(target=None), "Paris is the capital.")
    assert s[0].value == 0.0
    assert Hallucination.required_fields == frozenset()


def test_hallucination_parse_failure_is_error_not_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    judge, _ = stub(claims="1. A.\n2. B.", faith="1: SUPPORTED")  # one verdict, two claims
    samples = [rag_sample(actual_output="A. B.", actual_trace={"retrieved_contexts": ["c1"]})]
    r = ak.evaluate(samples, model="precomputed", scorers=[Hallucination(judge_model=judge)])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "hallucination"
    assert "hallucination" not in r.headline


# -- ContextRelevance (question + contexts, no gold) --------------------------

def test_context_relevance_mean_of_relevant_chunks():
    judge, prompts = stub(relevance="1: RELEVANT\n2: IRRELEVANT")
    s = ContextRelevance(judge_model=judge).score(rag_sample(), "unused answer")
    assert len(s) == 1 and s[0].name == "context_relevance" and s[0].value == 0.5


def test_context_relevance_ignores_target_reference():
    # Reference-free: even with a target set, the prompt must not carry it.
    judge, prompts = stub(relevance="1: RELEVANT\n2: RELEVANT")
    s = ContextRelevance(judge_model=judge).score(rag_sample(target="Paris."), "answer")
    assert s[0].value == 1.0
    assert "REFERENCE ANSWER" not in prompts[0] and "reference_answer" not in prompts[0]
    assert ContextRelevance.required_fields == frozenset()


def test_context_relevance_no_context_is_skipped():
    judge, prompts = stub(relevance="1: RELEVANT")
    assert ContextRelevance(judge_model=judge).score(rag_sample(retrieval_context=None), "x") == []
    assert prompts == []


def test_context_relevance_parse_failure_is_error_not_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    judge, _ = stub(relevance="1: RELEVANT")  # one verdict, two chunks
    samples = [rag_sample(actual_output="a", actual_trace={"retrieved_contexts": ["c1", "c2"]})]
    r = ak.evaluate(samples, model="precomputed", scorers=[ContextRelevance(judge_model=judge)])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "context_relevance"
    assert "context_relevance" not in r.headline


# -- registration: found by name through ak.evaluate --------------------------

def test_reference_free_metrics_are_registered_by_name():
    for name in ("answer_relevancy", "response_groundedness", "hallucination", "context_relevance"):
        assert name in ak.METRICS.names()
