"""Deterministic tests for the RAG-stress checks (no network, no LLM).

Each metric is proven with a positive and a negative control, plus the specific
failure it must locate independently (alternate evidence set, staleness given a
decision_date, cross-tenant hit, nonexistent/wrong-version citation, false
refusal), and the explicit unknown/not_tested result when gold or trace is
missing. One end-to-end test proves the Runner wiring: an ``actual_trace`` +
``actual_output`` sample scored through ``ak.evaluate(model="precomputed")``.
"""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.metrics.rag_stress import (
    Abstention, AclCompliance, CitationSupport, CorpusSnapshot, EvidenceSetRecall,
    Freshness, RAGCase, RAGTrace, make_doc, sha256_digest, synthetic_bank_kb,
)
from auditkit.sample import Sample


def _sample(case=None, corpus=None, **kw):
    meta = {}
    if case is not None:
        meta["rag_case"] = case
    if corpus is not None:
        meta["corpus"] = corpus
    kw.setdefault("input", case.question if case else "q")
    return Sample(metadata=meta, **kw)


def _ctx(trace: RAGTrace):
    return {"trace": trace.to_dict()}


def _by(scores, name):
    return next(s for s in scores if s.name == name)


def _is_unknown(score):
    return score.metadata.get("unknown") is True and score.label in ("unknown", "not_tested")


# -- contract dataclasses -----------------------------------------------------

def test_corpus_snapshot_digest_and_change_detection():
    d = sha256_digest("hello")
    assert d == sha256_digest("hello") and d != sha256_digest("hello!")
    a = CorpusSnapshot(documents={"X": make_doc(content="v1")})
    b = CorpusSnapshot(documents={"X": make_doc(content="v2")})
    assert a.changed(b) == {"X"}
    assert a.changed(CorpusSnapshot(documents={"X": make_doc(content="v1")})) == set()


def test_make_doc_redacts_raw_content():
    doc = make_doc(content="secret customer text", version="1")
    assert "content" not in doc and doc["digest"] == sha256_digest("secret customer text")


def test_contract_to_dict_is_strict_json():
    snap, cases = synthetic_bank_kb()
    trace = RAGTrace(query="q", retrieved=[{"id": "LP-CURRENT", "score": 0.9}],
                     cited_spans=[{"doc_id": "LP-CURRENT", "start": 0, "end": 10}])
    for obj in (snap, cases[0], cases[1], trace):
        json.dumps(obj.to_dict())  # raises if any non-JSON value slipped in
    # round-trips
    assert CorpusSnapshot.from_dict(snap.to_dict()).documents == snap.documents
    assert RAGCase.from_dict(cases[0].to_dict()).case_id == "C1-multihop"
    assert RAGTrace.from_dict(trace.to_dict()).retrieved_ids() == ["LP-CURRENT"]
    # to_dict exposes retrieved_contexts for the existing `retrieval` metric
    assert trace.to_dict()["retrieved_contexts"] == ["LP-CURRENT"]


def test_trace_coverage_missing_vs_observed_empty():
    assert RAGTrace().event_coverage("retrieved") == "missing"
    assert RAGTrace(retrieved=[]).event_coverage("retrieved") == "observed"
    assert RAGTrace(coverage={"retrieved": "inferred"}).event_coverage("retrieved") == "inferred"


# -- evidence_set_recall (RST-02) --------------------------------------------

def _multihop_case(**over):
    kw = dict(case_id="c", question="q",
              sufficient_evidence_sets=[["A", "B"], ["S"]], tenant="retail")
    kw.update(over)
    return RAGCase(**kw)


def test_evidence_set_recall_complete_set_passes():
    s = _sample(case=_multihop_case())
    sc = EvidenceSetRecall().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "A", "score": 1}, {"id": "B", "score": 1}])))[0]
    assert sc.value == 1.0


def test_evidence_set_recall_partial_set_fails_even_if_first():
    # A ranks first but B is missing -> incomplete -> fail (not enough).
    s = _sample(case=_multihop_case())
    sc = EvidenceSetRecall().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "A", "score": 1}, {"id": "Z", "score": 0.5}])))[0]
    assert sc.value == 0.0
    assert sc.metadata["best_set_recall"] == 0.5  # reused retrieval math


def test_evidence_set_recall_alternate_set_passes():
    s = _sample(case=_multihop_case())
    sc = EvidenceSetRecall().score(s, "", _ctx(RAGTrace(retrieved=[{"id": "S", "score": 1}])))[0]
    assert sc.value == 1.0


def test_evidence_set_recall_changed_digest_invalidates(monkeypatch=None):
    # A's pinned digest no longer matches the snapshot -> the {A,B} set is
    # dropped; the alternate {S} still passes.
    corpus = CorpusSnapshot(documents={"A": make_doc(content="new"), "B": make_doc(content="b"),
                                       "S": make_doc(content="s")})
    case = _multihop_case(evidence_digests={"A": sha256_digest("old")})
    s = _sample(case=case, corpus=corpus)
    sc = EvidenceSetRecall().score(s, "", _ctx(RAGTrace(retrieved=[{"id": "S", "score": 1}])))[0]
    assert sc.value == 1.0
    # if only the invalidated set could have matched, result is unknown
    sc2 = EvidenceSetRecall().score(
        _sample(case=_multihop_case(sufficient_evidence_sets=[["A", "B"]],
                                    evidence_digests={"A": sha256_digest("old")}), corpus=corpus),
        "", _ctx(RAGTrace(retrieved=[{"id": "A", "score": 1}, {"id": "B", "score": 1}])))[0]
    assert _is_unknown(sc2)


def test_evidence_set_recall_unknown_without_gold_or_trace():
    assert _is_unknown(EvidenceSetRecall().score(_sample(case=_multihop_case(
        sufficient_evidence_sets=[])), "", _ctx(RAGTrace(retrieved=[])))[0])
    assert _is_unknown(EvidenceSetRecall().score(_sample(case=_multihop_case()), "", {})[0])


# -- freshness (RST-01/03) ----------------------------------------------------

def _fresh_corpus():
    return CorpusSnapshot(documents={
        "CUR": make_doc(content="cur", version="2", effective_date="2026-01-01"),
        "OLD": make_doc(content="old", version="1", effective_date="2024-01-01",
                        superseded_date="2026-01-01"),
    })


def test_freshness_flags_stale_given_decision_date():
    case = RAGCase(case_id="c", question="q", decision_date="2026-07-01", tenant="retail")
    s = _sample(case=case, corpus=_fresh_corpus())
    scores = Freshness().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "CUR", "score": 1}, {"id": "OLD", "score": 0.9}])))
    assert _by(scores, "stale_hit_rate").value == 0.5
    assert _by(scores, "freshness").value == 0.5
    assert "OLD" in _by(scores, "freshness").metadata["stale"]


def test_freshness_positive_control_earlier_decision_date():
    # As of 2025, OLD was not yet superseded -> not stale. decision_date matters.
    case = RAGCase(case_id="c", question="q", decision_date="2025-06-01", tenant="retail")
    s = _sample(case=case, corpus=_fresh_corpus())
    scores = Freshness().score(s, "", _ctx(RAGTrace(retrieved=[{"id": "OLD", "score": 1}])))
    assert _by(scores, "stale_hit_rate").value == 0.0


def test_freshness_digest_mismatch_is_stale():
    corpus = _fresh_corpus()
    case = RAGCase(case_id="c", question="q", decision_date="2026-07-01", tenant="retail")
    s = _sample(case=case, corpus=corpus)
    scores = Freshness().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "CUR", "score": 1, "digest": "STALE-DIGEST"}])))
    assert _by(scores, "stale_hit_rate").value == 1.0


def test_freshness_unknown_without_dates_or_decision():
    assert _is_unknown(Freshness().score(_sample(case=RAGCase(case_id="c", question="q"),
                       corpus=_fresh_corpus()), "", _ctx(RAGTrace(retrieved=[])))[0])
    # decision_date present but retrieved doc carries no version info -> unknown
    corpus = CorpusSnapshot(documents={"N": make_doc(content="n")})
    case = RAGCase(case_id="c", question="q", decision_date="2026-01-01")
    assert _is_unknown(Freshness().score(_sample(case=case, corpus=corpus), "",
                       _ctx(RAGTrace(retrieved=[{"id": "N", "score": 1}])))[0])


# -- acl_compliance (RST-05) --------------------------------------------------

def _acl_corpus():
    return CorpusSnapshot(documents={
        "RETAIL": make_doc(content="r", acl=["retail", "private"]),
        "PRIV": make_doc(content="p", acl=["private"]),
    })


def test_acl_compliance_authorized_passes():
    case = RAGCase(case_id="c", question="q", tenant="retail")
    s = _sample(case=case, corpus=_acl_corpus())
    scores = AclCompliance().score(s, "", _ctx(RAGTrace(retrieved=[{"id": "RETAIL", "score": 1}])))
    assert _by(scores, "acl_compliance").value == 1.0


def test_acl_compliance_cross_tenant_hit_flagged():
    case = RAGCase(case_id="c", question="q", tenant="retail")
    s = _sample(case=case, corpus=_acl_corpus())
    scores = AclCompliance().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "RETAIL", "score": 1}, {"id": "PRIV", "score": 0.9}])))
    assert _by(scores, "unauthorized_hit_rate").value == 0.5
    assert ("PRIV", "acl") in _by(scores, "acl_compliance").metadata["violations"]


def test_acl_compliance_prohibited_and_unknown_doc():
    case = RAGCase(case_id="c", question="q", tenant="retail", prohibited_sources=["POISON"])
    s = _sample(case=case, corpus=_acl_corpus())
    scores = AclCompliance().score(s, "", _ctx(RAGTrace(retrieved=[
        {"id": "POISON", "score": 1}, {"id": "GHOST", "score": 0.5}])))
    assert _by(scores, "acl_compliance").value == 0.0  # prohibited + not-in-snapshot


def test_acl_compliance_unknown_without_identity_or_labels():
    assert _is_unknown(AclCompliance().score(_sample(case=RAGCase(case_id="c", question="q"),
                       corpus=_acl_corpus()), "", _ctx(RAGTrace(retrieved=[{"id": "RETAIL"}])))[0])
    corpus = CorpusSnapshot(documents={"NOACL": make_doc(content="n")})
    case = RAGCase(case_id="c", question="q", tenant="retail")
    assert _is_unknown(AclCompliance().score(_sample(case=case, corpus=corpus), "",
                       _ctx(RAGTrace(retrieved=[{"id": "NOACL", "score": 1}])))[0])


# -- citation_support (RST-06) ------------------------------------------------

def _cite_corpus():
    return CorpusSnapshot(documents={
        "D": make_doc(content="d", version="2", spans=[{"start": 0, "end": 100, "page": 1}]),
    })


def _cite_case(**over):
    kw = dict(case_id="c", question="q",
              gold_citations=[{"doc_id": "D", "start": 10, "end": 20, "version": "2"}])
    kw.update(over)
    return RAGCase(**kw)


def test_citation_support_valid_supporting_citation_passes():
    s = _sample(case=_cite_case(), corpus=_cite_corpus())
    scores = CitationSupport().score(s, "", _ctx(RAGTrace(cited_spans=[
        {"doc_id": "D", "start": 12, "end": 18, "version": "2"}])))
    assert _by(scores, "citation_support").value == 1.0
    assert _by(scores, "citation_precision").value == 1.0
    assert _by(scores, "citation_recall").value == 1.0


def test_citation_support_nonexistent_span_fails():
    s = _sample(case=_cite_case(), corpus=_cite_corpus())
    scores = CitationSupport().score(s, "", _ctx(RAGTrace(cited_spans=[
        {"doc_id": "D", "start": 200, "end": 250, "version": "2"}])))
    assert _by(scores, "citation_support").value == 0.0
    assert _by(scores, "citation_support").metadata["n_nonexistent"] == 1


def test_citation_support_wrong_version_fails():
    s = _sample(case=_cite_case(), corpus=_cite_corpus())
    scores = CitationSupport().score(s, "", _ctx(RAGTrace(cited_spans=[
        {"doc_id": "D", "start": 12, "end": 18, "version": "1"}])))
    assert _by(scores, "citation_support").value == 0.0
    assert _by(scores, "wrong_version_rate").value == 1.0


def test_citation_support_unknown_and_empty():
    # no gold citations -> not_tested
    assert _is_unknown(CitationSupport().score(_sample(case=RAGCase(case_id="c", question="q"),
                       corpus=_cite_corpus()), "", _ctx(RAGTrace(cited_spans=[])))[0])
    # no citation event in trace -> not_tested
    assert _is_unknown(CitationSupport().score(_sample(case=_cite_case(), corpus=_cite_corpus()),
                       "", _ctx(RAGTrace()))[0])
    # observed-empty citations -> real 0, not unknown
    sc = _by(CitationSupport().score(_sample(case=_cite_case(), corpus=_cite_corpus()), "",
             _ctx(RAGTrace(cited_spans=[]))), "citation_support")
    assert sc.value == 0.0 and not _is_unknown(sc)


# -- abstention (RST-04) ------------------------------------------------------

def test_abstention_correct_on_unanswerable():
    case = RAGCase(case_id="c", question="q", answerability="unanswerable")
    sc = Abstention().score(_sample(case=case), "", _ctx(RAGTrace(abstained=True)))[0]
    assert sc.value == 1.0


def test_abstention_answered_unanswerable_fails():
    case = RAGCase(case_id="c", question="q", answerability="unanswerable")
    sc = Abstention().score(_sample(case=case), "The number is 42.",
                            _ctx(RAGTrace(abstained=False)))[0]
    assert sc.value == 0.0


def test_abstention_false_refusal_on_answerable_fails():
    case = RAGCase(case_id="c", question="q", answerability="answerable")
    sc = Abstention().score(_sample(case=case), "I cannot answer that.", {})[0]
    assert sc.value == 0.0 and sc.metadata["detection"] == "keyword"
    assert "false refusal" in sc.reason


def test_abstention_answered_answerable_passes():
    case = RAGCase(case_id="c", question="q", answerability="answerable")
    sc = Abstention().score(_sample(case=case), "The LTV cap is 80 percent.", {})[0]
    assert sc.value == 1.0


def test_abstention_unknown_without_oracle_or_signal():
    # no answerability/expected_action -> not_tested
    assert _is_unknown(Abstention().score(_sample(case=RAGCase(case_id="c", question="q",
                       answerability="")), "x", {})[0])
    # answerable but empty output and no trace signal -> not_tested
    assert _is_unknown(Abstention().score(_sample(case=RAGCase(case_id="c", question="q")),
                       "", {})[0])


# -- registration + end-to-end runner wiring ----------------------------------

def test_metrics_registered_by_name():
    for name in ("evidence_set_recall", "freshness", "acl_compliance",
                 "citation_support", "abstention"):
        assert name in ak.METRICS.names()


def test_public_exports():
    for name in ("CorpusSnapshot", "RAGCase", "RAGTrace", "EvidenceSetRecall",
                 "Freshness", "AclCompliance", "CitationSupport", "Abstention",
                 "make_doc", "synthetic_bank_kb"):
        assert hasattr(ak, name) and name in ak.__all__


def test_end_to_end_via_precomputed_runner():
    """The Runner copies Sample.actual_trace into context['trace']; prove the
    metric reads the evidence set from there and scores through ak.evaluate."""
    snapshot, cases = synthetic_bank_kb()
    case = cases[0]
    trace = RAGTrace(query=case.question, retrieved=[
        {"id": "LP-CURRENT", "score": 0.9}, {"id": "FIN-TABLE", "score": 0.8}])
    sample = Sample(input=case.question, actual_output="LTV cap is 80 percent; Q1 NII 1200.",
                    actual_trace=trace.to_dict(),
                    metadata={"rag_case": case.to_dict(), "corpus": snapshot.to_dict()})
    result = ak.evaluate(dataset=[sample], model="precomputed",
                         scorers=["evidence_set_recall", "freshness", "acl_compliance"])
    stats = result.stats
    assert stats["evidence_set_recall"].mean == 1.0   # both docs of set 1 retrieved
    assert stats["stale_hit_rate"].mean == 0.0        # both current as of 2026-07-01
    assert stats["acl_compliance"].mean == 1.0        # retail-authorized docs


def test_acl_and_citation_are_not_tested_without_a_corpus_snapshot():
    """Absent snapshot is absent evidence, not a violation."""
    import json
    from auditkit.metrics.rag_stress import (AclCompliance, CitationSupport, RAGCase, RAGTrace)
    from auditkit.sample import Sample
    case = RAGCase(case_id="c", question="q", tenant="t",
                   gold_citations=[{"doc_id": "D1", "span": "s1"}])
    trace = RAGTrace(retrieved=[{"id": "D1"}], cited_spans=[{"doc_id": "D1", "span": "s1"}])
    sample = Sample(input="q", metadata={"rag_case": case})  # no "corpus"
    ctx = {"trace": trace.to_dict()}
    for M in (AclCompliance, CitationSupport):
        sc = M().score(sample, "", ctx)[0]
        assert sc.metadata.get("status") == "not_tested", M.__name__
        assert sc.metadata.get("unknown") is True
    # strict-JSON to_dict even with a non-finite score
    t = RAGTrace(retrieved=[{"id": "d", "score": float("inf")}])
    json.dumps(t.to_dict(), allow_nan=False)


# -- C1 / C2 (fix plan #15) -------------------------------------------------------------------------------

def test_c1_a_document_not_yet_in_force_is_not_current():
    from auditkit.metrics.rag_stress import CorpusSnapshot, Freshness, RAGCase, RAGTrace, make_doc
    corpus = CorpusSnapshot(documents={
        "NOW": make_doc(content="now", version="2", effective_date="2026-01-01"),
        "FUTURE": make_doc(content="future", version="3", effective_date="2027-01-01"),
        "SAME_DAY": make_doc(content="same", version="2", effective_date="2026-07-01"),
    })
    case = RAGCase(case_id="c", question="q", decision_date="2026-07-01")
    s = Sample(input="q", metadata={"rag_case": case.to_dict(), "corpus": corpus.to_dict()})
    trace = RAGTrace(retrieved=[{"id": "NOW"}, {"id": "FUTURE"}, {"id": "SAME_DAY"}])
    fresh, rate = Freshness().score(s, "", {"trace": trace.to_dict()})
    assert fresh.value == pytest.approx(2 / 3) and rate.value == pytest.approx(1 / 3)
    assert fresh.metadata["not_yet_effective"] == ["FUTURE"] and "not yet in force" in fresh.reason


@pytest.mark.parametrize("output,abstained", [
    ("I cannot answer that from the documents.", True),
    ("Insufficient information in the sources.", True),
    ("There is no information about that in the context.", True),
    ("Customers not able to pay by the due date are charged a 2% late fee.", False),   # a real answer
    ("The rate is 5%. I don't have more detail on fees.", False),                      # refusal phrase later on
    ("The rate is 5%.", False),
])
def test_c2_keyword_fallback_reads_only_the_first_sentence(output, abstained):
    from auditkit.metrics.rag_stress import Abstention, RAGCase
    case = RAGCase(case_id="c", question="q", answerability="answerable")
    [sc] = Abstention().score(Sample(input="q", metadata={"rag_case": case.to_dict()}), output, {})
    assert sc.metadata["abstained"] is abstained and sc.metadata["detection"] == "keyword"


_LONG = ("Based on the documents, there is insufficient information about the 2019 schedule, but "
         "the current policy sets a 5% rate for secured loans and a 7% rate for unsecured loans, "
         "reviewed each quarter by the credit committee.")


@pytest.mark.parametrize("output,abstained", [
    ("Based on the documents, there is insufficient information to answer.", True),   # short reply
    ("The context has no relevant passage for this question.", True),
    (_LONG, False),                                      # a long answer merely mentioning the phrase
    ("The rate is 5%. There is no information on fees.", False),   # later sentence: a partial answer
])
def test_c2_short_replies_with_a_generic_refusal_phrase_abstain(output, abstained):
    from auditkit.metrics.rag_stress import Abstention, RAGCase
    case = RAGCase(case_id="c", question="q", answerability="answerable")
    [sc] = Abstention().score(Sample(input="q", metadata={"rag_case": case.to_dict()}), output, {})
    assert sc.metadata["abstained"] is abstained and sc.metadata["detection"] == "keyword"


def test_c2_refusal_lists_and_word_limit_are_arguments():
    from auditkit.metrics.rag_stress import Abstention, RAGCase
    case = RAGCase(case_id="c", question="q", answerability="answerable")
    sample = Sample(input="q", metadata={"rag_case": case.to_dict()})
    short = "Based on the documents, there is insufficient information to answer."
    assert Abstention(short_reply_max_words=5).score(sample, short, {})[0].metadata["abstained"] is False
    assert Abstention(short_reply_markers=()).score(sample, short, {})[0].metadata["abstained"] is False
    assert Abstention(refusal_markers=["rate is"]).score(
        sample, "The rate is 5%.", {})[0].metadata["abstained"] is True
    assert Abstention().identity() != Abstention(short_reply_max_words=5).identity()
