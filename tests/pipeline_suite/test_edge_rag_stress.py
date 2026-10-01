"""Edge cases for the RAG-stress metrics (PR #1): evidence_set_recall, freshness,
acl_compliance, citation_support, abstention, numeric_accuracy, context_retention.

Each metric is called directly on a hand-built corpus, case and trace, so every
expected value can be worked out from the fixture below. A result that is not a
measurement comes back as ``("not_tested" | "unknown", None)``.
"""

from __future__ import annotations

import pytest

rs = pytest.importorskip("auditkit.metrics.rag_stress")
rsr = pytest.importorskip("auditkit.metrics.rag_stress_runner")

from auditkit.sample import Sample  # noqa: E402

CorpusSnapshot, RAGCase, RAGTrace, make_doc = rs.CorpusSnapshot, rs.RAGCase, rs.RAGTrace, rs.make_doc

# A: current, retail-only, spans 0-100, v2.     B: superseded on 2026-01-01, corporate-only, v1.
# C: current, retail and corporate.              N: no version, dates or ACL (untestable for both).
CORPUS = CorpusSnapshot(documents={
    "A": make_doc(content="policy a", version="v2", effective_date="2025-06-01", acl=["retail"],
                  spans=[{"start": 0, "end": 100, "page": 1}]),
    "B": make_doc(content="policy b", version="v1", effective_date="2024-01-01", superseded_date="2026-01-01",
                  acl=["corporate"], spans=[{"start": 0, "end": 50, "page": 1}]),
    "C": make_doc(content="table c", version="v1", effective_date="2025-01-01", acl=["retail", "corporate"],
                  spans=[{"start": 0, "end": 40, "page": 2}, {"start": 60, "end": 90, "page": 2}]),
    "N": make_doc(content="note n"),
})


def case(**kw):
    kw.setdefault("case_id", "c")
    kw.setdefault("question", "q")
    return RAGCase(**kw)


def run(metric, rag_case=None, trace=None, corpus=CORPUS, output=""):
    """{score name: (label or None, value or None)}; the value is None when flagged unknown."""
    meta = {}
    if rag_case is not None:
        meta["rag_case"] = rag_case.to_dict()
    if corpus is not None:
        meta["corpus"] = corpus.to_dict()
    s = Sample(input="q", metadata=meta)
    ctx = {"trace": trace.to_dict()} if trace is not None else None
    out = {}
    for sc in metric.score(s, output, ctx):
        flagged = (sc.metadata or {}).get("unknown")
        out[sc.name] = (sc.label, None) if flagged else (None, pytest.approx(sc.value))
    return out


def ret(*ids, **extra):
    return RAGTrace(retrieved=[{"id": i, **extra.get(i, {})} for i in ids])


def status(result, name):
    return result[name][0]


def value(result, name):
    return result[name][1]


# -- evidence_set_recall ---------------------------------------------------------------------------

ESR = rs.EvidenceSetRecall()
SETS = [["A", "C"], ["B"]]


@pytest.mark.parametrize("retrieved,want", [
    (("A", "C"), 1.0),                 # the first complete set
    (("C", "A"), 1.0),                 # order does not matter
    (("B",), 1.0),                     # an alternate complete set is enough
    (("A", "C", "N", "B"), 1.0),       # extra documents do not hurt
    (("A",), 0.0),                     # half of a set is not a set
    (("A", "N"), 0.0),
    ((), 0.0),                         # retrieval observed but empty is a real failure
])
def test_evidence_set_recall(retrieved, want):
    assert value(run(ESR, case(sufficient_evidence_sets=SETS), ret(*retrieved)), "evidence_set_recall") == want


def test_evidence_set_recall_reports_the_best_partial_set():
    # best set is [A, C] with one of two retrieved -> ranked recall 0.5
    [sc] = ESR.score(Sample(input="q", metadata={"rag_case": case(sufficient_evidence_sets=SETS).to_dict()}),
                     "", {"trace": ret("A").to_dict()})
    assert sc.metadata["best_set_recall"] == pytest.approx(0.5)


def test_changed_digest_drops_the_sets_that_use_that_document():
    # A's pinned digest no longer matches: set [A, C] is dropped, only [B] can pass
    c = case(sufficient_evidence_sets=SETS, evidence_digests={"A": "stale-digest"})
    assert value(run(ESR, c, ret("A", "C")), "evidence_set_recall") == 0.0
    assert value(run(ESR, c, ret("B")), "evidence_set_recall") == 1.0


def test_pinned_document_missing_from_the_snapshot_invalidates_its_set():
    c = case(sufficient_evidence_sets=[["GONE"], ["B"]], evidence_digests={"GONE": "x"})
    assert value(run(ESR, c, ret("GONE")), "evidence_set_recall") == 0.0


def test_all_sets_invalidated_is_unknown():
    c = case(sufficient_evidence_sets=[["A"]], evidence_digests={"A": "stale"})
    assert status(run(ESR, c, ret("A")), "evidence_set_recall") == "unknown"


def test_matching_pinned_digest_keeps_the_set():
    c = case(sufficient_evidence_sets=[["A"]], evidence_digests={"A": CORPUS.documents["A"]["digest"]})
    assert value(run(ESR, c, ret("A")), "evidence_set_recall") == 1.0


def test_without_a_corpus_nothing_is_invalidated():
    c = case(sufficient_evidence_sets=[["A"]], evidence_digests={"A": "stale"})
    assert value(run(ESR, c, ret("A"), corpus=None), "evidence_set_recall") == 1.0


@pytest.mark.parametrize("rag_case,trace", [
    (None, ret("A")),                                              # no case
    (case(), ret("A")),                                            # no sets
    (case(sufficient_evidence_sets=SETS), None),                   # no trace
    (case(sufficient_evidence_sets=SETS), RAGTrace()),             # retrieval never captured
    (case(sufficient_evidence_sets=SETS),
     RAGTrace(retrieved=[{"id": "A"}], coverage={"retrieved": "missing"})),  # coverage label wins
])
def test_evidence_set_recall_not_tested(rag_case, trace):
    assert status(run(ESR, rag_case, trace), "evidence_set_recall") == "not_tested"


# -- freshness ---------------------------------------------------------------------------------------

FR = rs.Freshness()


@pytest.mark.parametrize("decision,retrieved,fresh", [
    ("2026-07-01", ("A",), 1.0),
    ("2026-07-01", ("B",), 0.0),           # superseded before the decision date
    ("2026-01-01", ("B",), 0.0),           # superseded ON the decision date counts as stale
    ("2025-12-31", ("B",), 1.0),           # the day before it was still current
    ("2026-07-01", ("A", "B"), 0.5),
    ("2026-07-01", ("A", "B", "N"), 0.5),  # N has no dates: not testable, not in the denominator
    ("2026-07-01", ("A", "GHOST"), 1.0),   # not in the snapshot: a coverage question, skipped here
])
def test_freshness(decision, retrieved, fresh):
    r = run(FR, case(decision_date=decision), ret(*retrieved))
    assert value(r, "freshness") == fresh and value(r, "stale_hit_rate") == pytest.approx(1 - fresh)


def test_a_digest_mismatch_in_the_trace_is_stale_even_when_dates_are_fine():
    r = run(FR, case(decision_date="2026-07-01"), ret("A", A={"digest": "old-digest"}))
    assert value(r, "freshness") == 0.0


def test_a_matching_trace_digest_is_fine():
    r = run(FR, case(decision_date="2026-07-01"), ret("A", A={"digest": CORPUS.documents["A"]["digest"]}))
    assert value(r, "freshness") == 1.0


@pytest.mark.parametrize("retrieved", [("N",), ("GHOST",), ()])
def test_freshness_with_nothing_testable_is_unknown(retrieved):
    assert status(run(FR, case(decision_date="2026-07-01"), ret(*retrieved)), "freshness") == "unknown"


@pytest.mark.parametrize("rag_case,trace,corpus", [
    (case(), ret("A"), CORPUS),                                 # no decision date
    (case(decision_date="2026-07-01"), ret("A"), None),         # no corpus
    (case(decision_date="2026-07-01"), RAGTrace(), CORPUS),     # no retrieval
])
def test_freshness_not_tested(rag_case, trace, corpus):
    assert status(run(FR, rag_case, trace, corpus=corpus), "freshness") == "not_tested"


def test_a_document_not_yet_in_force_is_not_current():
    corpus = CorpusSnapshot(documents={"F": make_doc(content="future", version="v3", effective_date="2027-01-01")})
    assert value(run(FR, case(decision_date="2026-07-01"), ret("F"), corpus=corpus), "freshness") == 0.0


# -- acl_compliance ------------------------------------------------------------------------------------

ACL = rs.AclCompliance()


@pytest.mark.parametrize("kw,retrieved,compliance", [
    ({"identity": "retail"}, ("A",), 1.0),
    ({"identity": "retail"}, ("B",), 0.0),                 # corporate-only document
    ({"identity": "retail"}, ("A", "B"), 0.5),
    ({"tenant": "corporate"}, ("B", "C"), 1.0),            # tenant alone is a principal
    ({"tenant": "corporate", "identity": "retail"}, ("A", "B"), 1.0),   # either principal is enough
    ({"identity": "retail"}, ("A", "N"), 1.0),             # N has no ACL: untestable, left out
    ({"identity": "retail"}, ("A", "GHOST"), 0.5),         # not in the snapshot: fail closed
    ({"identity": "retail", "prohibited_sources": ["A"]}, ("A",), 0.0),   # prohibited beats the ACL
    ({"identity": "retail", "prohibited_sources": ["GHOST"]}, ("GHOST",), 0.0),
])
def test_acl_compliance(kw, retrieved, compliance):
    r = run(ACL, case(**kw), ret(*retrieved))
    assert value(r, "acl_compliance") == compliance
    assert value(r, "unauthorized_hit_rate") == pytest.approx(1 - compliance)


def test_an_empty_acl_admits_nobody():
    corpus = CorpusSnapshot(documents={"E": make_doc(content="e", acl=[])})
    assert value(run(ACL, case(identity="retail"), ret("E"), corpus=corpus), "acl_compliance") == 0.0


def test_acl_violations_name_the_document_and_the_rule():
    [sc, _] = ACL.score(Sample(input="q", metadata={"rag_case": case(identity="retail", prohibited_sources=["C"])
                                                    .to_dict(), "corpus": CORPUS.to_dict()}),
                        "", {"trace": ret("B", "C", "GHOST").to_dict()})
    assert [tuple(v) for v in sc.metadata["violations"]] == [("B", "acl"), ("C", "prohibited"),
                                                             ("GHOST", "not in snapshot")]


def test_acl_with_only_unlabelled_documents_is_unknown():
    assert status(run(ACL, case(identity="retail"), ret("N")), "acl_compliance") == "unknown"


@pytest.mark.parametrize("rag_case,trace,corpus", [
    (case(), ret("A"), CORPUS),                          # no tenant or identity
    (case(identity="retail"), RAGTrace(), CORPUS),       # no retrieval
    (case(identity="retail"), ret("A"), None),           # no corpus
])
def test_acl_not_tested(rag_case, trace, corpus):
    assert status(run(ACL, rag_case, trace, corpus=corpus), "acl_compliance") == "not_tested"


# -- citation_support ------------------------------------------------------------------------------------

CS = rs.CitationSupport()
GOLD = [{"doc_id": "A", "start": 10, "end": 30, "version": "v2"}]


def cite(*spans):
    return RAGTrace(cited_spans=list(spans))


def sp(doc, start, end, **kw):
    return {"doc_id": doc, "start": start, "end": end, **kw}


@pytest.mark.parametrize("cited,support,precision,recall,wrong_version", [
    ([sp("A", 12, 20)], 1.0, 1.0, 1.0, 0.0),                     # real, current, on the gold span
    ([sp("A", 12, 20, version="v2")], 1.0, 1.0, 1.0, 0.0),        # explicit matching version
    ([sp("A", 12, 20, version="v1")], 0.0, 0.0, 0.0, 1.0),        # wrong version
    ([sp("A", 60, 80)], 1.0, 0.0, 0.0, 0.0),                     # real span, but not the supporting one
    ([sp("A", 90, 120)], 0.0, 0.0, 0.0, 0.0),                    # runs past the declared span: not real
    ([sp("C", 45, 55)], 0.0, 0.0, 0.0, 0.0),                     # falls in the gap between C's spans
    ([sp("GHOST", 0, 5)], 0.0, 0.0, 0.0, 0.0),                   # document not in the snapshot
    ([sp("N", 0, 5)], 0.0, 0.0, 0.0, 0.0),                       # N declares no spans
    ([{"doc_id": "A", "start": 12}], 0.0, 0.0, 0.0, 0.0),        # malformed: no end offset
    ([sp("A", 30, 40)], 1.0, 0.0, 0.0, 0.0),                     # touches the gold end but no overlap
    ([sp("A", 12, 20), sp("A", 60, 80)], 1.0, 0.5, 1.0, 0.0),
    ([], 0.0, 0.0, 0.0, 0.0),                                    # observed, cited nothing: a real failure
])
def test_citation_support(cited, support, precision, recall, wrong_version):
    r = run(CS, case(gold_citations=GOLD), cite(*cited))
    assert (value(r, "citation_support"), value(r, "citation_precision"), value(r, "citation_recall"),
            value(r, "wrong_version_rate")) == (support, precision, recall, wrong_version)


def test_citation_recall_over_two_gold_spans():
    gold = GOLD + [sp("C", 60, 70)]
    assert value(run(CS, case(gold_citations=gold), cite(sp("C", 62, 65))), "citation_recall") == 0.5


@pytest.mark.parametrize("rag_case,trace,corpus", [
    (case(), cite(sp("A", 12, 20)), CORPUS),               # no gold citations
    (case(gold_citations=GOLD), RAGTrace(), CORPUS),       # citations never captured
    (case(gold_citations=GOLD), cite(sp("A", 12, 20)), None),
])
def test_citation_not_tested(rag_case, trace, corpus):
    assert status(run(CS, rag_case, trace, corpus=corpus), "citation_support") == "not_tested"


# -- abstention ------------------------------------------------------------------------------------------

AB = rs.Abstention()
UNANS, ANS = case(answerability="unanswerable"), case(answerability="answerable")


@pytest.mark.parametrize("rag_case,trace,output,want", [
    (UNANS, RAGTrace(abstained=True), "", 1.0),
    (UNANS, RAGTrace(abstained=False), "", 0.0),
    (ANS, RAGTrace(abstained=True), "", 0.0),                            # false refusal
    (ANS, RAGTrace(abstained=False), "I cannot answer", 1.0),            # the explicit flag wins over text
    (UNANS, RAGTrace(answer_claims=[]), "", 1.0),                        # no claims = abstained
    (UNANS, RAGTrace(answer_claims=["the rate is 5%"]), "", 0.0),
    (UNANS, None, "I cannot answer that from the documents.", 1.0),      # keyword fallback
    (ANS, None, "The rate is 5%.", 1.0),
    (ANS, None, "Insufficient information in the sources.", 0.0),
    (case(answerability="unanswerable", expected_action="answer"), None, "The rate is 5%.", 1.0),  # override
])
def test_abstention(rag_case, trace, output, want):
    assert value(run(AB, rag_case, trace, output=output), "abstention") == want


@pytest.mark.parametrize("rag_case,output", [
    (None, "x"),                                   # no case
    (case(answerability="partial"), "x"),          # no usable expected action
    (ANS, ""),                                     # no answer signal at all
    (ANS, "   "),
])
def test_abstention_not_tested(rag_case, output):
    assert status(run(AB, rag_case, None, output=output), "abstention") == "not_tested"


def test_a_real_answer_that_mentions_a_refusal_phrase_is_not_an_abstention():
    out = "Customers not able to pay by the due date are charged a 2% late fee."
    assert value(run(AB, ANS, None, output=out), "abstention") == 1.0


# -- numeric_accuracy --------------------------------------------------------------------------------------

NA = rsr.NumericAccuracy()


@pytest.mark.parametrize("gold,claims,want", [
    ("The rate is 5.25%.", ["It is 5.25 percent."], 1.0),
    ("Limit: 1,000 USD", ["the limit is 1000 dollars"], 1.0),            # thousands separator
    ("Fee 10.50", ["fee of 10.5"], 1.0),                                  # trailing zero
    ("5 and 7", ["5"], 0.0),                                              # one number missing
    ("5 and 7", ["5", "and also 7"], 1.0),                                # numbers can span claims
    ("5", ["5, 6 and 99"], 1.0),                                          # extra numbers do not hurt
    ("5", ["five"], 0.0),                                                 # words are not numbers
    ("5", [], 0.0),                                                       # observed, no claims
])
def test_numeric_accuracy(gold, claims, want):
    assert value(run(NA, case(gold_answer=gold), RAGTrace(answer_claims=claims)), "numeric_accuracy") == want


def test_numeric_accuracy_without_numbers_in_the_gold_is_unknown():
    assert status(run(NA, case(gold_answer="yes"), RAGTrace(answer_claims=["yes"])), "numeric_accuracy") == "unknown"


@pytest.mark.parametrize("rag_case,trace", [
    (case(), RAGTrace(answer_claims=["5"])),       # no gold answer
    (case(gold_answer="5"), RAGTrace()),           # claims never captured
    (None, RAGTrace(answer_claims=["5"])),
])
def test_numeric_accuracy_not_tested(rag_case, trace):
    assert status(run(NA, rag_case, trace), "numeric_accuracy") == "not_tested"


def test_a_numeric_range_matches_its_spelled_out_form():
    r = run(NA, case(gold_answer="Tenor is 5-10 years"), RAGTrace(answer_claims=["between 5 and 10 years"]))
    assert value(r, "numeric_accuracy") == 1.0


# -- context_retention ------------------------------------------------------------------------------------

CR = rsr.ContextRetention()


def ctx(*ids):
    return RAGTrace(final_context=[{"doc_id": i, "start": 0, "end": 10} for i in ids])


@pytest.mark.parametrize("ids,want", [
    (("A", "C"), 1.0),
    (("B",), 1.0),              # alternate set
    (("A",), 0.0),              # the assembler dropped C
    ((), 0.0),                  # observed, empty context
    (("A", "C", "N"), 1.0),
])
def test_context_retention(ids, want):
    assert value(run(CR, case(sufficient_evidence_sets=SETS), ctx(*ids)), "context_retention") == want


def test_retrieval_can_pass_while_context_retention_fails():
    c = case(sufficient_evidence_sets=SETS)
    trace = RAGTrace(retrieved=[{"id": "A"}, {"id": "C"}], final_context=[{"doc_id": "A", "start": 0, "end": 5}])
    assert value(run(ESR, c, trace), "evidence_set_recall") == 1.0
    assert value(run(CR, c, trace), "context_retention") == 0.0


@pytest.mark.parametrize("rag_case,trace", [
    (case(), ctx("A")),
    (case(sufficient_evidence_sets=SETS), RAGTrace()),
])
def test_context_retention_not_tested(rag_case, trace):
    assert status(run(CR, rag_case, trace), "context_retention") == "not_tested"


# -- the contract round-trips -------------------------------------------------------------------------------

def test_case_trace_and_corpus_round_trip_through_dicts():
    c = case(sufficient_evidence_sets=SETS, gold_citations=GOLD, decision_date="2026-07-01")
    t = RAGTrace(retrieved=[{"id": "A", "score": 0.9}], cited_spans=[sp("A", 1, 2)])
    assert RAGCase.from_dict(c.to_dict()) == c
    assert RAGTrace.from_dict(t.to_dict()) == t
    assert CorpusSnapshot.from_dict(CORPUS.to_dict()) == CORPUS
    assert t.to_dict()["retrieved_contexts"] == ["A"]


def test_make_doc_never_stores_the_raw_text():
    d = make_doc(content="secret source text")
    assert "secret source text" not in str(d) and d["digest"] == rs.sha256_digest("secret source text")
    with pytest.raises(ValueError):
        make_doc()
