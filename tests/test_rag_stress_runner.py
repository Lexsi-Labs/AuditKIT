"""Deterministic tests for the RAG-stress fixture generator and paired runner.

No network, no LLM. They prove: the generator is reproducible under a seed (and
seed-sensitive), each stress kind is flagged only by its ``expected_failure``
metric (independence), the paired runner reports the stress arm failing while
the benign control passes, benign utility loss is reported separately,
unknown/not_tested cases are counted rather than scored 0, the report is strict
JSON with a per-kind/per-metric matrix, and a >= 20-case run locates retrieval
misses, stale evidence, context drops, wrong numbers, unsupported claims,
citation errors and access failures independently.
"""

from __future__ import annotations

import json

import pytest

from auditkit.metrics.rag_stress import CorpusSnapshot, RAGCase, RAGTrace
from auditkit.metrics.rag_stress_runner import (
    DEFAULT_METRICS, ContextRetention, NumericAccuracy, StressCase,
    generate_stress_cases, run_stress,
)
from auditkit.sample import Sample


def _cases_fingerprint(cases):
    return [(c.case.case_id, c.stress_kind, c.expected_failure,
             c.stress_trace.to_dict(), c.control_trace.to_dict()) for c in cases]


# -- generator ---------------------------------------------------------------

def test_generator_reproducible_under_seed():
    a = generate_stress_cases(seed=0)
    b = generate_stress_cases(seed=0)
    assert _cases_fingerprint(a) == _cases_fingerprint(b)


def test_generator_is_seed_sensitive():
    # Proves the rng is actually wired (benign variation), not unused.
    a = _cases_fingerprint(generate_stress_cases(seed=0))
    c = _cases_fingerprint(generate_stress_cases(seed=1))
    assert a != c


def test_generator_kinds_filter():
    cases = generate_stress_cases(seed=0, kinds=["poisoned_doc"], repeats=1)
    assert cases and {c.stress_kind for c in cases} == {"poisoned_doc"}


def test_unknown_kind_raises_not_silent_empty():
    with pytest.raises(ValueError):
        generate_stress_cases(seed=0, kinds=["typo"])


def test_stresscase_unpacks_as_documented_triple():
    sc = generate_stress_cases(seed=0, kinds=["stale_policy"], repeats=1)[0]
    case, trace, snapshot = sc  # list[(RAGCase, RAGTrace, CorpusSnapshot)]
    assert isinstance(case, RAGCase) and isinstance(trace, RAGTrace)
    assert isinstance(snapshot, CorpusSnapshot)


# -- independence: one kind, one metric --------------------------------------

def test_each_kind_flagged_only_by_expected_metric():
    report = run_stress(generate_stress_cases(seed=0))
    for r in report.cases:
        flagged = set(r.stress_flagged)
        want = set() if r.expected_failure is None else {r.expected_failure}
        assert flagged == want, (r.stress_kind, r.expected_failure, sorted(flagged))


def test_first_failure_stage_located_per_kind():
    expected_stage = {
        "stale_policy": "retrieval", "conflicting_policies": "retrieval",
        "missing_evidence": "answer", "false_refusal": "answer",
        "financial_table_citation": "citation", "financial_numeric": "answer",
        "long_context": "context", "multilingual": "retrieval",
        "poisoned_doc": "retrieval", "cross_tenant": "retrieval",
        "partial_retrieval": None,
    }
    report = run_stress(generate_stress_cases(seed=0))
    seen = {r.stress_kind: r.first_failure_stage for r in report.cases}
    assert seen == expected_stage
    # The distinct stages the PRD "first proof" must locate independently.
    assert {s for s in seen.values() if s} == {"retrieval", "context", "answer", "citation"}


# -- paired arms: stress fails, benign control passes ------------------------

def test_stress_arm_fails_control_passes():
    report = run_stress(generate_stress_cases(seed=0))
    for r in report.cases:
        if r.expected_failure is not None:
            assert r.stress[r.expected_failure]["flagged"] is True
        # every generated control arm is benign and must not trip a metric
        assert r.benign_control_passed is True
    assert report.benign_utility_loss() == []


def test_benign_utility_loss_reported_separately():
    # A control arm that itself retrieves a stale doc is utility loss, not an attack.
    snap = generate_stress_cases(seed=0, kinds=["stale_policy"], repeats=1)[0].snapshot
    case = RAGCase(case_id="ul", question="q",
                   sufficient_evidence_sets=[["LP-CURRENT"]], tenant="retail",
                   identity="retail", decision_date="2026-07-01")
    stale = RAGTrace(retrieved=[{"id": "LP-OLD", "score": 0.9}], abstained=False)
    sc = StressCase(case=case, snapshot=snap, stress_trace=stale, control_trace=stale,
                    stress_kind="stale_policy", expected_failure="freshness")
    report = run_stress([sc])
    assert report.benign_utility_loss() == ["ul"]
    assert report.cases[0].benign_control_passed is False


# -- unknown / not_tested never a silent 0 -----------------------------------

def test_partial_retrieval_is_all_not_tested_not_zero():
    report = run_stress(generate_stress_cases(seed=0, kinds=["partial_retrieval"], repeats=1))
    r = report.cases[0]
    assert r.stress_flagged == [] and r.first_failure_stage is None
    for m in DEFAULT_METRICS:
        res = r.stress[m]
        assert res["status"] in ("unknown", "not_tested")
        assert res["flagged"] is False  # not scored 0


def test_coverage_counts_unknown_and_not_tested_separately():
    report = run_stress(generate_stress_cases(seed=0))
    cov = report.to_dict()["coverage"]
    # partial_retrieval (x repeats) makes every metric not_tested at least twice.
    for m in DEFAULT_METRICS:
        assert cov[m]["not_tested"] >= 2
        # scored count excludes unknown/not_tested
        assert cov[m]["scored"] == cov[m]["flagged"] + cov[m]["pass"]


# -- strict JSON + matrix ----------------------------------------------------

def test_report_is_strict_json_with_matrix():
    report = run_stress(generate_stress_cases(seed=0))
    d = report.to_dict()
    round_tripped = json.loads(json.dumps(d, allow_nan=False))
    assert round_tripped == d
    assert d["n_cases"] >= 20
    matrix = d["matrix"]
    # nested dicts keyed by kind then metric (never tuple keys)
    for kind, per_metric in matrix.items():
        assert set(per_metric) == set(DEFAULT_METRICS)
        for cell in per_metric.values():
            assert set(cell) == {"flagged", "pass", "unknown", "not_tested"}
    # the >=20-case matrix shows each expected failure located in its own cell
    assert matrix["poisoned_doc"]["acl_compliance"]["flagged"] >= 1
    assert matrix["financial_numeric"]["numeric_accuracy"]["flagged"] >= 1
    assert matrix["long_context"]["context_retention"]["flagged"] >= 1


def test_default_run_is_at_least_twenty_paired_cases():
    cases = generate_stress_cases()
    assert len(cases) >= 20


# -- the two extra locators: not_tested paths --------------------------------

def _score(metric, case, trace):
    sample = Sample(input="q", actual_output="", actual_trace=trace.to_dict(),
                    metadata={"rag_case": case})
    return metric.score(sample, "", {"trace": trace.to_dict()})[0]


def test_numeric_accuracy_not_tested_without_gold_answer():
    case = RAGCase(case_id="n", question="q")
    s = _score(NumericAccuracy(), case, RAGTrace(answer_claims=["100"]))
    assert s.metadata.get("unknown") and s.label == "not_tested"


def test_numeric_accuracy_flags_wrong_number():
    case = RAGCase(case_id="n", question="q", gold_answer="the total is 1200 USD")
    s = _score(NumericAccuracy(), case, RAGTrace(answer_claims=["the total is 1500 USD"]))
    assert s.value == 0.0 and not s.metadata.get("unknown")


def test_context_retention_not_tested_without_final_context():
    case = RAGCase(case_id="c", question="q", sufficient_evidence_sets=[["A"]])
    s = _score(ContextRetention(), case, RAGTrace(retrieved=[{"id": "A"}]))
    assert s.metadata.get("unknown") and s.label == "not_tested"


def test_context_retention_flags_dropped_evidence():
    case = RAGCase(case_id="c", question="q", sufficient_evidence_sets=[["A", "B"]])
    s = _score(ContextRetention(), case, RAGTrace(final_context=[{"doc_id": "A", "start": 0, "end": 1}]))
    assert s.value == 0.0 and not s.metadata.get("unknown")


# -- C3 (fix plan #15) ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("5-10", {5, 10}), ("Q3-2026", {3, 2026}), ("-3", {-3}), ("(−3)", {-3}), ("1,2,3", {1, 2, 3}),
    ("1,000", {1000}), ("10.50", {10.5}), ("rate is 5.", {5}), ("x -2.5 y", {-2.5}),
    # a hyphen after a letter is not a sign; a first group over 3 digits is not grouping
    ("COVID-19", {19}), ("FY-2026", {2026}), ("A-5", {5}), ("12345,678", {12345, 678}),
    ("1,234.5", {1234.5}),
])
def test_c3_number_parsing(text, want):
    from auditkit.metrics.rag_stress_runner import _numbers
    assert _numbers(text) == want


def test_c3_a_range_matches_its_spelled_out_form():
    from auditkit.metrics.rag_stress import RAGCase, RAGTrace
    from auditkit.metrics.rag_stress_runner import NumericAccuracy
    case = RAGCase(case_id="c", question="q", gold_answer="Tenor is 5-10 years")
    s = Sample(input="q", metadata={"rag_case": case.to_dict()})
    [sc] = NumericAccuracy().score(s, "", {"trace": RAGTrace(answer_claims=["between 5 and 10 years"]).to_dict()})
    assert sc.value == 1.0
