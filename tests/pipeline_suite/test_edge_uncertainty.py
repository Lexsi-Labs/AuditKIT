"""Edge cases for PR #4's uncertainty code: the UQLM importer, the four calibration
reports, claim-level reports and the code/SQL execution checks.

Expected numbers are worked out by hand in the comments. Skips before #4 merges.
"""

from __future__ import annotations

import pytest

uq = pytest.importorskip("auditkit.uncertainty")
UQResult, CodeCase = uq.UQResult, uq.CodeCase


def R(item, conf, correct):
    return UQResult(item_id=item, confidence=conf, correct=correct)


# -- importer ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    (True, True), (False, False), (1, True), (0, False), (1.0, True), (0.0, False),
    ("yes", True), ("Correct", True), (" TRUE ", True), ("wrong", False), ("incorrect", False), ("0", False),
    (None, None),
])
def test_import_reads_correctness_labels(raw, want):
    [r] = uq.import_uqlm([{"id": "x", "confidence": 0.5, "correct": raw}])
    assert r.correct is want and "correct_issue" not in r.provenance


@pytest.mark.parametrize("raw", ["maybe", 2, 0.5, "", []])
def test_import_never_turns_junk_into_a_wrong_label(raw):
    [r] = uq.import_uqlm([{"id": "x", "confidence": 0.5, "correct": raw}])
    assert r.correct is None and "correct_issue" in r.provenance


@pytest.mark.parametrize("raw,issue", [
    (None, "missing"), ("", "missing"), ("high", "non-numeric"), ("nan", "non-finite"),
    (float("inf"), "non-finite"), (-0.01, "outside"),
])
def test_import_labels_a_bad_confidence(raw, issue):
    [r] = uq.import_uqlm([{"id": "x", "confidence": raw}])
    assert r.confidence is None and issue in r.provenance["confidence_issue"]


def test_import_confidence_bounds_are_inclusive_and_strings_parse():
    rows = [{"id": "a", "confidence": 0}, {"id": "b", "confidence": "1"}]
    assert [r.confidence for r in uq.import_uqlm(rows)] == [0.0, 1.0]


def test_flipping_only_applies_to_a_present_confidence():
    a, b = uq.import_uqlm([{"id": "a", "score": 0.0}, {"id": "b"}], higher_is_confident=False)
    assert a.confidence == 1.0 and a.provenance["orientation"] == "flipped"
    assert b.confidence is None and "orientation" not in b.provenance


def test_an_explicit_score_field_that_is_absent_is_missing_not_another_column():
    [r] = uq.import_uqlm([{"id": "x", "confidence": 0.9}], score_field="semantic_negentropy")
    assert r.confidence is None and r.provenance["extra"] == {"confidence": 0.9}


def test_method_comes_from_the_scorer_column_else_the_score_column():
    a, b = uq.import_uqlm([{"qid": "a", "scorer": "noncontradiction", "confidence": 0.5},
                           {"qid": "b", "uq_score": 0.5}])
    assert (a.item_id, a.method) == ("a", "noncontradiction") and (b.item_id, b.method) == ("b", "uq_score")


def test_the_importer_tags_the_provider():
    [r] = uq.import_uqlm([{"id": "x", "confidence": 0.5}], provider="p", provider_version="1.2", method="m")
    assert (r.provider, r.provider_version, r.method) == ("p", "1.2", "m")


# -- reports -------------------------------------------------------------------------------------------------

def test_counts_separate_unlabelled_and_unscored_items():
    rep = uq.confidence_calibration([R("a", 0.9, True), R("b", None, True), R("c", 0.5, None)], bins=1)
    assert (rep["n_total"], rep["n_labeled"], rep["n_unlabeled"], rep["n_no_confidence"]) == (3, 2, 1, 1)
    assert rep["ece"] == pytest.approx(0.1)                           # only a is usable: |1 - .9|


@pytest.mark.parametrize("items,fp,fr", [
    ([R("a", 0.5, False)], 1.0, None),                                # confidence == threshold is confident
    ([R("a", 0.49, True)], None, 1.0),                                # everything deferred
    ([R("a", 0.9, True), R("b", 0.2, False)], 0.0, 0.0),
])
def test_false_pass_rate_edges(items, fp, fr):
    rep = uq.false_pass_rate(items, threshold=0.5)
    assert rep["false_pass_rate"] == fp and rep["false_reject_rate"] == fr


ERR4 = [R("a", 0.1, False), R("b", 0.4, True), R("c", 0.6, False), R("d", 0.9, True)]


@pytest.mark.parametrize("budget,k,rate", [
    (0.0, 0, 0.0), (0.2, 1, 0.5),       # ceil(.2 * 4) = 1 reviewed -> 1 of 2 errors
    (0.5, 2, 0.5), (0.75, 3, 1.0), (1.0, 4, 1.0),
    (-1, 0, 0.0), (5, 4, 1.0),          # clamped to [0, 1]
])
def test_review_capacity_edges(budget, k, rate):
    rep = uq.review_capacity(ERR4, review_budget=budget)
    assert rep["n_reviewed"] == k and rep["catch_rate"] == rate


def test_review_capacity_curve_never_decreases_and_is_none_without_errors():
    curve = [p["catch_rate"] for p in uq.review_capacity(ERR4)["curve"]]
    assert curve == sorted(curve)
    assert uq.review_capacity([R("a", 0.5, True)])["catch_rate"] is None


def test_review_capacity_breaks_confidence_ties_by_id():
    rep = uq.review_capacity([R("b", 0.5, False), R("a", 0.5, True)], review_budget=0.5)
    assert rep["catch_rate"] == 0.0                                   # "a" is reviewed first, and it was right


@pytest.mark.parametrize("items,aurc", [
    ([R("a", 0.9, True), R("b", 0.1, True)], 0.0),
    ([R("a", 0.9, False)], 1.0),
    ([R("a", 0.9, True), R("b", 0.1, False)], 0.25),                  # risks 0, .5
    ([R("a", 0.9, False), R("b", 0.1, True)], 0.75),                  # risks 1, .5 : a miscalibrated order
])
def test_selective_risk_edges(items, aurc):
    assert uq.selective_risk(items)["aurc"] == pytest.approx(aurc)


def test_every_report_is_registered_by_name():
    assert set(uq.REPORTS) == {"confidence_calibration", "false_pass_rate", "review_capacity", "selective_risk"}


# -- claims --------------------------------------------------------------------------------------------------

def test_decompose_accepts_dict_rows_and_validates_them():
    d = uq.decompose_answer("A.", [{"claim_id": 1, "text": "A", "confidence": "0.7", "supported": True}])
    assert d.claims[0].claim_id == "1" and d.claims[0].confidence == 0.7
    assert uq.DecomposedAnswer.from_dict(d.to_dict()) == d
    with pytest.raises(ValueError):
        uq.decompose_answer("A.", [{"claim_id": "c", "text": "A", "confidence": 2}])


def test_confident_false_claim_rate_threshold_is_inclusive_and_needs_labels():
    claims = [uq.Claim("c1", "x", 0.5, False), uq.Claim("c2", "y", 0.4, False)]
    rep = uq.confident_false_claim_rate(claims, threshold=0.5)
    assert rep["confident_false_claim_rate"] == 1.0 and rep["confident_false_claim_ids"] == ["c1"]
    assert uq.confident_false_claim_rate([uq.Claim("c", "x", 0.9, None)])["status"] == "unknown"


def test_claim_calibration_by_hand():
    # one bin: mean confidence .75, support rate .5 -> ECE .25
    claims = [uq.Claim("a", "x", 0.5, True), uq.Claim("b", "y", 1.0, False)]
    assert uq.claim_support_calibration(claims, bins=1)["ece"] == pytest.approx(0.25)


# -- code execution --------------------------------------------------------------------------------------------

def run(code, tests, timeout=2.0, **kw):
    return uq.run_execution_tests(CodeCase("x", code, tests=tests, **kw), timeout=timeout)


@pytest.mark.parametrize("code,reason", [
    ("def solve():\n    return open('/etc/passwd').read()", "open"),
    ("def solve():\n    return eval('1')", "eval"),
    ("from os import path\ndef solve():\n    return 1", "import"),
    ("def solve():\n    return (x for x in ()).gi_frame", "gi_frame"),     # frame walk out of the namespace
    ("def solve():\n    return ().__class__", "__class__"),
])
def test_the_guard_refuses_before_anything_runs(code, reason):
    r = run(code, [((), 1)])
    assert r["status"] == "blocked" and r["correct"] is None and reason in r["reason"] and r["tests"] == []


def test_a_hung_snippet_times_out_as_a_failed_test():
    r = run("def solve():\n    for _ in range(20000000):\n        pass\n    return 1", [((), 1)], timeout=0.05)
    assert r["correct"] is False and r["tests"][0]["note"] == "timeout"


@pytest.mark.parametrize("code,status", [
    ("raise ValueError('at import')", "exec_error"),
    ("def other():\n    return 1", "no_entry"),
])
def test_code_that_runs_but_cannot_be_called_is_a_measured_failure(code, status):
    r = run(code, [((), 1)])
    assert r["status"] == status and r["correct"] is False


def test_a_raising_test_and_a_callable_check_are_both_supported():
    code = "def solve(x):\n    if x < 0:\n        raise ValueError('neg')\n    return x * 2"
    r = run(code, [(-1, 0), (lambda f: f(3) == 6), (lambda f: f(-1))])
    assert [t["passed"] for t in r["tests"]] == [False, True, False]
    assert r["tests"][0]["note"] == "error" and r["tests"][2]["note"] == "error"


def test_a_tuple_input_is_splatted_and_a_list_is_not():
    assert run("def solve(a, b):\n    return a + b", [((1, 2), 3)])["correct"] is True
    assert run("def solve(xs):\n    return sum(xs)", [([1, 2], 3)])["correct"] is True


SETUP = "CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1), (2);"


@pytest.mark.parametrize("query,expected,passed,note", [
    ("SELECT a FROM t ORDER BY a", [(1,), (2,)], True, None),
    ("SELECT a FROM t ORDER BY a DESC", [(1,), (2,)], False, None),    # row order is part of the answer
    ("SELECT a FROM t; DROP TABLE t", [(1,), (2,)], False, "error"),    # one statement only
    ("DELETE FROM t", [], False, "error"),                              # the database is read-only
    ("SELEC a FROM t", [], False, "error"),
])
def test_sql_checks(query, expected, passed, note):
    [t] = run(query, [(SETUP, expected)], language="sql-lite")["tests"]
    assert t["passed"] is passed and t.get("note") == note


def test_equivalent_is_not_correct():
    wrong_a = CodeCase("a", "def solve(x):\n    return x * 3")
    wrong_b = CodeCase("b", "def solve(x):\n    return x + x + x")
    r = uq.functional_equivalence(wrong_a, wrong_b, [(1, 2), (2, 4)])
    assert r["equivalent"] is True and r["both_pass"] is False


@pytest.mark.parametrize("b,status", [
    (CodeCase("b", "SELECT 1", language="sql-lite"), "error"),          # language mismatch
    (CodeCase("b", "import os\ndef solve(x):\n    return x"), "unrunnable"),
])
def test_equivalence_needs_two_runnable_candidates_in_one_language(b, status):
    r = uq.functional_equivalence(CodeCase("a", "def solve(x):\n    return x"), b, [(1, 1)])
    assert r["status"] == status and r["equivalent"] is None


def test_equivalence_without_tests_is_unknown():
    a = CodeCase("a", "def solve(x):\n    return x")
    assert uq.functional_equivalence(a, a, [])["equivalent"] is None


def test_code_confidence_vs_correctness_by_hand():
    good = "def solve(x):\n    return x + 1"
    bad = "def solve(x):\n    return x"
    cases = [CodeCase("good", good, confidence=0.9, tests=[(1, 2)]),
             CodeCase("bad", bad, confidence=0.9, tests=[(1, 2)]),
             CodeCase("blocked", "import os\n" + good, confidence=0.9, tests=[(1, 2)]),
             CodeCase("unsure", bad, confidence=0.2, tests=[(1, 2)])]
    rep = uq.code_confidence_vs_correctness(cases)
    # confident & labelled: good (right), bad (wrong) -> 1/2 ; blocked is unlabelled, not wrong
    assert rep["confident_wrong_rate"] == pytest.approx(0.5)
    assert rep["false_pass"]["n_unlabeled"] == 1 and rep["calibration"]["n_labeled"] == 3


@pytest.mark.parametrize("kw", [{"confidence": float("nan")}, {"confidence": 1.1}, {"language": "bash"}])
def test_code_case_validation(kw):
    with pytest.raises(ValueError):
        CodeCase("x", "def solve():\n    return 1", **kw)
