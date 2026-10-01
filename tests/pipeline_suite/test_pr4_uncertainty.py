"""PR #4 (feat/uncertainty-evals): UQ importer, calibration reports, claims, code checks.

Every expected number is computed by hand in the comment next to it.
Skips when the ``auditkit.uncertainty`` package is not present (before #4 merges).
"""

from __future__ import annotations

import json

import pytest

uq = pytest.importorskip("auditkit.uncertainty")
UQResult = uq.UQResult


def R(item, conf, correct):
    return UQResult(item_id=item, confidence=conf, correct=correct)


# 4 items: a(0.9, right) b(0.9, wrong) c(0.1, wrong) d(0.1, wrong)
ITEMS = [R("a", 0.9, True), R("b", 0.9, False), R("c", 0.1, False), R("d", 0.1, False)]


# -- UQResult / importer -----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_uqresult_rejects_out_of_range_confidence(bad):
    with pytest.raises(ValueError):
        UQResult(item_id="x", confidence=bad)


def test_uqresult_allows_missing_confidence_and_roundtrips():
    r = UQResult(item_id=7, confidence=None, correct=True)
    assert r.item_id == "7" and r.confidence is None
    assert UQResult.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_import_labels_problems_instead_of_defaulting():
    rows = [{"prompt": "q", "response": "r", "confidence": 1.7, "extra_col": "keep"},      # out of range
            {"id": "k2", "response": "r", "confidence": 0.8}]
    out = uq.import_uqlm(rows)
    assert out[0].item_id == "0" and out[0].provenance.get("id_source") == "row_index"
    assert out[0].confidence is None and "confidence_issue" in out[0].provenance
    assert out[1].item_id == "k2" and out[1].confidence == 0.8


def test_import_flips_orientation_when_lower_is_confident():
    [r] = uq.import_uqlm([{"id": "x", "score": 0.2}], score_field="score", higher_is_confident=False)
    assert r.confidence == pytest.approx(0.8)


def test_import_reads_jsonl_path(tmp_path):
    p = tmp_path / "uq.jsonl"
    p.write_text("\n".join(json.dumps({"id": f"r{i}", "confidence": 0.5}) for i in range(3)))
    assert [r.item_id for r in uq.import_uqlm(str(p))] == ["r0", "r1", "r2"]


# -- calibration reports -------------------------------------------------------------------------

def test_confidence_calibration_ece_by_hand():
    # bin 9: mean conf .9, accuracy .5 -> |.4| * 2/4 = .2 ; bin 1: .1 vs 0 -> .1 * 2/4 = .05 ; ECE .25
    rep = uq.confidence_calibration(ITEMS, bins=10)
    assert rep["status"] == "ok" and rep["ece"] == pytest.approx(0.25)


def test_calibration_perfect_is_zero_and_confidence_one_lands_in_last_bin():
    rep = uq.confidence_calibration([R("a", 1.0, True), R("b", 0.0, False)], bins=10)
    assert rep["ece"] == pytest.approx(0.0)


def test_false_pass_rate_by_hand():
    # confident (>= .5): a, b -> 1 of 2 wrong = .5 ; deferred: c, d -> 0 of 2 right = 0
    rep = uq.false_pass_rate(ITEMS, threshold=0.5)
    assert rep["false_pass_rate"] == pytest.approx(0.5) and rep["false_reject_rate"] == pytest.approx(0.0)


def test_review_capacity_by_hand():
    # review the lowest 50% (c, d): 2 of the 3 errors caught
    rep = uq.review_capacity(ITEMS, review_budget=0.5)
    assert rep["n_reviewed"] == 2 and rep["errors_total"] == 3 and rep["catch_rate"] == pytest.approx(2 / 3)
    rates = [p["catch_rate"] for p in rep["curve"]]
    assert rates == sorted(rates)                                   # non-decreasing


def test_selective_risk_aurc_by_hand():
    # retained most-confident first (a, b, c, d): risks 0, .5, 2/3, .75 -> AURC = mean
    rep = uq.selective_risk(ITEMS)
    assert rep["aurc"] == pytest.approx((0 + 0.5 + 2 / 3 + 0.75) / 4)


@pytest.mark.parametrize("fn", ["confidence_calibration", "false_pass_rate", "review_capacity", "selective_risk"])
def test_reports_are_unknown_without_labels(fn):
    rep = getattr(uq, fn)([R("a", 0.9, None), R("b", None, True)])
    assert rep["status"] == "unknown"


def test_a_ranking_perfect_but_overconfident_scorer_still_fails_calibration():
    # every wrong item scored .95, every right one 1.0: ranking is perfect, calibration is not
    items = [R(f"r{i}", 1.0, True) for i in range(5)] + [R(f"w{i}", 0.95, False) for i in range(5)]
    assert uq.false_pass_rate(items, threshold=0.5)["false_pass_rate"] == pytest.approx(0.5)
    assert uq.confidence_calibration(items)["ece"] > 0.4


# -- claims --------------------------------------------------------------------------------------

def test_duplicate_claim_ids_are_rejected():
    with pytest.raises(ValueError):
        uq.decompose_answer("a", [{"claim_id": "1", "text": "x"}, {"claim_id": "1", "text": "y"}])


def test_confident_false_claims_are_localized():
    claims = [uq.Claim("c1", "true", 0.9, True), uq.Claim("c2", "false", 0.8, False),
              uq.Claim("c3", "unlabeled", 0.9, None), uq.Claim("c4", "hedged false", 0.2, False)]
    rep = uq.confident_false_claim_rate(claims, threshold=0.5)
    # confident & labeled: c1, c2 -> 1 of 2 unsupported ; c3 unlabeled is not counted as wrong
    assert rep["confident_false_claim_rate"] == pytest.approx(0.5)
    assert rep["confident_false_claim_ids"] == ["c2"]


# -- code evaluation -------------------------------------------------------------------------------

def test_code_case_passes_and_fails_on_its_tests():
    ok = uq.CodeCase("ok", "def solve(a, b):\n    return a + b", tests=[((1, 2), 3), ((0, 0), 0)])
    bad = uq.CodeCase("bad", "def solve(a, b):\n    return a - b", tests=[((1, 2), 3), ((0, 0), 0)])
    r_ok, r_bad = uq.run_execution_tests(ok), uq.run_execution_tests(bad)
    assert r_ok["status"] == "ok" and r_ok["correct"] is True and r_ok["n_passed"] == 2
    assert r_bad["correct"] is False and r_bad["n_failed"] == 1


@pytest.mark.parametrize("code,status", [
    ("import os\ndef solve():\n    return 1", "blocked"),
    ("def solve(:\n    pass", "syntax_error"),
])
def test_unsafe_or_broken_code_never_runs(code, status):
    r = uq.run_execution_tests(uq.CodeCase("x", code, tests=[((), 1)]))
    assert r["status"] == status


def test_no_tests_is_not_a_pass():
    r = uq.run_execution_tests(uq.CodeCase("x", "def solve():\n    return 1"))
    assert r["status"] == "no_tests" and r["correct"] is None


def test_functional_equivalence():
    tests = [(1, 2), (3, 6)]
    a = uq.CodeCase("a", "def solve(x):\n    return x * 2")
    b = uq.CodeCase("b", "def solve(x):\n    return x + x")
    c = uq.CodeCase("c", "def solve(x):\n    return x + 1")          # agrees on 1 -> 2, not on 3
    assert uq.functional_equivalence(a, b, tests)["equivalent"] is True
    assert uq.functional_equivalence(a, c, tests)["equivalent"] is False


def test_invalid_code_case_rejected():
    with pytest.raises(ValueError):
        uq.CodeCase("x", "select 1", language="cobol")
