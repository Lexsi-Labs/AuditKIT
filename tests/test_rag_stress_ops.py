"""Deterministic tests for the RAG-stress operational checks (build steps 4-5).

No network, no LLM, no real load. They prove: judge_calibration's agreement /
Cohen's kappa / false-pass / false-negative are hand-checkable on a tiny set and
degenerate cases return None (never a silent 0); review_queue selects the right
cases within budget and records a reviewer override without auto-resolving;
operational_stress computes p95 from supplied samples, flags a false answer under
partial retrieval, reports the error budget, and marks every missing measure None;
every result round-trips as strict JSON.
"""

from __future__ import annotations

import json

import pytest

from auditkit.metrics.rag_stress_ops import (
    HumanReview, false_answer_under_partial_retrieval, judge_calibration,
    operational_stress, review_queue,
)
from auditkit.metrics.rag_stress_runner import (
    StressCaseResult, StressReport, generate_stress_cases, run_stress,
)


def _strict_json(obj) -> None:
    json.dumps(obj, allow_nan=False)  # raises on nan/inf or a non-JSON type


# -- judge_calibration -------------------------------------------------------

# 6 rows, pass=1/fail=0: pp=2, pf=1 (false negative), fp=1 (false pass), ff=2.
_HAND = [
    {"human": "pass", "judge": "pass"},
    {"human": True, "judge": True},
    {"human": "fail", "judge": "pass"},   # false pass
    {"human": "fail", "judge": "fail"},
    {"human": "pass", "judge": "fail"},   # false negative
    {"human": False, "judge": False},
]


def test_judge_calibration_hand_checked():
    out = judge_calibration(_HAND)
    assert out["status"] == "scored"
    assert out["confusion"] == {"pp": 2, "pf": 1, "fp": 1, "ff": 2}
    assert out["n"] == 6
    assert out["n_human_pass"] == 3 and out["n_human_fail"] == 3
    assert out["agreement"] == pytest.approx(2 / 3)      # po = (2+2)/6
    assert out["cohens_kappa"] == pytest.approx(1 / 3)   # (2/3 - 1/2)/(1/2)
    assert out["false_pass_rate"] == pytest.approx(1 / 3)      # fp / n_human_fail
    assert out["false_negative_rate"] == pytest.approx(1 / 3)  # pf / n_human_pass
    _strict_json(out)


def test_judge_calibration_degenerate_kappa_is_none_not_zero():
    # All agree on one class: chance agreement is 1.0, kappa undefined -> None.
    out = judge_calibration([{"human": True, "judge": True}] * 4)
    assert out["agreement"] == 1.0
    assert out["cohens_kappa"] is None            # not 0.0
    assert out["false_pass_rate"] is None         # no human-fail rows
    assert out["false_negative_rate"] == 0.0      # 0 of 4 human-pass were failed


def test_judge_calibration_no_labels_is_unknown():
    out = judge_calibration([{"judge": "pass"}, {"judge": "fail"}])
    assert out["status"] == "unknown"
    assert out["n"] == 0 and out["n_unlabeled"] == 2
    for k in ("agreement", "cohens_kappa", "false_pass_rate", "false_negative_rate"):
        assert out[k] is None                     # explicit unknown, never 0
    _strict_json(out)


def test_judge_calibration_missing_judge_counted_not_scored():
    out = judge_calibration([
        {"human": True, "judge": True},
        {"human": False},                         # judge produced no verdict
    ])
    assert out["n"] == 1 and out["n_missing_judge"] == 1


def test_judge_calibration_by_dimension():
    rows = [
        {"human": True, "judge": True, "dimension": "faithfulness"},
        {"human": True, "judge": False, "dimension": "citation"},
    ]
    out = judge_calibration(rows)
    assert set(out["by_dimension"]) == {"faithfulness", "citation"}
    assert out["by_dimension"]["faithfulness"]["agreement"] == 1.0
    assert out["by_dimension"]["citation"]["agreement"] == 0.0


def test_judge_calibration_bad_label_raises():
    with pytest.raises(ValueError):
        judge_calibration([{"human": "maybe", "judge": "pass"}])


# -- review_queue ------------------------------------------------------------


def _result(status: str, flagged: bool = False, value: float = 1.0) -> dict:
    return {"value": value, "label": None, "status": status,
            "flagged": flagged, "reason": None}


def _case(case_id, *, expected_failure, stress, stress_flagged,
          benign_control_passed=True) -> StressCaseResult:
    return StressCaseResult(
        case_id=case_id, stress_kind="synthetic", expected_failure=expected_failure,
        stress=stress, control={m: _result("scored") for m in stress},
        first_failure_stage=None, stress_flagged=stress_flagged,
        benign_control_passed=benign_control_passed)


def test_review_queue_selects_and_orders_within_budget():
    clean = _case("clean", expected_failure="freshness",
                  stress={"freshness": _result("scored", flagged=True, value=0.0),
                          "acl_compliance": _result("scored")},
                  stress_flagged=["freshness"])
    ambiguous = _case("ambiguous", expected_failure=None,
                      stress={"freshness": _result("unknown"),
                              "acl_compliance": _result("scored")},
                      stress_flagged=[])
    disagree = _case("disagree", expected_failure="acl_compliance",
                     stress={"acl_compliance": _result("scored", value=1.0)},
                     stress_flagged=[])   # expected metric did NOT flag -> disagreement
    report = StressReport(metrics=["freshness", "acl_compliance"],
                          cases=[clean, ambiguous, disagree])

    queue = review_queue(report, budget=2)
    assert [h.case_id for h in queue] == ["disagree", "ambiguous"]   # priority order
    assert "clean" not in {h.case_id for h in queue}                 # clean never queued
    assert queue[0].priority > queue[1].priority
    assert queue[1].unknown_metrics == ["freshness"]
    _strict_json([h.to_dict() for h in queue])


def test_review_queue_budget_respected():
    report = StressReport(metrics=["freshness"], cases=[
        _case(f"c{i}", expected_failure="freshness",
              stress={"freshness": _result("scored")}, stress_flagged=[])  # all disagree
        for i in range(5)])
    assert len(review_queue(report, budget=3)) == 3
    assert review_queue(report, budget=0) == []


def test_review_queue_records_override_without_auto_resolution():
    disagree = _case("disagree", expected_failure="acl_compliance",
                     stress={"acl_compliance": _result("scored")}, stress_flagged=[])
    queue = review_queue(StressReport(metrics=["acl_compliance"], cases=[disagree]), budget=5)
    hr = queue[0]
    # The queue surfaced work; it did not resolve it.
    assert hr.detected is None and hr.override is None and hr.final_answer_changed is None
    hr.record(detected=True, override="fail", final_answer_changed=True)
    assert hr.detected is True and hr.override == "fail" and hr.final_answer_changed is True
    assert hr.correction is None                      # unset field stays None
    assert hr.to_dict()["override"] == "fail"


def test_review_queue_on_real_report_prioritizes_and_excludes_resolved():
    # Generated fixtures are built so only expected_failure flags. The candidates
    # are the operational partial-retrieval cases (nothing scored, priority 10) and
    # the missing-evidence cases (acl unknown on empty retrieval, priority 5).
    # Resolved cases (stale_policy, poisoned_doc, ...) never queue.
    report = run_stress(generate_stress_cases(seed=0))   # 22 cases, repeats=2
    top = review_queue(report, budget=2)
    assert {h.case_id for h in top} == {"partial-0", "partial-1"}   # highest priority first
    all_candidates = {h.case_id for h in review_queue(report, budget=10)}
    assert all_candidates == {"partial-0", "partial-1", "missing-0", "missing-1"}
    assert not any(h.stress_kind in ("stale_policy", "poisoned_doc", "cross_tenant")
                   for h in review_queue(report, budget=10))
    assert len(review_queue(report, budget=1)) == 1


# -- false_answer_under_partial_retrieval ------------------------------------


def test_false_answer_check_truth_table():
    assert false_answer_under_partial_retrieval(
        {"partial_retrieval": True, "answered": True}) is True     # answered anyway: bad
    assert false_answer_under_partial_retrieval(
        {"partial_retrieval": True, "abstained": True}) is False   # abstained: safe
    assert false_answer_under_partial_retrieval(
        {"partial_retrieval": True}) is None                       # no answer signal
    assert false_answer_under_partial_retrieval(
        {"partial_retrieval": False, "answered": True}) is None    # not applicable
    assert false_answer_under_partial_retrieval({}) is None        # field absent


# -- operational_stress ------------------------------------------------------


def test_operational_stress_p95_from_supplied_samples():
    scn = {"name": "burst-1", "kind": "burst", "latencies_ms": list(range(21))}
    out = operational_stress([scn])
    r = out["scenarios"][0]
    assert r["p50_ms"] == 10 and r["p95_ms"] == 19   # exact, no interpolation
    _strict_json(out)


def test_operational_stress_error_budget():
    over = operational_stress([{"name": "o", "kind": "rate_limit",
                                "outcomes": ["ok", "ok", "ok", "error"],
                                "error_budget": 0.1}])["scenarios"][0]
    assert over["error_rate"] == pytest.approx(0.25)
    assert over["within_error_budget"] is False
    ok = operational_stress([{"name": "o", "kind": "rate_limit",
                              "outcomes": ["ok", "ok", "ok", "error"],
                              "error_budget": 0.5}])["scenarios"][0]
    assert ok["within_error_budget"] is True


def test_operational_stress_flags_false_answer_and_reports_slices():
    out = operational_stress([
        {"name": "safe", "kind": "partial_search_failure",
         "partial_retrieval": True, "abstained": True},
        {"name": "bad", "kind": "partial_search_failure",
         "partial_retrieval": True, "answered": True},
        {"name": "unknown", "kind": "partial_search_failure",
         "partial_retrieval": True},                       # no answer signal
    ])
    assert out["any_false_answer"] is True
    assert out["n_partial_retrieval"] == 3
    assert out["n_partial_retrieval_checked"] == 2
    assert out["n_partial_retrieval_unknown"] == 1          # untested slice surfaced
    _strict_json(out)


def test_operational_stress_missing_fields_are_none_not_zero():
    out = operational_stress([{"name": "bare", "kind": "concurrency"}])["scenarios"][0]
    assert out["p50_ms"] is None and out["p95_ms"] is None   # no timing samples
    assert out["error_rate"] is None                         # no outcomes
    assert out["within_error_budget"] is None                # no budget
    assert out["false_answer_under_partial_retrieval"] is None
    assert out["retried"] is None and out["recovered"] is None


def test_operational_stress_unknown_kind_raises():
    with pytest.raises(ValueError):
        operational_stress([{"name": "x", "kind": "not-a-kind"}])
