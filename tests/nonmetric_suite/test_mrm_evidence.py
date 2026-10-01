"""PR #3's evidence and review workflow, apart from the metrics: the schemas
(ModelUse, ValidationPlan, EvidenceBundle), jurisdiction views, findings, reviews,
regression freezing and replay, change triggers, the dossier export and the fixtures.

Skips before #3 merges.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("auditkit.mrm")
from auditkit.mrm import fixtures as F  # noqa: E402
from auditkit.mrm import jurisdiction as J  # noqa: E402
from auditkit.mrm import metrics as M  # noqa: E402
from auditkit.mrm import review as RV  # noqa: E402
from auditkit.mrm import tampering as T  # noqa: E402
from auditkit.mrm.schemas import DatasetRef, EvidenceBundle, ModelUse, ValidationPlan  # noqa: E402

MU = ModelUse("m", "1.0", "credit_underwriting", "approve_deny", "high", "score", "auto-decline")
PLAN = ValidationPlan("p", datasets=(DatasetRef("hold", "2026-06-30", digest="sha:a", n=10),),
                      metrics_requested=("brier_score",), acceptance_thresholds={"brier_score": 0.2})


def bundle(results=(), provenance=None, mu=MU, bundle_id="b"):
    return EvidenceBundle(bundle_id, mu, PLAN, tuple(results), ("synthetic",), provenance or {})


OK = M.brier_score([0.8, 0.3], [1, 0])                     # status ok
UNKNOWN = M.brier_score([0.5], [None])                     # status unknown


# -- schemas ---------------------------------------------------------------------------------------------

def test_every_schema_round_trips_through_strict_json():
    b = bundle([OK], {"run": "r1"})
    again = EvidenceBundle.from_dict(json.loads(json.dumps(b.to_dict())))
    assert again == b and again.fingerprint() == b.fingerprint()
    assert ModelUse.from_dict(MU.to_dict()) == MU and ValidationPlan.from_dict(PLAN.to_dict()) == PLAN


def test_the_fingerprint_covers_every_part_of_the_dossier():
    base = bundle([OK], {"run": "r1"})
    variants = [
        bundle([OK], {"run": "r2"}),                                        # provenance
        bundle([OK, UNKNOWN], {"run": "r1"}),                               # results
        bundle([OK], {"run": "r1"}, mu=ModelUse(**{**MU.to_dict(), "model_version": "1.1"})),
        EvidenceBundle("b", MU, PLAN, (OK,), ("other limitation",), {"run": "r1"}),
    ]
    assert len({base.fingerprint(), *(v.fingerprint() for v in variants)}) == 5


def test_the_bundle_id_is_a_label_not_evidence():
    assert bundle([OK], bundle_id="x").fingerprint() == bundle([OK], bundle_id="y").fingerprint()


def test_a_nan_in_a_result_fails_loudly():
    with pytest.raises(ValueError):
        bundle([{"name": "x", "value": float("nan")}]).fingerprint()


def test_there_is_no_approval_field_anywhere():
    def keys(obj):
        if isinstance(obj, dict):
            return set(obj) | {k for v in obj.values() for k in keys(v)}
        if isinstance(obj, list):
            return {k for v in obj for k in keys(v)}
        return set()
    names = " ".join(keys(bundle([OK]).to_dict())).lower()
    assert "approv" not in names and "sign_off" not in names


@pytest.mark.xfail(strict=True, reason="FINDING: EvidenceBundle is documented as immutable, but its "
                   "provenance dict (and the result dicts) can be changed after creation, which silently "
                   "changes its fingerprint.")
def test_a_frozen_bundle_cannot_change_under_its_fingerprint():
    b = bundle([OK], {"run": "r1"})
    fp = b.fingerprint()
    try:
        b.provenance["run"] = "r2"
    except TypeError:
        pass
    assert b.fingerprint() == fp


# -- jurisdiction views ----------------------------------------------------------------------------------------

def req(rid, by, **kw):
    return J.EvidenceRequirement(rid, f"requirement {rid}", by, **kw)


def profile(*reqs, jid="EU"):
    return J.JurisdictionProfile(jid, jid, "bank-2026-09", reqs)


@pytest.mark.parametrize("results,provenance,by,coverage", [
    ([OK], {}, "brier_score", "present"),
    ([UNKNOWN], {}, "brier_score", "missing"),                      # ran but unscored is a gap
    ([], {"model_card": "doc://mc"}, "model_card", "present"),       # provenance evidence
    ([], {"model_card": ""}, "model_card", "missing"),               # empty provenance value
    ([], {}, "anything", "missing"),
])
def test_evidence_coverage(results, provenance, by, coverage):
    v = J.jurisdiction_view(bundle(results, provenance), profile(req("r1", by)))
    assert v.binding_coverage[0]["coverage"] == coverage


def test_the_banks_applicability_decision_short_circuits_the_lookup():
    v = J.jurisdiction_view(bundle([OK]), profile(req("na", "brier_score", applicability_status="not_applicable"),
                                                  req("lr", "brier_score", applicability_status="needs_legal_review")))
    assert [r["coverage"] for r in v.binding_coverage] == ["not_applicable", "needs_legal_review"]
    assert v.binding_gaps == ("lr",)                                 # legal review is a gap, never a pass


def test_binding_and_guidance_are_reported_separately():
    v = J.jurisdiction_view(bundle(), profile(req("b", "x"), req("g", "y", binding=False)))
    assert v.binding_gaps == ("b",) and [r["requirement_id"] for r in v.guidance_coverage] == ["g"]
    assert v.coverage_summary()["guidance"]["missing"] == 1


def test_a_bare_country_id_is_refused():
    with pytest.raises(TypeError):
        J.jurisdiction_view(bundle(), "EU")


@pytest.mark.parametrize("make", [
    lambda: req("r", "x", applicability_status="maybe"),
    lambda: profile(req("r", "x"), req("r", "y")),
])
def test_invalid_profiles_are_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_perimeter_report_shares_one_bundle_fingerprint_and_is_order_independent():
    b = bundle([OK])
    eu, uk = profile(req("r", "brier_score"), jid="EU"), profile(req("r", "model_card"), jid="UK-PRA")
    a, c = J.perimeter_report(b, (eu, uk)), J.perimeter_report(b, (uk, eu))
    assert a["fingerprint"] == c["fingerprint"]
    assert {v["bundle_fingerprint"] for v in a["views"]} == {b.fingerprint()}
    assert a["coverage_by_jurisdiction"]["UK-PRA"]["binding_gaps"] == ["r"]
    assert "AE-DIFC" in a["open_research_jurisdictions"]


def test_no_profiles_means_no_coverage_whatever_the_tags_say():
    r = J.perimeter_report(bundle(mu=ModelUse(**{**MU.to_dict(), "jurisdiction_tags": ("EU", "US")})), ())
    assert r["views"] == [] and r["coverage_by_jurisdiction"] == {}


def test_duplicate_jurisdictions_are_rejected():
    with pytest.raises(ValueError):
        J.perimeter_report(bundle(), (profile(jid="EU"), profile(jid="EU")))


def test_profile_round_trip_keeps_its_fingerprint():
    p = profile(req("r", "x", binding=False, source_citation="Art. 9"))
    assert J.JurisdictionProfile.from_dict(json.loads(json.dumps(p.to_dict()))).fingerprint() == p.fingerprint()


# -- findings, reviews, regression replay --------------------------------------------------------------------------

def finding(status="confirmed", fid="f1"):
    return RV.Finding(fid, "desc", reviewer_status=status)


@pytest.mark.parametrize("status", ["open", "false_positive", "waived"])
def test_only_a_confirmed_finding_can_be_frozen(status):
    with pytest.raises(ValueError):
        RV.freeze_regression(finding(status), OK)


def test_an_invalid_review_status_is_rejected():
    with pytest.raises(ValueError):
        RV.Finding("f", "d", reviewer_status="approved")


def oracle(verdict, codes=()):
    return {"name": "action_oracle", "status": "ok", "verdict": verdict, "value": float(len(codes)),
            "violations": [{"code": c} for c in codes]}


RC = RV.freeze_regression(finding(), oracle("fail", ["amount_over_limit"]))


@pytest.mark.parametrize("new,status", [
    (None, "incomparable"),
    ({"name": "brier_score", "status": "ok"}, "incomparable"),                      # a different metric
    ({"name": "action_oracle", "status": "not_tested"}, "incomparable"),
    (oracle("pass"), "fixed"),
    (oracle("fail", ["amount_over_limit"]), "persistent"),
    (oracle("fail", ["recipient_not_allowed"]), "new"),
    (oracle("fail", ["amount_over_limit", "recipient_not_allowed"]), "new"),
])
def test_replay_against_a_frozen_oracle_case(new, status):
    assert RV.replay(RC, new)["status"] == status


def test_replay_of_a_metric_needs_a_frozen_threshold():
    bad = M.brier_score([0.1], [1])                                                # .81
    no_bar = RV.freeze_regression(finding(), bad)
    with_bar = RV.freeze_regression(finding(), bad, pass_threshold=0.2)
    better = M.brier_score([0.9], [1])                                             # .01
    assert RV.replay(no_bar, better)["status"] == "incomparable"
    assert RV.replay(with_bar, better)["status"] == "fixed"
    assert RV.replay(with_bar, M.brier_score([0.2], [1]))["status"] == "persistent"   # .64, same signature


def test_a_regression_case_round_trips():
    assert RV.RegressionCase.from_dict(json.loads(json.dumps(RC.to_dict()))).fingerprint() == RC.fingerprint()


@pytest.mark.xfail(strict=True, reason="FINDING: replay's threshold check assumes lower is better "
                   "(value <= pass_threshold). For tamper_delta (safety_delta, negative = worse) a safety "
                   "drop that got six times worse, -0.1 -> -0.6, is reported as 'fixed'.")
def test_replay_does_not_call_a_worse_safety_drop_fixed():
    budget = T.ModificationBudget(fine_tune_steps=100)
    rc = RV.freeze_regression(finding(), T.tamper_delta(0.9, 0.8, 0.8, 0.8, budget), pass_threshold=-0.05)
    assert RV.replay(rc, T.tamper_delta(0.9, 0.3, 0.8, 0.8, budget))["status"] != "fixed"


def test_review_fingerprint_ignores_the_timestamp_and_keeps_both_labels():
    a = RV.ReviewRecord("r1", "f1", "alice", "confirmed", automatic_label="pass", reviewer_label="fail",
                        reviewed_at="2026-09-01")
    b = RV.ReviewRecord.from_dict({**a.to_dict(), "reviewed_at": "2026-09-28"})
    assert a.fingerprint() == b.fingerprint()
    assert (b.automatic_label, b.reviewer_label) == ("pass", "fail")


# -- change triggers -------------------------------------------------------------------------------------------------

def test_change_triggers():
    old = bundle(provenance={"component_digests": {"prompt": "p1", "index": "i1"}, "data_digests": {"hold": "d1"}})
    new = bundle(provenance={"component_digests": {"prompt": "p2", "guard": "g1"}, "data_digests": {"hold": "d1"}},
                 mu=ModelUse(**{**MU.to_dict(), "model_version": "1.1"}))
    # prompt changed, index removed, guard added, model version bumped; data unchanged
    assert RV.change_triggers(old, new) == ["guard", "index", "model", "prompt"]
    assert RV.change_triggers(old, old) == []


def test_unrecorded_components_are_ignored():
    old = bundle(provenance={"component_digests": {"unknown_thing": "a"}})
    new = bundle(provenance={"component_digests": {"unknown_thing": "b"}})
    assert RV.change_triggers(old, new) == []


# -- dossier export -----------------------------------------------------------------------------------------------------

def test_export_is_order_independent_and_rejects_orphan_reviews():
    b = bundle([OK])
    f1, f2 = finding(fid="f1"), finding("open", fid="f2")
    r1, r2 = RV.ReviewRecord("r1", "f1", "alice", "confirmed"), RV.ReviewRecord("r2", "f2", "bob", "waived")
    assert (RV.export_evidence(b, (f1, f2), (r1, r2))["fingerprint"]
            == RV.export_evidence(b, (f2, f1), (r2, r1))["fingerprint"])
    with pytest.raises(ValueError):
        RV.export_evidence(b, (f1,), (r2,))
    assert "not an approval" in RV.export_evidence(b)["note"]


@pytest.mark.xfail(strict=True, reason="FINDING: export_evidence accepts two findings with the same id; the "
                   "dossier fingerprint then depends on the input order, breaking its 'regardless of input "
                   "order' guarantee (and a review pointing at that id is ambiguous).")
def test_duplicate_finding_ids_are_rejected():
    with pytest.raises(ValueError):
        RV.export_evidence(bundle(), (RV.Finding("dup", "first"), RV.Finding("dup", "second")))


# -- fixtures ------------------------------------------------------------------------------------------------------------

def test_the_credit_fixture_holds_its_documented_traps():
    d = F.synthetic_credit_cases()
    r = M.dated_backtest(d["probs"], d["labels"], d["decision_dates"], d["as_of"], d["horizon_days"])
    assert (r["n"], r["n_not_matured"], r["n_unobserved"]) == (8, 1, 1)


def test_the_fixture_bundle_is_reproducible():
    assert F.build_evidence_bundle().fingerprint() == F.build_evidence_bundle().fingerprint()
    assert all(r["status"] == "ok" for r in F.build_evidence_bundle().metric_results)
