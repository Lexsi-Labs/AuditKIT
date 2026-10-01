"""A3: repeated-trial reliability (pass@k / all-k / variance / stable aggregate).

Deterministic given scripted per-trial verdicts. Independence is claimed only
with reset_confirmed=True (AG-12); a single trial is an explicit 'unknown'.
"""

from __future__ import annotations

import math

from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    FinalStateAssertion,
    aggregate_verdict,
    episode_from_openai_messages,
    reliability,
)


def _ep(state, cid="c"):
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "t"}, {"role": "assistant", "content": "done"}],
        case_id=cid, final_answer="done")
    ep.final_state = state
    return ep


# -- the reliability() estimator (hand-verified numbers) --------------------
def test_reliability_two_of_three_hand_checked():
    r = reliability(["success", "success", "failure"], independent=False)
    assert r["n_trials"] == 3 and r["n_success"] == 2 and r["n_failure"] == 1
    assert r["n_decided"] == 3
    assert r["pass_at_k"] == 1.0                       # >=1 of 3 passes
    assert r["all_k"] == 0.0                            # not all 3 pass
    assert math.isclose(r["success_rate"], 2 / 3)
    assert math.isclose(r["consistency"], 2 / 3)
    assert math.isclose(r["variance"], 2 / 9)
    assert r["status"] == "mixed"
    assert r["independent"] is False
    assert any("independence unconfirmed" in f for f in r["flags"])


def test_reliability_single_trial_is_explicit_unknown():
    r = reliability(["success"], independent=True)
    assert r["status"] == "unknown"
    assert "single trial" in r["reason"]
    assert "pass_at_k" not in r          # nothing to estimate


def test_reliability_all_pass_confirmed_independent():
    r = reliability(["success", "success"], independent=True, k=1)
    assert r["status"] == "reliable" and r["all_k"] == 1.0 and r["pass_at_k"] == 1.0
    assert r["independent"] is True
    assert not any("independence unconfirmed" in f for f in r["flags"])


def test_reliability_flags_duplicate_trial_id():
    r = reliability(["success", "success"], trial_ids=["t0", "t0"], independent=True)
    assert any("duplicate trial_id" in f for f in r["flags"])


def test_aggregate_verdict_majority_and_tie():
    assert aggregate_verdict(["success", "success", "failure"])[0] == "success"
    assert aggregate_verdict(["success", "failure"])[0] == "unknown"        # tie
    assert aggregate_verdict(["unknown", "unknown"])[0] == "unknown"        # none decided


# -- runner integration: N recorded episodes per case aggregate to one row --
def test_runner_trials_aggregate_reliability():
    eps = [_ep({"shipped": True}), _ep({"shipped": True}), _ep({"shipped": False})]
    case = AgentCase(id="c", task="ship it", outcome=FinalStateAssertion("shipped", equals=True))
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=eps, trials=3, scorers=[])
    row = AgentEvalRunner().run(spec).rows[0]
    assert row.reliability["n_trials"] == 3
    assert row.reliability["n_success"] == 2 and row.reliability["n_failure"] == 1
    assert row.outcome["verdict"] == "success"          # 2/3 majority stable aggregate
    assert "2/3" in row.outcome["reason"]
    assert row.reliability["independent"] is False       # no reset contract confirmed
    assert len(row.trials) == 3


def test_runner_reset_confirmed_claims_independence():
    eps = [_ep({"shipped": True}), _ep({"shipped": True})]
    case = AgentCase(id="c", task="ship it", outcome=FinalStateAssertion("shipped", equals=True))
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=eps, trials=2,
                         reset_confirmed=True, scorers=[])
    row = AgentEvalRunner().run(spec).rows[0]
    assert row.reliability["independent"] is True
    assert row.reliability["status"] == "reliable"


def test_reliability_needs_more_than_one_decided_trial():
    """One decided trial padded with errors is not a reliability estimate."""
    from auditkit.agent_eval.reliability import reliability
    r = reliability(["success", "error"])
    assert r["status"] == "unknown" and "decided" in r["reason"]
