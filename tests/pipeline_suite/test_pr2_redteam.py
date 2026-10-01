"""PR #2 (feat/redteam-attack-suite): the attack suite's accounting, offline.

A scripted target (complies only when the behavior contains COMPLY) and a scripted,
behavior-aware judge make every verdict known in advance, so ASR, the unknown/error
bookkeeping, budgets and labels can be checked exactly. GPU attack paths (GCG/PGD)
are out of scope here. Skips when the attack suite is not present (before #2 merges).
"""

from __future__ import annotations

import pytest

rt = pytest.importorskip("auditkit.redteam")
if not hasattr(rt, "AttackSuite"):
    pytest.skip("AttackSuite not present (PR #2 not merged)", allow_module_level=True)

from auditkit.redteam import (  # noqa: E402
    AttackCase, AttackSuite, BestOfN, Budget, Judge, JudgeVerdict, RefusalJudge,
)

COMPLY = "Sure, here is exactly how to do it, step by step, with all the details you asked for."
REFUSE = "I'm sorry, I can't help with that."


def target(prompts):
    return [COMPLY if "COMPLY" in p else REFUSE for p in prompts]


class ScriptedJudge(Judge):
    name = "scripted"
    behavior_aware = True

    def judge(self, behavior, output, *, prompt=None, transcript=None):
        if "UNKNOWN" in behavior:
            return JudgeVerdict(False, 0.0, "cannot tell", self.name, unknown=True)
        if "BOOM" in behavior:
            raise RuntimeError("judge crashed")
        ok = output.startswith("Sure")
        return JudgeVerdict(ok, 1.0 if ok else 0.0, "scripted", self.name)


CASES = [AttackCase("c1", "COMPLY one"), AttackCase("c2", "refuse two"),
         AttackCase("c3", "UNKNOWN three"), AttackCase("c4", "BOOM four")]


def run(cases=CASES, methods=(), **kw):
    return AttackSuite(kw.pop("tgt", target), judge=kw.pop("judge", ScriptedJudge()), **kw).run(cases, list(methods))


def test_baseline_is_always_included():
    assert run().methods() == ["direct_request"]


def test_asr_keeps_unknown_and_judge_errors_in_the_denominator():
    # c1 success, c2 failure, c3 unknown verdict, c4 judge crash -> ASR 1/4, decided ASR 1/2
    r = run()
    m = "direct_request"
    assert r.asr_counts(m) == (1, 4) and r.asr(m) == pytest.approx(0.25)
    assert r.decided_asr(m) == pytest.approx(0.5)
    assert r.unknown_count(m) == 2


def test_verdicts_and_statuses_per_case():
    by_case = {a.case_id: (a.final_verdict, a.status) for a in run().attempts}
    assert by_case == {"c1": ("success", "completed"), "c2": ("failure", "completed"),
                       "c3": ("unknown", "judge_error"), "c4": ("unknown", "judge_error")}


def test_a_crashing_judge_does_not_sink_the_run():
    r = run([AttackCase("c4", "BOOM"), AttackCase("c1", "COMPLY")])
    assert len(r.attempts) == 2 and r.attempts[1].final_verdict == "success"


def test_a_failing_target_is_an_error_never_a_defended_attempt():
    def down(prompts):
        raise ConnectionError("target down")
    [a] = run([AttackCase("c1", "COMPLY")], tgt=down).attempts
    assert (a.final_verdict, a.status) == ("error", "target_error")
    assert a.final_verdict != "failure"


def test_target_query_budget_is_respected():
    r = run([AttackCase("c1", "refuse")], [BestOfN(n=5)], budget=Budget(max_target_queries=2))
    assert all(a.target_queries <= 2 for a in r.attempts)


def test_two_configs_of_one_method_are_reported_separately():
    r = run([AttackCase("c1", "refuse")], [BestOfN(n=2), BestOfN(n=3)])
    assert r.methods() == ["direct_request", "best_of_n#1", "best_of_n#2"]


def test_runs_are_deterministic():
    a = [(x.case_id, x.method, x.final_verdict, x.prompt) for x in run(CASES[:2], [BestOfN(n=3, seed=7)]).attempts]
    b = [(x.case_id, x.method, x.final_verdict, x.prompt) for x in run(CASES[:2], [BestOfN(n=3, seed=7)]).attempts]
    assert a == b


def test_manifest_records_reproducibility_without_secrets():
    m = run().manifest
    assert {"case_set_sha256", "auditkit_version", "judge", "settings", "budget"} <= set(m)
    assert "api_key" not in str(m).lower()


def test_attempts_serialize_to_json():
    import json
    for a in run().attempts:
        json.dumps(a.to_dict())


@pytest.mark.parametrize("output,success", [
    (COMPLY, True),
    (REFUSE, False),
    ("ok", False),                                    # too short to be substantive
])
def test_refusal_judge(output, success):
    assert RefusalJudge().judge("anything", output).success is success
