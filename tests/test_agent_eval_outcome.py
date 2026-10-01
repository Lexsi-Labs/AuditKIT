"""AG-07 (deterministic oracles, missing evidence -> unknown) and AG-08
(judge is diagnostic; parse failure -> unknown; never overrides state)."""

from __future__ import annotations

import pytest

from auditkit.agent_eval import (
    AgentCase,
    AnswerAssertion,
    ArtifactAssertion,
    CustomPredicate,
    FinalStateAssertion,
    episode_from_openai_messages,
    judge_outcome,
    resolve_outcome,
)
from auditkit.model import CallableModel


def _episode(answer, *, final_state=None, artifacts=None):
    ep = episode_from_openai_messages(
        [{"role": "assistant", "content": answer}], final_answer=answer)
    ep.final_state = final_state
    ep.artifacts = artifacts
    return ep


# -- AG-07: fabricated correct answer + wrong state FAILS a state case ------
def test_fabricated_answer_wrong_state_fails():
    ep = _episode("Shipment is ready.", final_state={"shipment_decision": "blocked"})
    case = AgentCase(id="s", task="ship?",
                     outcome=FinalStateAssertion("shipment_decision", equals="ready"),
                     metadata={"target": "Shipment is ready."})
    oc = resolve_outcome(ep, case)
    assert oc.verdict == "failure"


def test_state_match_succeeds():
    ep = _episode("done", final_state={"shipment_decision": "ready"})
    case = AgentCase(id="s", task="ship?",
                     outcome=FinalStateAssertion("shipment_decision", equals="ready"))
    assert resolve_outcome(ep, case).verdict == "success"


def test_no_state_snapshot_against_state_oracle_is_unknown():
    ep = _episode("Shipment is ready.")  # no final_state recorded
    case = AgentCase(id="s", task="ship?",
                     outcome=FinalStateAssertion("shipment_decision", equals="ready"))
    oc = resolve_outcome(ep, case)
    assert oc.verdict == "unknown"
    assert "state" in oc.reason.lower()


def test_artifact_assertion_contains():
    ep = _episode("done", artifacts={"report": "Q3 revenue was 42"})
    case = AgentCase(id="a", task="report",
                     outcome=ArtifactAssertion("report", contains="42"))
    assert resolve_outcome(ep, case).verdict == "success"
    case_missing = AgentCase(id="a2", task="report",
                             outcome=ArtifactAssertion("missing", equals="x"))
    assert resolve_outcome(ep, case_missing).verdict == "unknown"


def test_answer_assertion_modes():
    ep = _episode("The answer is Paris.")
    assert AnswerAssertion("paris", mode="contains").evaluate(ep).verdict == "success"
    assert AnswerAssertion("Paris", mode="exact").evaluate(ep).verdict == "failure"
    assert AnswerAssertion("The answer is Paris.", mode="exact").evaluate(ep).verdict == "success"


def test_no_oracle_is_unknown():
    ep = _episode("x")
    case = AgentCase(id="c", task="t")  # no outcome
    assert resolve_outcome(ep, case).verdict == "unknown"


# -- custom predicate needs an explicit version (reproducibility) -----------
def test_custom_predicate_requires_version():
    with pytest.raises(ValueError):
        CustomPredicate(lambda ep, case: True, version="")


def test_custom_predicate_runs_and_has_identity():
    oc_true = CustomPredicate(lambda ep, case: ep.final_answer == "yes", version="1")
    ep = _episode("yes")
    out = oc_true.evaluate(ep, None)
    assert out.verdict == "success"
    assert out.source["version"] == "1" and out.source["type"] == "custom_predicate"


def test_custom_predicate_error_is_error_not_failure():
    def boom(ep, case):
        raise RuntimeError("bad")
    assert CustomPredicate(boom, version="1").evaluate(_episode("x")).verdict == "error"


# -- AG-08: judge is diagnostic; parse failure -> unknown -------------------
def _judge(reply_text):
    from auditkit.metrics.agent import TaskCompletion
    return TaskCompletion(judge_model=CallableModel(lambda prompts: [reply_text]))


def test_judge_parse_failure_is_unknown():
    ep = _episode("some answer")
    case = AgentCase(id="c", task="t")
    oc = judge_outcome(ep, case, _judge("I really cannot tell you either way here."))
    assert oc.verdict == "unknown"
    assert oc.diagnostic is True
    assert oc.evidence.get("parse_status") == "failed"


def test_judge_success_is_diagnostic():
    ep = _episode("done well")
    case = AgentCase(id="c", task="t")
    oc = judge_outcome(ep, case, _judge("Reasoning here.\nCHOICE: complete"))
    assert oc.verdict == "success"
    assert oc.diagnostic is True


def test_judge_does_not_override_verified_state():
    """A judge that says 'complete' cannot flip a verified state FAILURE."""
    ep = _episode("Shipment is ready.", final_state={"shipment_decision": "blocked"})
    case = AgentCase(id="s", task="ship?",
                     outcome=FinalStateAssertion("shipment_decision", equals="ready"))
    verified = resolve_outcome(ep, case)
    judged = judge_outcome(ep, case, _judge("CHOICE: complete"))
    assert verified.verdict == "failure"   # authoritative
    assert judged.verdict == "success" and judged.diagnostic  # diagnostic only


# -- D4 / D5 (fix plan #15) --------------------------------------------------------------------------------

@pytest.mark.parametrize("ref,answer,mode,verdict", [
    ("1000", "1,000", "normalized", "success"),          # D4: thousands separator
    ("$1,000", "1000", "normalized", "success"),
    ("1,5", "15", "normalized", "failure"),              # a decimal comma is not a thousands separator
    ("5", "The fee is 15%", "contains", "failure"),      # D5: not inside a longer number
    ("5", "The fee is 0.5%", "contains", "failure"),
    ("5", "The fee is 5%.", "contains", "success"),
    ("paris", "The answer is Paris.", "contains", "success"),
    ("new york", "It is in New York City.", "contains", "success"),
])
def test_d4_d5_answer_matching(ref, answer, mode, verdict):
    assert AnswerAssertion(ref, mode=mode).evaluate(_episode(answer)).verdict == verdict
