"""AG-05: judge/source vs verified-oracle disagreement.

The verified oracle always wins the headline; a disagreeing judge or source
verdict is RECORDED (never silently dropped) and surfaced by a configurable
filter. Deterministic: a scripted judge that always says 'complete'.
"""

from __future__ import annotations

from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    FinalStateAssertion,
    episode_from_openai_messages,
)
from auditkit.score import Score


class _AlwaysComplete:
    """A diagnostic judge that always votes success (task complete)."""

    name = "task_completion"

    def score(self, sample, output, context):
        return Score("task_completion", 1.0, reason="looks complete",
                     metadata={"choice": "complete"})


def _failing_episode():
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "ship it"}, {"role": "assistant", "content": "All shipped!"}],
        case_id="c", final_answer="All shipped!")
    ep.final_state = {"shipped": False}       # the real state says it did NOT ship
    ep.source_verdict = True                  # imported source claims it passed
    return ep


def test_oracle_wins_and_disagreement_is_recorded():
    case = AgentCase(id="c", task="ship it",
                     outcome=FinalStateAssertion("shipped", equals=True))
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=[_failing_episode()],
                         judge=_AlwaysComplete(), scorers=["task_completion"])
    result = AgentEvalRunner().run(spec)
    row = result.rows[0]

    # Verified oracle wins the headline (state says failure), judge kept diagnostic.
    assert row.outcome["verdict"] == "failure"
    assert row.judge["verdict"] == "success"

    d = row.disagreement
    assert d["judge_vs_oracle"] == {"oracle": "failure", "judge": "success",
                                    "resolution": "verified oracle wins; judge kept as diagnostic"}
    # the imported source verdict also disagreed with the verified oracle
    assert d["source_vs_oracle"]["oracle"] == "failure"
    assert d["source_vs_oracle"]["source_verdict"] == "success"

    # filter surfaces the case; a high threshold still keeps a single-trial 1.0 rate
    assert len(result.disagreements()) == 1
    assert len(result.disagreements(min_rate=0.9)) == 1


def test_no_disagreement_when_verdicts_agree():
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "ship it"}, {"role": "assistant", "content": "done"}],
        case_id="c", final_answer="done")
    ep.final_state = {"shipped": True}        # oracle success, judge success -> agree
    case = AgentCase(id="c", task="ship it",
                     outcome=FinalStateAssertion("shipped", equals=True))
    result = AgentEvalRunner().run(AgentEvalSpec(
        cases=[case], mode="recorded", episodes=[ep],
        judge=_AlwaysComplete(), scorers=["task_completion"]))
    assert result.rows[0].disagreement == {}
    assert result.disagreements() == []


def test_unknown_oracle_is_not_a_disagreement():
    # No state recorded -> oracle 'unknown' (missing evidence), not disagreement.
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "ship it"}, {"role": "assistant", "content": "done"}],
        case_id="c", final_answer="done")
    case = AgentCase(id="c", task="ship it",
                     outcome=FinalStateAssertion("shipped", equals=True))
    result = AgentEvalRunner().run(AgentEvalSpec(
        cases=[case], mode="recorded", episodes=[ep],
        judge=_AlwaysComplete(), scorers=["task_completion"]))
    assert result.rows[0].outcome["verdict"] == "unknown"
    assert result.rows[0].disagreement == {}
