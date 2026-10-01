"""Results that are not measurements never enter the aggregate.

A Score flagged ``metadata["unknown"]`` carries a placeholder value. Before this
change the Runner averaged it in, so an unreadable judge verdict pulled a headline
down (1.0 + unknown -> 0.5), a ``not_tested`` RAG-stress result did the same, and for
a lower-is-better metric a placeholder 0.0 made the model look better.

Now:
- ``unknown`` (the metric could not decide): left out of the average, recorded in
  ``RunResult.errors`` like a metric that raised (the rule task_completion follows).
- ``not_tested`` (nothing to measure): left out, not an error.
- both are counted in ``RunResult.unscored``.
- ``LLMJudge(unknown_score=x)`` is an explicit opt-in: ``x`` is averaged, no error.
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metric import Metric
from auditkit.metrics.judge import LLMJudge
from auditkit.model import CallableModel
from auditkit.report import RunResult
from auditkit.runner import unscored_status
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def _samples(n=2):
    return [ak.Sample(id=f"s{i}", input="q", actual_output="a") for i in range(n)]


def _judge(replies, **kw):
    it = iter(replies)
    return LLMJudge(judge_model=CallableModel(lambda ps: [next(it) for _ in ps]),
                    choices={"yes": 1.0, "no": 0.0}, prompt="{output}", **kw)


# -- LLM judges ----------------------------------------------------------------------------------

def test_unreadable_judge_verdict_is_an_error_not_a_zero():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["CHOICE: yes", "Unclear."])])
    assert r.headline["llm_judge"] == 1.0                          # was 0.5
    assert len(r.errors) == 1 and r.errors[0]["sample_id"] == "s1" and r.errors[0]["metric"] == "llm_judge"
    assert r.unscored == {"llm_judge": {"unknown": 1, "not_tested": 0}}
    assert r.stats["llm_judge"].count == 1


def test_all_unreadable_means_no_headline_at_all():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["??", "??"])])
    assert "llm_judge" not in r.headline and len(r.errors) == 2


def test_unknown_score_opt_in_is_averaged_as_asked():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["CHOICE: yes", "??"], unknown_score=0.5)])
    assert r.headline["llm_judge"] == pytest.approx(0.75) and r.errors == []
    assert r.unscored == {}


def test_unknown_score_is_part_of_the_judge_identity():
    assert _judge([]).identity() != _judge([], unknown_score=0.5).identity()


def test_the_unknown_flag_is_still_on_the_prediction():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["CHOICE: yes", "??"])])
    flagged = [s for p in r.predictions for s in p.metadata["scores"] if (s.get("metadata") or {}).get("unknown")]
    assert len(flagged) == 1


def test_unreadable_verdict_counts_as_a_failed_sample():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["CHOICE: yes", "??"])])
    by_id = {p.sample_id: p for p in r.predictions}
    assert by_id["s1"].correct is False and r.failed_count == 1


def test_tool_selection_all_unparsed_is_an_error():
    s = ak.Sample(id="a", input="q", actual_output="",
                  actual_trace={"tool_calls": [[{"name": "f", "arguments": {}}]]})
    r = ak.evaluate([s], model="precomputed",
                    scorers=[ak.ToolSelectionJudge(judge_model=CallableModel(lambda ps: ["??"] * len(ps)))])
    assert "tool_selection" not in r.headline and len(r.errors) == 1


def test_task_completion_still_errors_as_before():
    r = ak.evaluate(_samples(1), model="precomputed",
                    scorers=[ak.TaskCompletion(judge_model=CallableModel(lambda ps: ["??"] * len(ps)))])
    assert "task_completion" not in r.headline and len(r.errors) == 1


# -- not_tested vs unknown, and lower-is-better metrics -----------------------------------------------

class Flagged(Metric):
    """Returns a real value, or a flagged placeholder, depending on the sample id."""

    name = "flagged"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MINIMIZE

    def score(self, sample, output, context=None):
        if sample.id == "real":
            return Score(name=self.name, value=0.81, kind=self.kind)
        status = sample.id                                         # "unknown" / "not_tested"
        return Score(name=self.name, value=0.0, kind=self.kind, label=status, reason=f"{status} here",
                     metadata={"unknown": True, "status": status})


def _ids(*ids):
    return [ak.Sample(id=i, input="q", actual_output="x") for i in ids]


def test_placeholder_zero_no_longer_flatters_a_lower_is_better_metric():
    r = ak.evaluate(_ids("real", "unknown"), model="precomputed", scorers=[Flagged()])
    assert r.headline["flagged"] == pytest.approx(0.81)            # was 0.405


def test_not_tested_is_left_out_but_not_an_error():
    r = ak.evaluate(_ids("real", "not_tested"), model="precomputed", scorers=[Flagged()])
    assert r.headline["flagged"] == pytest.approx(0.81) and r.errors == []
    assert r.unscored == {"flagged": {"unknown": 0, "not_tested": 1}} and r.failed_count == 0


def test_rag_stress_not_tested_is_left_out():
    rs = pytest.importorskip("auditkit.metrics.rag_stress")
    corpus, cases = rs.synthetic_bank_kb()
    c1 = cases[0]
    meta = {"rag_case": c1.to_dict(), "corpus": corpus.to_dict()}
    ss = [ak.Sample(id="m", input=c1.question, actual_output="x", metadata=meta,
                    actual_trace={"retrieved": [{"id": "SUMMARY"}]}),
          ak.Sample(id="u", input=c1.question, actual_output="x", metadata=meta, actual_trace={})]
    r = ak.evaluate(ss, model="precomputed", scorers=[rs.EvidenceSetRecall()])
    assert r.headline["evidence_set_recall"] == 1.0                 # was 0.5
    assert r.unscored["evidence_set_recall"]["not_tested"] == 1 and r.errors == []


def test_mrm_brier_placeholder_is_left_out_when_present():
    M = pytest.importorskip("auditkit.mrm.metrics")
    ss = [ak.Sample(id="a", input="q", target="1", actual_output="0.1"),
          ak.Sample(id="b", input="q", target="1", actual_output="not a probability")]
    r = ak.evaluate(ss, model="precomputed", scorers=[M.BrierScoreMetric()])
    assert r.headline["brier_score"] == pytest.approx(0.81)


# -- the helper and serialization --------------------------------------------------------------------

@pytest.mark.parametrize("meta,label,want", [
    ({}, None, None),
    ({"unknown": True}, None, "unknown"),
    ({"unknown": True, "status": "not_tested"}, None, "not_tested"),
    ({"unknown": True}, "not_tested", "not_tested"),
    ({"unknown": True, "count_in_aggregate": True}, None, None),
])
def test_unscored_status(meta, label, want):
    assert unscored_status(Score(name="m", value=0.0, kind=ScoreKind.BENCHMARK, label=label, metadata=meta)) == want


def test_unscored_survives_serialization():
    r = ak.evaluate(_samples(), model="precomputed", scorers=[_judge(["CHOICE: yes", "??"])])
    assert RunResult.from_dict(r.to_dict()).unscored == r.unscored
