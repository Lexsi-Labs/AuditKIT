"""A result that was not measured is not a 0 (and not a pass).

Each test fails on the code before the fix: the unmeasured result was averaged in as 0,
ranked, graded, or read as safe.
"""

from __future__ import annotations

import auditkit as ak
from auditkit._bootstrap import score_pairs
from auditkit.comparison import DeltaGrade, RunComparison
from auditkit.diff import RunDiff
from auditkit.experiment import Experiment
from auditkit.metrics.guard import GuardJudge
from auditkit.model import Generated, Result_
from auditkit.model_compare import CompareResult
from auditkit.report import Prediction, RunResult
from auditkit.sample import Sample
from auditkit.score import Score, Stat
from auditkit.scoring import WeightedSum

UNMEASURED = {"unknown": True, "status": "not_measured"}


def _pred(i, value, *, unknown=False, task="t", metric="m"):
    doc = {"name": metric, "value": value}
    if unknown:
        doc["metadata"] = dict(UNMEASURED)
    return Prediction(run_id="r", task=task, sample_id=str(i), prompt="", raw_output="",
                      parsed_answer="", expected="", correct=None,
                      score=None if unknown else value, metadata={"scores": [doc]})


def _run(rid, preds=(), headline=None, stats=None):
    return RunResult(run_id=rid, fingerprint=rid, stats=stats or {}, predictions=list(preds),
                     headline=headline or {})


# (a) -----------------------------------------------------------------------

def test_leaderboard_run_without_the_metric_does_not_rank_first():
    exp = Experiment("e")
    exp.add(_run("a", headline={"loss": -0.5}))
    exp.add(_run("b", headline={}))
    assert [r["run_id"] for r in exp.leaderboard()] == ["a", "b"]


def test_score_pairs_skips_unmeasured_samples():
    base = _run("b", [_pred(0, 1.0), _pred(1, 0.0, unknown=True)])
    cand = _run("c", [_pred(0, 1.0), _pred(1, 1.0)])
    assert score_pairs(base, cand, "m") == [(1.0, 1.0)]


def test_per_task_delta_ignores_unmeasured_samples():
    base = _run("b", [_pred(0, 1.0), _pred(1, 0.0, unknown=True)])
    cand = _run("c", [_pred(0, 1.0), _pred(1, 0.0, unknown=True)])
    (td,) = RunComparison(base, cand).per_task_deltas()
    assert td.baseline == 1.0 and td.baseline_count == 1 and td.delta == 0.0


def test_sample_diff_does_not_call_an_unmeasured_sample_wrong():
    base = _run("b", [_pred(0, 1.0)])
    cand = _run("c", [_pred(0, 0.0, unknown=True)])
    (row,) = RunDiff(base, cand).sample_diff("m")
    assert row["change"] == "unknown"


def test_compare_summary_shows_na_for_a_model_without_the_metric():
    s = Stat("m")
    s.add(0.8)
    res = CompareResult(runs={"a": _run("a", headline={"m": 0.8}, stats={"m": s}),
                              "b": _run("b")}, dataset_size=1, scorers=["m"])
    text = res.summary()
    assert "0.0000" not in text and "n/a" in text


def test_compare_per_task_table_ignores_unmeasured_samples():
    a = _run("a", [_pred(0, 1.0), _pred(1, 0.0, unknown=True)])
    res = CompareResult(runs={"a": a}, dataset_size=2, scorers=["m"])
    (row,) = res.per_task_table()
    assert row["a_score"] == 1.0


# (b) -----------------------------------------------------------------------

class _Guard:
    """Replies "" to the sample whose text mentions q2, "unsafe" to the rest."""

    name = "g"

    def generate(self, requests):
        return [Result_(completions=[Generated(
            text="" if "q2" in str(r.params.get("messages") or r.prompt) else "unsafe\nS1")])
            for r in requests]


def test_empty_guard_reply_is_not_safe():
    m = GuardJudge(judge_model=_Guard())
    assert m.score(Sample(input="q1"), "a").metadata.get("unknown") is None
    assert m.score(Sample(input="q2"), "a").metadata.get("unknown") is True


def test_guard_headline_leaves_out_the_empty_reply():
    r = ak.evaluate([Sample(input="q1"), Sample(input="q2")],
                    model=lambda prompts: ["a"] * len(prompts),
                    scorers=[GuardJudge(judge_model=_Guard())])
    assert r.headline["guard_judge:llama_guard"] == 1.0   # 0.5 when the empty reply counted as safe


# (c) -----------------------------------------------------------------------

def test_weighted_sum_ignores_a_placeholder():
    flagged = Score(name="x", value=0.0, metadata=dict(UNMEASURED))
    assert WeightedSum().compute([flagged, Score(name="y", value=1.0)]).value == 1.0


def test_weighted_sum_of_only_placeholders_is_not_measured():
    out = WeightedSum().compute([Score(name="x", value=0.0, metadata=dict(UNMEASURED))])
    assert out.metadata.get("unknown") is True


# (d) -----------------------------------------------------------------------

def test_two_empty_runs_do_not_grade_pass():
    assert RunComparison(_run("a"), _run("b")).grade() != DeltaGrade.PASS


# review fixes ----------------------------------------------------------------

import pytest


class _Replies:
    """A guard model that answers every request with one fixed reply."""

    name = "r"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, requests):
        return [Result_(completions=[Generated(text=self.reply)]) for _ in requests]


def _guard_score(profile, reply):
    return GuardJudge(profile, judge_model=_Replies(reply)).score(Sample(input="q"), "a")


@pytest.mark.parametrize("profile", ["harmbench", "shield_gemma", "granite_guardian"])
@pytest.mark.parametrize("reply", ["None", "Not sure, I cannot classify this.", "Notably, yes - this violates the policy.",
                                   "nonsense", "Nope", "Yesterday"])
def test_a_reply_that_only_starts_with_yes_or_no_letters_is_not_a_verdict(profile, reply):
    assert _guard_score(profile, reply).metadata.get("unknown") is True


@pytest.mark.parametrize("profile", ["harmbench", "shield_gemma", "granite_guardian"])
@pytest.mark.parametrize("reply, value", [("Yes", 1.0), ("Yes.", 1.0), ("  yes\n", 1.0), ("No", 0.0), ("No, it is fine.", 0.0), ("NO", 0.0)])
def test_real_yes_no_verdicts_still_parse(profile, reply, value):
    s = _guard_score(profile, reply)
    assert s.metadata.get("unknown") is None and s.value == value


@pytest.mark.parametrize("reply", ["safely ignore this", "unsafety is a concept", "S1"])
def test_llama_guard_needs_the_whole_word_safe_or_unsafe(reply):
    assert _guard_score("llama_guard", reply).metadata.get("unknown") is True


@pytest.mark.parametrize("reply, value", [("safe", 0.0), ("\n\nsafe", 0.0), ("unsafe\nS1", 1.0), ("Unsafe: S2", 1.0), ("unsafe,S1", 1.0)])
def test_real_llama_guard_verdicts_still_parse(reply, value):
    s = _guard_score("llama_guard", reply)
    assert s.metadata.get("unknown") is None and s.value == value


def test_a_run_where_the_guard_gave_no_verdict_is_not_significantly_safer():
    samples = [Sample(id=str(i), input=f"q{i}") for i in range(8)]
    seen = ak.evaluate(samples, model=lambda ps: ["a"] * len(ps), scorers=[GuardJudge(judge_model=_Replies("unsafe\nS1"))])
    blank = ak.evaluate(samples, model=lambda ps: ["b"] * len(ps), scorers=[GuardJudge(judge_model=_Replies(""))])
    assert all(p.score is None for p in blank.predictions)          # a placeholder is not a score
    assert score_pairs(seen, blank, None) == []
    assert RunComparison(seen, blank).significance().get("significant") is not True


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
@pytest.mark.parametrize("backend", ["minimal", "jsonschema"])
def test_a_ref_string_inside_plain_data_is_not_a_reference(keyword, backend):
    if backend == "jsonschema":
        pytest.importorskip("jsonschema")
    from auditkit.metrics.structured import JsonSchemaValid
    data = {"$ref": "https://example.invalid/x"}
    value = [data] if keyword in ("enum", "examples") else data
    JsonSchemaValid({"type": "object", keyword: value}, backend=backend)   # builds: it is data, not a reference
