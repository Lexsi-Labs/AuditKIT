"""Tests for auditkit.metrics.agent: tool-call F1, trajectories, parallel calls,
schema validity, redundancy, the TaskCompletion judge, and the offline
(model="precomputed") path end to end."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

import auditkit as ak
from auditkit.metrics.agent import (
    ParallelToolCalls, RedundantToolCalls, TaskCompletion, ToolCallF1, ToolCallValidity, TrajectoryMatch,
)
from auditkit.model import CallableModel
from auditkit.sample import Sample
from auditkit.scenario import ListScenario
from auditkit.types import Direction, ScoreKind


def c(name, **args):
    return {"name": name, "arguments": args}


A, B, C, D, X = c("A", v=1), c("B", v=2), c("C", v=3), c("D", v=4), c("X", v=9)


def run(metric, expected, predicted=None, output="", tools=None, target=None):
    """{score_name: value} for one sample; predicted turns go in the trace."""
    s = Sample(input="task", expected_tool_calls=expected, tools=tools, target=target)
    ctx = {"trace": {"tool_calls": predicted}} if predicted is not None else {}
    out = metric.score(s, output, ctx)
    return {x.name: x.value for x in (out if isinstance(out, list) else [out])}


# -- ToolCallF1 ---------------------------------------------------------------

def test_f1_hand_computed():
    got = run(ToolCallF1(), [[A, B], [C]], [[A], [B], [D]])
    assert got == pytest.approx({"tool_call_precision": 2 / 3, "tool_call_recall": 2 / 3,
                                 "tool_call_f1": 2 / 3, "tool_call_exact": 0.0})


def test_f1_ignores_turn_boundaries_and_order():
    assert run(ToolCallF1(), [[A, B], [C]], [[C, B, A]])["tool_call_exact"] == 1.0


def test_f1_counts_duplicates_as_a_multiset():
    # Set-based F1 (RAGAS) would call [A, A] vs [A] perfect.
    got = run(ToolCallF1(), [A], [[A, A]])
    assert got["tool_call_precision"] == 0.5 and got["tool_call_recall"] == 1.0
    assert got["tool_call_f1"] == pytest.approx(2 / 3) and got["tool_call_exact"] == 0.0
    got = run(ToolCallF1(), [[A], [A]], [[A]])
    assert got["tool_call_precision"] == 1.0 and got["tool_call_recall"] == 0.5


def test_f1_irrelevance_case():
    assert set(run(ToolCallF1(), [], []).values()) == {1.0}
    assert set(run(ToolCallF1(), [], [[A]]).values()) == {0.0}
    # text path: a plain answer on an irrelevance sample is perfect
    assert set(run(ToolCallF1(), [], output="I can't do that with these tools.").values()) == {1.0}
    # ... and a JSON-formatted answer is not a phantom call
    assert set(run(ToolCallF1(), [], output='{"answer": "Paris"}').values()) == {1.0}


def test_f1_no_prediction_against_nonempty_reference():
    got = run(ToolCallF1(), [A, B], [])
    assert set(got.values()) == {0.0}


def test_f1_skipped_without_reference():
    assert not ToolCallF1().applicable(Sample(input="q"))
    assert ToolCallF1().applicable(Sample(input="q", expected_tool_calls=[]))


def test_f1_parses_text_output_when_no_trace():
    out = ('<tool_call>{"name": "A", "arguments": {"v": 1}}</tool_call>\n'
           '<tool_call>{"name": "B", "arguments": {"v": 2}}</tool_call>')
    assert run(ToolCallF1(), [[A, B]], output=out)["tool_call_exact"] == 1.0


def test_f1_malformed_arguments_never_match():
    bad = {"function": {"name": "A", "arguments": '{"v": 1'}}
    got = run(ToolCallF1(arg_mode="name"), [A], [[bad]])
    assert got["tool_call_f1_name"] == 0.0


def test_f1_arg_modes_and_names():
    pred = [[c("A", v=1, extra=True)]]
    assert run(ToolCallF1(), [A], pred)["tool_call_f1"] == 0.0
    assert run(ToolCallF1(arg_mode="subset"), [A], pred)["tool_call_f1_subset"] == 1.0
    assert run(ToolCallF1(arg_mode="name"), [A], [[c("A", v=99)]])["tool_call_f1_name"] == 1.0
    assert ToolCallF1(arg_mode="subset").name == "tool_call_f1_subset"
    with pytest.raises(ValueError):
        ToolCallF1(arg_mode="fuzzy")


def test_f1_custom_arg_match_with_allowed_values():
    allowed = lambda name, p, r: all(p.get(k) in v for k, v in r.items())  # noqa: E731
    ref = [{"A": {"x": [1, 2]}}, {"A": {"x": [1]}}]  # BFCL ground-truth shape
    m = ToolCallF1(arg_match=allowed)
    got = run(m, ref, [[c("A", x=1), c("A", x=2)]])
    # Custom matchers get their own hash-suffixed score names, so they never mix.
    tag = m.name[len(m.base_name):]  # "_custom_<hash>"
    assert tag.startswith("_custom_")
    assert got["tool_call_exact" + tag] == 1.0  # a greedy matcher would give 0.5 recall


def test_identity_differs_across_modes_and_matchers():
    ids = [ToolCallF1().identity(), ToolCallF1(arg_mode="subset").identity(),
           ToolCallF1(arg_mode="name").identity()]
    assert len({json.dumps(i, sort_keys=True) for i in ids}) == 3
    f1 = lambda n, p, r: p == r  # noqa: E731
    f2 = lambda n, p, r: True  # noqa: E731
    assert ToolCallF1(arg_match=f1).identity() != ToolCallF1(arg_match=f2).identity()
    assert ToolCallF1(arg_match=f1).identity() == ToolCallF1(arg_match=f1).identity()


def test_two_custom_matchers_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    ci = lambda n, p, r: {k: str(v).lower() for k, v in p.items()} == {  # noqa: E731
        k: str(v).lower() for k, v in r.items()}
    strict = lambda n, p, r: p == r  # noqa: E731
    m_ci, m_strict = ToolCallF1(arg_match=ci), ToolCallF1(arg_match=strict)
    assert m_ci.identity() != m_strict.identity() and m_ci.name != m_strict.name
    samples = [Sample(input="q", actual_output="x", expected_tool_calls=[c("g", city="Paris")],
                      actual_trace={"tool_calls": [[c("g", city="paris")]]})]
    r = ak.evaluate(samples, model="precomputed", scorers=[m_ci, m_strict])
    # Distinct headline keys with distinct values: no averaging into one number.
    assert r.headline[m_ci.name] == 1.0 and r.headline[m_strict.name] == 0.0


# -- TrajectoryMatch ----------------------------------------------------------

@pytest.mark.parametrize("pred,strict,in_order", [
    ([[A, B], [C]], 1.0, 1.0),          # identical
    ([[B, A], [C]], 1.0, 1.0),          # order inside a parallel turn is free
    ([[A], [B], [C]], 0.0, 1.0),        # same calls, split differently
    ([[A, B, C]], 0.0, 1.0),            # same calls, all batched
    ([[A, B], [X], [C]], 0.0, 1.0),     # extra call interleaved
    ([[A], [C], [B]], 0.0, 0.0),        # dependent C before the group finished
    ([[C], [A, B]], 0.0, 0.0),          # reversed
    ([[A, B]], 0.0, 0.0),               # missing last turn
    ([], 0.0, 0.0),
])
def test_trajectory(pred, strict, in_order):
    got = run(TrajectoryMatch(), [[A, B], [C]], pred)
    assert got == {"trajectory_strict": strict, "trajectory_in_order": in_order}


def test_trajectory_in_order_with_repeated_calls():
    assert run(TrajectoryMatch(), [[A], [A]], [[A]])["trajectory_in_order"] == 0.0
    assert run(TrajectoryMatch(), [[A], [B], [A]], [[A], [A], [B], [A]])["trajectory_in_order"] == 1.0


def test_trajectory_irrelevance():
    assert run(TrajectoryMatch(), [], [])["trajectory_strict"] == 1.0
    assert run(TrajectoryMatch(), [], [[A]])["trajectory_strict"] == 0.0


def test_trajectory_name_mode_suffix():
    got = run(TrajectoryMatch(arg_mode="name"), [[A]], [[c("A", v=7)]])
    assert got == {"trajectory_strict_name": 1.0, "trajectory_in_order_name": 1.0}


# -- ParallelToolCalls --------------------------------------------------------

def test_parallel_correct():
    got = run(ParallelToolCalls(), [[A, B], [C]], [[B, A], [C]])
    assert got == {"parallel_recall": 1.0, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_serialized_independent_calls():
    got = run(ParallelToolCalls(), [[A, B], [C]], [[A], [B], [C]])
    assert got == {"parallel_recall": 0.0, "parallel_detection": 0.0}  # precision: no batched turn


def test_parallel_wrongly_batched_dependent_call():
    got = run(ParallelToolCalls(), [[A, B], [C]], [[A, B, C]])
    assert got == {"parallel_recall": 1.0, "parallel_precision": 0.0, "parallel_detection": 1.0}


def test_parallel_batched_when_reference_is_sequential():
    got = run(ParallelToolCalls(), [[A], [B]], [[A, B]])
    assert got == {"parallel_precision": 0.0, "parallel_detection": 0.0}  # recall: no group


def test_parallel_partial_recall_and_precision():
    got = run(ParallelToolCalls(), [[A, B], [C, D]], [[A, B], [C], [D]])
    assert got == {"parallel_recall": 0.5, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_nothing_to_measure():
    assert run(ParallelToolCalls(), [[A], [B]], [[A], [B]]) == {"parallel_detection": 1.0}
    assert run(ParallelToolCalls(), [], []) == {"parallel_detection": 1.0}


def test_parallel_turn_with_one_matched_call_is_not_a_batch():
    got = run(ParallelToolCalls(), [[A]], [[A, X]])
    assert got == {"parallel_detection": 0.0}  # only one matched call in the turn


def test_parallel_same_call_in_two_reference_turns():
    # A perfect prediction must score perfectly even when a call repeats
    # across turns (a flat matching can pair turn-1 A with turn-2 A).
    ref = [[c("A", q="a"), c("B", id=1)], [c("A", q="a"), c("B", id=2)]]
    got = run(ParallelToolCalls(), ref, ref)
    assert got == {"parallel_recall": 1.0, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_repeated_group_made_once_counts_once():
    got = run(ParallelToolCalls(), [[A, B], [A, B]], [[A, B]])
    assert got == {"parallel_recall": 0.5, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_two_dependent_groups_in_one_turn():
    # Both groups were made together (recall), but batched with each other (precision).
    got = run(ParallelToolCalls(), [[A, B], [C, D]], [[A, B, C, D]])
    assert got == {"parallel_recall": 1.0, "parallel_precision": 0.0, "parallel_detection": 1.0}


def test_parallel_overlapping_groups_find_their_own_turns():
    got = run(ParallelToolCalls(), [[A, B], [A, B, C]], [[A, B, C], [A, B]])
    assert got == {"parallel_recall": 1.0, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_name_mode_tool_reused_across_turns_with_extra_turn():
    ref = [[c("search", q="a"), c("lookup", id=1)], [c("search", q="b"), c("fetch", id=2)]]
    pred = [[c("ping")], [c("search", q="a"), c("lookup", id=1)], [c("search", q="b"), c("fetch", id=2)]]
    got = run(ParallelToolCalls(arg_mode="name"), ref, pred)
    assert got == {"parallel_recall_name": 1.0, "parallel_precision_name": 1.0,
                   "parallel_detection_name": 1.0}


def test_parallel_recall_exact_search_cross_turn():
    # Monitoring agent re-checks services after a restart. Group [SA,SB] fits both
    # turn 1 and turn 3; a greedy "tightest fit" puts it in turn 3 and strands
    # [SB,LB]. A disjoint assignment (group1->turn1, group2->turn3) exists -> 1.0.
    SA, SB = c("status", svc="A"), c("status", svc="B")
    LB, RB = c("logs", svc="B"), c("restart", svc="B")
    SC, SD = c("status", svc="C"), c("status", svc="D")
    ref = [[SA, SB], [RB], [SB, LB]]
    pred = [[SA, SB, SC, SD], [RB], [SA, SB, LB]]
    assert run(ParallelToolCalls(), ref, pred)["parallel_recall"] == 1.0


def test_parallel_recall_exact_search_within_turn_subset():
    # Both groups matchable into the single predicted turn; a greedy Kuhn pick of
    # the lang=it call for group 1 would strand group 2. subset mode -> 1.0.
    ref = [[c("search", q="rome"), c("ping")],
           [c("search", q="rome", lang="it"), c("ping")]]
    pred = [[c("search", q="rome", lang="it"), c("search", q="rome", lang="en"),
             c("ping"), c("ping")]]
    assert run(ParallelToolCalls(arg_mode="subset"), ref, pred)["parallel_recall_subset"] == 1.0


# -- ToolCallValidity ---------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                       "unit": {"type": "string", "enum": ["C", "F"]}, "detail": {"type": "boolean"},
                       "lat": {"type": "number"}, "tags": {"type": ["array", "null"]}},
        "required": ["city"]}}},
    {"name": "ping", "parameters": {"type": "object", "properties": {}}},  # flat shape
    {"name": "free", "parameters": {}},  # no declared properties
]


def validity(*calls):
    s = Sample(input="q", tools=TOOLS)
    out = ToolCallValidity().score(s, "", {"trace": {"tool_calls": [list(calls)]}})
    return out[0] if out else None


@pytest.mark.parametrize("call,problem", [
    (c("weather", city="Paris"), None),
    (c("weather", city="Paris", days=3, unit="C", detail=False, lat=4, tags=None), None),
    (c("weather", city="Paris", lat=4.5, tags=["x"]), None),
    (c("ping"), None),
    (c("free", anything=1), None),
    (c("nuke", city="Paris"), "unknown tool"),
    (c("weather", days=3), "missing required argument 'city'"),
    (c("weather", city="Paris", country="FR"), "unknown argument 'country'"),
    (c("weather", city="Paris", days=True), "has type bool"),
    (c("weather", city="Paris", detail=1), "has type int"),
    (c("weather", city="Paris", lat=True), "has type bool"),
    (c("weather", city="Paris", days="3"), "has type str"),
    (c("weather", city="Paris", unit="K"), "not in enum"),
    ({"function": {"name": "weather", "arguments": '{"city": "Par'}}, "not valid JSON"),
])
def test_validity_single_call(call, problem):
    s = validity(call)
    if problem is None:
        assert s.value == 1.0 and s.reason is None
    else:
        assert s.value == 0.0 and problem in s.reason


def test_validity_fraction_and_metadata():
    s = validity(c("weather", city="Paris"), c("nuke"), c("ping"), c("weather"))
    assert s.value == pytest.approx(0.5)
    assert s.metadata == {"n_calls": 4, "n_invalid": 2}
    assert s.kind == ScoreKind.AGENT


def test_validity_skips_samples_without_calls_or_tools():
    assert validity() is None
    assert not ToolCallValidity().applicable(Sample(input="q"))
    # tools=[] offered: every call is to an unknown tool
    s = Sample(input="q", tools=[])
    assert ToolCallValidity().score(s, "", {"trace": {"tool_calls": [c("a")]}})[0].value == 0.0


# -- RedundantToolCalls -------------------------------------------------------

def test_redundant_calls():
    m = RedundantToolCalls()
    assert m.direction == Direction.MINIMIZE
    s = Sample(input="q")
    out = m.score(s, "", {"trace": {"tool_calls": [[A], [A], [B]]}})
    assert out[0].value == pytest.approx(1 / 3) and out[0].metadata["n_duplicates"] == 1
    # argument key order doesn't hide a repeat; a different value is not a repeat
    same = [[c("f", a=1, b=2)], [{"name": "f", "arguments": {"b": 2, "a": 1}}], [c("f", a=1, b=3)]]
    assert m.score(s, "", {"trace": {"tool_calls": same}})[0].value == pytest.approx(1 / 3)
    assert m.score(s, "", {"trace": {"tool_calls": [[A, B]]}})[0].value == 0.0
    assert m.score(s, "no calls here", None) == []


def test_redundant_calls_ignores_malformed_calls():
    # Two truncated (unparseable) calls with different argument text are not repeats:
    # both have arguments={} so keying on them would falsely count a duplicate.
    m = RedundantToolCalls()
    text = ('<tool_call>{"name": "search", "arguments": {"q": "a"</tool_call>'
            '<tool_call>{"name": "search", "arguments": {"q": "b"</tool_call>')
    out = m.score(Sample(input="q"), text, None)
    assert out[0].value == 0.0 and out[0].metadata["n_calls"] == 2


# -- TaskCompletion -----------------------------------------------------------

class Recorder:
    def __init__(self, reply):
        self.prompts: list[str] = []
        self.reply = reply

    def __call__(self, prompts):
        self.prompts.extend(prompts)
        return [self.reply for _ in prompts]


def test_task_completion_prompt_shows_the_trajectory():
    rec = Recorder("The agent booked it.\nCHOICE: complete")
    m = TaskCompletion(judge_model=CallableModel(rec))
    s = Sample(input="Book the cheapest flight to Rome", target="flight AZ123 booked")
    trace = {"tool_calls": [[c("search_flights", dest="Rome")], [c("book", flight="AZ123")]]}
    score = m.score(s, "Booked AZ123.", {"trace": trace})
    assert score.value == 1.0 and score.kind == ScoreKind.AGENT and score.name == "task_completion"
    (prompt,) = rec.prompts
    assert "search_flights" in prompt and '"dest": "Rome"' in prompt
    assert "book" in prompt and '"flight": "AZ123"' in prompt
    assert "[turn 2]" in prompt and "{trajectory}" not in prompt
    assert "Booked AZ123." in prompt and "Book the cheapest flight to Rome" in prompt
    assert "flight AZ123 booked" in prompt


def test_task_completion_renders_transcript_with_tool_results():
    rec = Recorder("CHOICE: partial")
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "lookup", "arguments": '{"id": 7}'}}]},
            {"role": "tool", "content": "RESULT-XYZ"}]
    score = TaskCompletion(judge_model=CallableModel(rec)).score(Sample(input="q"), "done", {"trace": {"messages": msgs}})
    assert score.value == 0.5
    assert "RESULT-XYZ" in rec.prompts[0] and "lookup" in rec.prompts[0]


def test_task_completion_no_calls_and_failed():
    rec = Recorder("reason\nCHOICE: failed")
    score = TaskCompletion(judge_model=CallableModel(rec)).score(Sample(input="q"), "I did it!", None)
    assert score.value == 0.0 and "(no tool calls)" in rec.prompts[0]


def test_task_completion_identity_tracks_judge():
    a = TaskCompletion(judge_model="openai:gpt-4o-mini").identity()
    b = TaskCompletion(judge_model="openai:gpt-4o").identity()
    assert a != b and "{trajectory}" in a["prompt"]


# -- import hygiene -----------------------------------------------------------

def test_import_emits_no_warnings():
    r = subprocess.run([sys.executable, "-W", "error", "-c", "import auditkit, auditkit.trace, "
                        "auditkit.metrics.agent, auditkit.metrics.retrieval, auditkit.metrics.rag_judge"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- end to end: model="precomputed" ------------------------------------------

def _agent_samples():
    return [
        Sample(input="weather in Paris and Rome", id="s1", tools=TOOLS,
               expected_tool_calls=[[c("weather", city="Paris"), c("weather", city="Rome")]],
               actual_output="Paris sunny, Rome rain.",
               actual_trace={"tool_calls": [[c("weather", city="Rome"), c("weather", city="Paris")]]}),
        Sample(input="weather in Oslo then Bergen", id="s2", tools=TOOLS,
               expected_tool_calls=[[c("weather", city="Oslo")], [c("weather", city="Bergen")]],
               actual_output="done",
               actual_trace={"tool_calls": [[c("weather", city="Oslo"), c("weather", city="Oslo"),
                                             c("weather", city="Bergen", country="NO")]]}),
    ]


def test_evaluate_precomputed_agent_traces(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    r = ak.evaluate(_agent_samples(), model="precomputed",
                    scorers=[ToolCallF1(), TrajectoryMatch(), ParallelToolCalls(), ToolCallValidity(),
                             RedundantToolCalls()])
    assert r.errors == []
    h = r.headline
    # s1 perfect; s2: 1 of 3 preds matched (Oslo), 1 of 2 refs.
    assert h["tool_call_precision"] == pytest.approx((1 + 1 / 3) / 2)
    assert h["tool_call_recall"] == pytest.approx((1 + 1 / 2) / 2)
    assert h["tool_call_exact"] == 0.5
    assert h["trajectory_strict"] == 0.5 and h["trajectory_in_order"] == 0.5
    assert h["parallel_recall"] == 1.0 and r.stats["parallel_recall"].count == 1
    assert h["parallel_precision"] == 1.0 and r.stats["parallel_precision"].count == 1  # s2 has one matched call
    assert h["parallel_detection"] == 0.5
    assert h["tool_call_validity"] == pytest.approx((1 + 2 / 3) / 2)
    assert h["redundant_tool_calls"] == pytest.approx((0 + 1 / 3) / 2)


def test_evaluate_precomputed_text_only_and_registry_names(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = [Sample(input="q", expected_tool_calls=[A],
                      actual_output='<tool_call>{"name": "A", "arguments": {"v": 1}}</tool_call>'),
               Sample(input="q2", expected_tool_calls=[], actual_output="No tool needed.")]
    r = ak.evaluate(samples, model="precomputed", scorers=["tool_call_f1", "trajectory_match"])
    assert r.headline["tool_call_f1"] == 1.0 and r.headline["trajectory_strict"] == 1.0


def test_evaluate_task_completion_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rec = Recorder("ok\nCHOICE: complete")
    r = ak.evaluate(_agent_samples()[:1], model="precomputed",
                    scorers=[TaskCompletion(judge_model=CallableModel(rec))])
    assert r.headline["task_completion"] == 1.0
    assert '"city": "Rome"' in rec.prompts[0]


# -- fingerprints -------------------------------------------------------------

def test_fingerprint_changes_with_expected_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    mk = lambda exp: [Sample(input="q", actual_output="x", expected_tool_calls=exp,  # noqa: E731
                             actual_trace={"tool_calls": [[A]]})]
    assert ListScenario(mk([[A]])).name != ListScenario(mk([[B]])).name
    r1 = ak.evaluate(mk([[A]]), model="precomputed", scorers=[ToolCallF1()])
    r2 = ak.evaluate(mk([[B]]), model="precomputed", scorers=[ToolCallF1()])
    assert r1.fingerprint != r2.fingerprint
    assert r1.headline["tool_call_f1"] == 1.0 and r2.headline["tool_call_f1"] == 0.0


@pytest.mark.parametrize("field,a,b", [
    ("tools", [{"name": "a"}], [{"name": "b"}]),
    ("reference_contexts", ["d1"], ["d2"]),
    ("actual_trace", {"tool_calls": [[A]]}, {"tool_calls": [[B]]}),
    ("metadata", {"k": 1}, {"k": 2}),
])
def test_fingerprint_tracks_each_new_field(field, a, b):
    s1, s2 = Sample(input="q"), Sample(input="q")
    setattr(s1, field, a)
    setattr(s2, field, b)
    assert ListScenario([s1]).name != ListScenario([s2]).name


def test_old_style_dataset_fingerprint_unchanged():
    samples = [Sample(input="2+2?", target="4"), Sample(input="capital?", target="Paris", choices=["Paris", "Rome"]),
               Sample(input="q", retrieval_context=["ctx"], actual_output="a")]
    legacy = hashlib.sha256(json.dumps(
        [(s.input, s.target, s.choices, s.retrieval_context, s.actual_output) for s in samples],
        default=str, sort_keys=True).encode()).hexdigest()[:8]
    assert ListScenario(samples).name == f"inline_3_{legacy}"


# -- follow-up fixes (lead review) ---------------------------------------------

def test_validity_reads_anthropic_schema_and_empty_properties():
    from auditkit.metrics.agent import _tool_schemas, validate_call
    from auditkit.trace import ToolCall
    schemas = _tool_schemas([
        {"name": "search", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
        {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}},
    ])
    assert validate_call(ToolCall("search", {"q": "x"}), schemas) == []
    assert validate_call(ToolCall("search", {}), schemas) == ["missing required argument 'q'"]
    assert validate_call(ToolCall("ping", {"extra": 1}), schemas) == ["unknown argument 'extra'"]


def test_validity_boolean_subschema_does_not_crash():
    from auditkit.metrics.agent import _tool_schemas, validate_call
    from auditkit.trace import ToolCall
    # JSON Schema draft 6+: a property schema (or the whole `parameters`) may be
    # a boolean. `true` accepts anything; it must not raise AttributeError.
    schemas = _tool_schemas([{"name": "f", "parameters": {"type": "object", "properties": {"x": True}}}])
    assert validate_call(ToolCall("f", {"x": 123}), schemas) == []
    schemas2 = _tool_schemas([{"name": "g", "parameters": True}])
    assert validate_call(ToolCall("g", {"a": 1}), schemas2) == []


def test_task_completion_metadata_trajectory_does_not_hide_transcript():
    # A dataset column named 'trajectory' in metadata must not replace the real run.
    rec = Recorder("CHOICE: complete")
    s = Sample(input="task", metadata={"trajectory": "gold: search then book"})
    trace = {"tool_calls": [[c("search_flights", dest="Rome")]]}
    TaskCompletion(judge_model=CallableModel(rec)).score(s, "done", {"trace": trace})
    assert "search_flights" in rec.prompts[0]
    assert "gold: search then book" not in rec.prompts[0]


# -- edge cases: model-controlled / malformed input must not hang or crash -----

def test_parallel_recall_bounded_on_many_groups():
    import time
    # ~25 reference parallel groups, all placeable into one predicted turn, made the
    # group->turn backtracking exponential (~minutes, uncatchable during scoring).
    # The greedy seed meets the upper bound, so it is exact without any search.
    G = 25
    exp = [[c("a%d" % i), c("b%d" % i)] for i in range(G)]
    allc = [call for g in exp for call in g]
    t0 = time.perf_counter()
    out = ParallelToolCalls().score(Sample(input="q", expected_tool_calls=exp), "",
                                    {"trace": {"tool_calls": [allc]}})
    dt = time.perf_counter() - t0
    rec = next(s for s in out if s.name == "parallel_recall")
    assert dt < 3.0
    assert rec.value == 1.0
    out2 = ParallelToolCalls().score(Sample(input="q", expected_tool_calls=[[A, B], [C, D]]), "",
                                     {"trace": {"tool_calls": [[A, B, C, D]]}})
    rec2 = next(s for s in out2 if s.name == "parallel_recall")
    assert rec2.value == 1.0


@pytest.mark.parametrize("G", [15, 20, 100, 1000, 2000])
def test_parallel_recall_perfect_prediction_is_one(G):
    import time
    # A work budget cut the search after ~14 groups and reported 14/G for a
    # perfect prediction (0.7 headline at G=20, no error); at G>=1000 the
    # recursive search raised RecursionError and dropped all parallel_* scores.
    exp = [[c("a%d" % i), c("b%d" % i)] for i in range(G)]
    t0 = time.perf_counter()
    got = run(ParallelToolCalls(), exp, exp)
    assert time.perf_counter() - t0 < 5.0
    assert got == {"parallel_recall": 1.0, "parallel_precision": 1.0, "parallel_detection": 1.0}


def test_parallel_recall_identical_groups_exact_optimum():
    # G = P+1 identical [f, f] groups, P turns of [f, f, f]: each turn holds one
    # group, so the optimum is P/G (the budgeted search reported 0.087 at P=160).
    P = 60
    got = run(ParallelToolCalls(), [[c("f"), c("f")] for _ in range(P + 1)], [[c("f")] * 3 for _ in range(P)])
    assert got["parallel_recall"] == pytest.approx(P / (P + 1))


def test_parallel_recall_search_beats_greedy_and_budget_omits_not_guesses(monkeypatch):
    import auditkit.metrics.agent as agent_mod
    # Greedy puts group [f, g] in turn 0 (its first fit) and strands [f, h],
    # which fits only turn 0; the exact search moves [f, g] to turn 1.
    F, G_, H = c("f", v=1), c("g", v=2), c("h", v=3)
    ref, pred = [[F, G_], [F, H]], [[F, G_, H], [F, G_]]
    assert run(ParallelToolCalls(), ref, pred)["parallel_recall"] == 1.0
    # Out of budget: no parallel_recall at all (never an unproven number), and the
    # detection score says why; the other scores are still there.
    monkeypatch.setattr(agent_mod, "_PACK_BUDGET", 0)
    out = ParallelToolCalls().score(Sample(input="q", expected_tool_calls=ref), "", {"trace": {"tool_calls": pred}})
    by = {s.name: s for s in out}
    assert "parallel_recall" not in by
    assert by["parallel_detection"].value == 1.0 and "parallel_recall_omitted" in by["parallel_detection"].metadata
    assert "parallel_precision" in by


def test_trajectory_in_order_partly_covered_turn_is_fast():
    import time
    # A reference turn of 2n identical calls, n made, then n unrelated calls: every
    # unrelated pred re-ran a failing augment from each uncovered ref (~n^4, 10s at n=160).
    n = 400
    t0 = time.perf_counter()
    assert run(TrajectoryMatch(), [[c("f")] * (2 * n)],
               [[c("f")] * n + [c("x", i=i) for i in range(n)]])["trajectory_in_order"] == 0.0
    # K=200 distinct refs, 100 covered, then 2000 unrelated calls (~31s before)
    refs = [[c("read", path=str(i)) for i in range(200)]]
    preds = [[c("read", path=str(i)) for i in range(100)] + [c("ls", i=i) for i in range(2000)]]
    assert run(TrajectoryMatch(arg_mode="name"), refs, preds)["trajectory_in_order_name"] == 0.0
    assert time.perf_counter() - t0 < 2.0
    # results unchanged: a cover completed late, across a gap, still counts
    preds = [[c("f")] * n + [c("x", i=i) for i in range(n)] + [c("f")] * n]
    assert run(TrajectoryMatch(), [[c("f")] * (2 * n)], preds)["trajectory_in_order"] == 1.0


def test_degenerate_output_scores_as_failed_call_end_to_end():
    # '[' * 1000 raised RecursionError inside the metric; the Runner dropped the
    # sample, so the headline rose from 0.5 to 1.0 with failed_count 0.
    ref = [[c("f", x=1)]]
    good = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    samples = [Sample(input="q1", expected_tool_calls=ref, actual_output=good),
               Sample(input="q2", expected_tool_calls=ref, actual_output="[" * 1000)]
    r = ak.evaluate(samples, "precomputed", [ToolCallF1(), RedundantToolCalls()])
    assert r.errors == []
    assert r.headline["tool_call_f1"] == 0.5 and r.stats["tool_call_f1"].count == 2


def test_saved_run_is_strict_json_with_nan_and_inf_arguments(tmp_path):
    # 1e999 / NaN in model tool-call text ended up as bare Infinity / NaN tokens
    # in run.json (score metadata), which strict JSON parsers reject.
    out = '<tool_call>{"name": "f", "arguments": {"x": 1e999, "y": NaN}}</tool_call>'
    r = ak.evaluate([Sample(input="q", expected_tool_calls=[[c("f", x=1)]], actual_output=out)],
                    "precomputed", [ToolCallF1()])
    p = tmp_path / "run.json"
    r.save(str(p))

    def reject(tok):
        raise ValueError(f"non-strict JSON token {tok}")
    json.loads(p.read_text(), parse_constant=reject)


def test_nan_arguments_match_in_f1_consistent_with_redundancy():
    # F1 said a NaN call never equals itself (0.0) while RedundantToolCalls
    # counted two of them as duplicates.
    text = '<tool_call>{"name": "f", "arguments": {"x": NaN}}</tool_call>'
    ref = [[{"name": "f", "arguments": {"x": float("nan")}}]]
    assert run(ToolCallF1(), ref, output=text)["tool_call_f1"] == 1.0
    assert run(TrajectoryMatch(), ref, output=text)["trajectory_strict"] == 1.0
    assert run(RedundantToolCalls(), [], output=text + text)["redundant_tool_calls"] == 0.5


def test_trajectory_in_order_fast_on_many_predicted_calls():
    import time
    # A runaway agent emitting thousands of calls where a reference turn never
    # matches was O(preds^2). The incremental single-pass matching is linear.
    t0 = time.perf_counter()
    got = run(TrajectoryMatch(), [[c("never")]], [[c("x") for _ in range(20000)]])
    assert time.perf_counter() - t0 < 1.0
    assert got["trajectory_in_order"] == 0.0
    # a match far down the predicted list is still found (identical result, fast)
    preds = [[c("x") for _ in range(5000)] + [c("A", v=1)]]
    assert run(TrajectoryMatch(), [[A]], preds)["trajectory_in_order"] == 1.0


# -- reviewer findings 5-10 ----------------------------------------------------

def test_reference_metrics_skip_when_tool_coverage_unavailable():
    # An agent: reply with no transcript and no explicit tool_calls marks the
    # trace tool_calls_unavailable; the reference tool metrics are then
    # ineligible (no score), not scored as observed-zero-calls (finding 5).
    ctx = {"trace": {"tool_calls_unavailable": True}}
    for m in (ToolCallF1(), TrajectoryMatch(), ParallelToolCalls()):
        assert m.score(Sample(input="q", expected_tool_calls=[]), "", ctx) == []
        assert m.score(Sample(input="q", expected_tool_calls=[[A]]), "", ctx) == []
    # an explicit empty turn list is still observed-zero (scoreable)
    assert run(ToolCallF1(), [], [])["tool_call_f1"] == 1.0


def test_task_completion_unparseable_verdict_raises():
    # An unreadable verdict is an error, not a silent 0.0 (finding 6).
    rec = Recorder("I really cannot tell either way.")
    with pytest.raises(ValueError, match="no parseable verdict"):
        TaskCompletion(judge_model=CallableModel(rec)).score(Sample(input="q"), "did it", None)


def test_task_completion_unparseable_is_recorded_error_not_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rec = Recorder("hmm, unclear")
    r = ak.evaluate(_agent_samples()[:1], model="precomputed",
                    scorers=[TaskCompletion(judge_model=CallableModel(rec))])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "task_completion"
    assert "task_completion" not in r.headline and "task_completion" not in r.stats


def test_validity_recurses_into_arrays_and_nested_objects():
    from auditkit.metrics.agent import _tool_schemas, validate_call
    from auditkit.trace import ToolCall
    schemas = _tool_schemas([{"name": "meet", "parameters": {"type": "object", "properties": {
        "attendees": {"type": "array", "items": {"type": "string"}},
        "reminder": {"type": "object", "properties": {"minutes_before": {"type": "integer"}},
                     "required": ["minutes_before"]}}, "required": ["attendees"]}}])
    assert validate_call(ToolCall("meet", {"attendees": ["a", "b"], "reminder": {"minutes_before": 5}}), schemas) == []
    assert validate_call(ToolCall("meet", {"attendees": ["a", 3]}), schemas) == \
        ["argument 'attendees[1]' has type int, schema says string"]
    assert validate_call(ToolCall("meet", {"attendees": ["a"], "reminder": {"minutes_before": "soon"}}), schemas) == \
        ["argument 'reminder.minutes_before' has type str, schema says integer"]
    assert validate_call(ToolCall("meet", {"attendees": ["a"], "reminder": {"minutes_before": 5, "snooze": 1}}),
                         schemas) == ["unknown argument 'reminder.snooze'"]
    assert validate_call(ToolCall("meet", {"attendees": ["a"], "reminder": {}}), schemas) == \
        ["missing required argument 'reminder.minutes_before'"]


def test_trajectory_in_order_empty_reference_with_a_call_is_not_perfect():
    # Empty reference + any predicted call is not a perfect in-order match
    # (finding 8); strict/f1 already give 0 here.
    assert run(TrajectoryMatch(), [], [[A]])["trajectory_in_order"] == 0.0
    assert run(TrajectoryMatch(), [], [])["trajectory_in_order"] == 1.0


def test_redundant_and_f1_agree_on_int_float_equality():
    # f(a=1) and f(a=1.0) are one call for F1 (JSON-value equality); redundancy
    # must agree and count the repeat (finding 9).
    s = Sample(input="q")
    assert run(ToolCallF1(), [[c("f", a=1)]], [[c("f", a=1.0)]])["tool_call_f1"] == 1.0
    out = RedundantToolCalls().score(s, "", {"trace": {"tool_calls": [[c("f", a=1)], [c("f", a=1.0)]]}})
    assert out[0].value == 0.5 and out[0].metadata["n_duplicates"] == 1
    # a genuinely different value is still not a repeat
    assert RedundantToolCalls().score(
        s, "", {"trace": {"tool_calls": [[c("f", a=1)], [c("f", a=2)]]}})[0].value == 0.0
    # a whole-number float never collapses two distinct large integers
    big = 2 ** 60 + 1
    assert RedundantToolCalls().score(
        s, "", {"trace": {"tool_calls": [[c("f", a=2 ** 60)], [c("f", a=big)]]}})[0].value == 0.0


def test_f1_exact_tolerates_schema_declared_optional_arg():
    # BFCL: a schema-declared optional the reference omits does not break the
    # exact match; an undeclared extra still does (finding 10).
    tools = [{"type": "function", "function": {"name": "w", "parameters": {"type": "object",
             "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
             "required": ["city"]}}}]
    assert run(ToolCallF1(), [[c("w", city="Paris")]], [[c("w", city="Paris", unit="celsius")]],
               tools=tools)["tool_call_f1"] == 1.0
    assert run(ToolCallF1(), [[c("w", city="Paris")]], [[c("w", city="Paris", bogus=1)]],
               tools=tools)["tool_call_f1"] == 0.0
    # trajectory_strict also uses the effective matcher
    assert run(TrajectoryMatch(), [[c("w", city="Paris")]], [[c("w", city="Paris", unit="celsius")]],
               tools=tools)["trajectory_strict"] == 1.0
    # with no schemas available, exact stays strict (any extra breaks the match)
    assert run(ToolCallF1(), [[c("w", city="Paris")]], [[c("w", city="Paris", unit="celsius")]])["tool_call_f1"] == 0.0


def test_redundant_calls_with_mixed_type_argument_keys():
    # Native-object args (not a decoded JSON string) can carry non-str keys;
    # json.dumps(sort_keys=True) raised TypeError comparing str and int. The
    # str-coercing round-trip tolerates any keys while keeping order-free dedup.
    m = RedundantToolCalls()
    s = Sample(input="q")
    out = m.score(s, "", {"trace": {"tool_calls": [{"name": "f", "arguments": {1: "a", "b": 2}}]}})
    assert out[0].value == 0.0 and out[0].metadata["n_calls"] == 1
    # reordered mixed-key dicts still dedup as one repeat
    tc = [[{"name": "f", "arguments": {1: "a", "b": 2}}], [{"name": "f", "arguments": {"b": 2, 1: "a"}}]]
    assert m.score(s, "", {"trace": {"tool_calls": tc}})[0].value == 0.5
