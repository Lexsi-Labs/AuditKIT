"""Every agentic metric against hand-computed expectations (no model, no network).

Each row is (id, reference, predicted turns, expected scores). A score name
that is absent from ``expected`` must also be absent from the output, so the
"omitted when there is nothing to measure" rule is tested too.
"""

from __future__ import annotations

import pytest

from auditkit.metrics.agent import (
    ParallelToolCalls, RedundantToolCalls, TaskCompletion, ToolCallF1, ToolCallValidity, TrajectoryMatch,
)
from auditkit.model import CallableModel
from auditkit.sample import Sample

from .catalog import BOOK, SEARCH, STOCK, TOOLS, T, W, c


def scores(metric, expected, predicted=None, output="", tools=None, target=None):
    s = Sample(input="task", expected_tool_calls=expected, tools=tools, target=target)
    ctx = {"trace": {"tool_calls": predicted}} if predicted is not None else {}
    out = metric.score(s, output, ctx)
    return {x.name: x.value for x in (out if isinstance(out, list) else [out])}


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def prf(p, r, exact, sfx=""):
    return {f"tool_call_precision{sfx}": p, f"tool_call_recall{sfx}": r,
            f"tool_call_f1{sfx}": f1(p, r), f"tool_call_exact{sfx}": exact}


ALL0, ALL1 = prf(0, 0, 0), prf(1, 1, 1)
P, R = W("Paris"), W("Rome")
REF2 = [[P, R], [BOOK]]  # parallel group, then a dependent call

# -- ToolCallF1 -------------------------------------------------------------------

F1_CASES = [
    ("perfect-single", [[P]], [[P]], ALL1),
    ("wrong-argument-value", [[P]], [[W("Lyon")]], ALL0),
    ("wrong-tool-name", [[P]], [[T("Paris")]], ALL0),
    ("missing-one-of-parallel", [[P, R]], [[P]], prf(1, .5, 0)),
    ("extra-unrequested-call", [[P]], [[P, T("Paris")]], prf(.5, 1, 0)),
    ("duplicate-call-multiset", [[P]], [[P, P]], prf(.5, 1, 0)),
    ("duplicate-in-reference", [[P], [P]], [[P]], prf(1, .5, 0)),
    ("order-inside-turn-ignored", [[P, R]], [[R, P]], ALL1),
    ("turn-boundaries-ignored", [[P, R]], [[P], [R]], ALL1),
    ("multi-turn-all-correct", REF2, [[P, R], [BOOK]], ALL1),
    ("irrelevance-no-call", [], [], ALL1),
    ("irrelevance-call-made", [], [[P]], ALL0),
    ("no-prediction", [[P]], [], ALL0),
    ("int-equals-float", [[c("convert_currency", amount=100, from_currency="USD", to_currency="EUR")]],
     [[c("convert_currency", amount=100.0, from_currency="USD", to_currency="EUR")]], ALL1),
    ("bool-not-equal-int", [[c("set_alarm", time="07:00", enabled=True)]],
     [[c("set_alarm", time="07:00", enabled=1)]], ALL0),
    ("string-not-equal-number", [[c("convert_currency", amount=100, from_currency="USD", to_currency="EUR")]],
     [[c("convert_currency", amount="100", from_currency="USD", to_currency="EUR")]], ALL0),
    ("case-sensitive-strings", [[P]], [[W("paris")]], ALL0),
    ("whitespace-sensitive", [[P]], [[W("Paris ")]], ALL0),
    ("key-order-irrelevant", [[c("f", a=1, b=2)]], [[{"name": "f", "arguments": {"b": 2, "a": 1}}]], ALL1),
    ("nested-dict-key-order", [[c("f", o={"x": 1, "y": [1, 2]})]], [[c("f", o={"y": [1, 2], "x": 1})]], ALL1),
    ("nested-list-order-matters", [[c("f", l=[1, 2])]], [[c("f", l=[2, 1])]], ALL0),
    ("unicode-argument", [[W("São Paulo")]], [[W("São Paulo")]], ALL1),
    ("empty-arguments", [[c("ping")]], [[c("ping")]], ALL1),
    ("extra-optional-arg-exact", [[P]], [[W("Paris", unit="celsius")]], ALL0),
    ("malformed-json-arguments", [[P]],
     [[{"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": "{bad json"}}]], ALL0),
    ("arguments-json-string", [[P]],
     [[{"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]],
     ALL1),
    ("args-not-an-object", [[P]], [[{"name": "get_weather", "arguments": "[1, 2]"}]], ALL0),
    ("partial-3-of-4", [[W("A"), W("B"), W("C"), W("D")]], [[W("A"), W("B"), W("C"), W("X")]], prf(.75, .75, 0)),
]


@pytest.mark.parametrize("ref,pred,want", [x[1:] for x in F1_CASES], ids=[x[0] for x in F1_CASES])
def test_tool_call_f1(ref, pred, want):
    assert scores(ToolCallF1(), ref, pred) == pytest.approx(want)


def test_f1_subset_mode():
    m = ToolCallF1(arg_mode="subset")
    assert scores(m, [[P]], [[W("Paris", unit="celsius")]]) == pytest.approx(prf(1, 1, 1, "_subset"))
    # subset still requires every reference argument
    assert scores(m, [[W("Paris", unit="celsius")]], [[P]]) == pytest.approx(prf(0, 0, 0, "_subset"))


def test_f1_name_mode_ignores_arguments_but_not_names():
    m = ToolCallF1(arg_mode="name")
    assert scores(m, [[P]], [[W("Lyon")]]) == pytest.approx(prf(1, 1, 1, "_name"))
    assert scores(m, [[P]], [[T("Paris")]]) == pytest.approx(prf(0, 0, 0, "_name"))


def test_f1_custom_matcher_casefold_and_tolerance():
    def loose(name, pred, ref):
        if name == "convert_currency":
            return abs(pred.get("amount", 0) - ref["amount"]) <= 0.01 and pred.get("to_currency") == ref["to_currency"]
        return {k: str(v).strip().casefold() for k, v in pred.items()} == \
               {k: str(v).strip().casefold() for k, v in ref.items()}

    m = ToolCallF1(arg_match=loose)
    got = scores(m, [[P, c("convert_currency", amount=10, from_currency="USD", to_currency="EUR")]],
                 [[W(" paris "), c("convert_currency", amount=10.004, from_currency="usd", to_currency="EUR")]])
    assert set(got.values()) == {1.0}
    assert all(k.startswith("tool_call_") and "_custom_" in k for k in got)


def test_f1_bipartite_beats_greedy():
    """A predicted call compatible with two references must go where it is needed.

    Greedy first-fit pairs ref0 (a=1) with pred0 (a=1,b=2) under subset, leaving
    ref1 (a=1,b=2) with nothing -> recall 0.5. The exact matching finds 2/2.
    """
    ref = [[c("s", a=1), c("s", a=1, b=2)]]
    pred = [[c("s", a=1, b=2), c("s", a=1)]]
    assert scores(ToolCallF1(arg_mode="subset"), ref, pred)["tool_call_recall_subset"] == 1.0


def test_f1_metadata_counts():
    s = Sample(input="t", expected_tool_calls=[[P, R]])
    out = ToolCallF1().score(s, "", {"trace": {"tool_calls": [[P, W("Lyon"), T("Paris")]]}})
    meta = out[0].metadata
    assert (meta["n_expected"], meta["n_predicted"], meta["n_matched"]) == (2, 3, 1)


def test_f1_not_applicable_without_reference():
    assert not ToolCallF1().applicable(Sample(input="q"))
    assert ToolCallF1().applicable(Sample(input="q", expected_tool_calls=[]))


def test_invalid_arg_mode_rejected():
    with pytest.raises(ValueError):
        ToolCallF1(arg_mode="fuzzy")


# -- TrajectoryMatch --------------------------------------------------------------

def traj(strict, in_order):
    return {"trajectory_strict": strict, "trajectory_in_order": in_order}


TRAJ_CASES = [
    ("identical", REF2, [[P, R], [BOOK]], traj(1, 1)),
    ("order-inside-turn", REF2, [[R, P], [BOOK]], traj(1, 1)),
    ("serialized-parallel-group", REF2, [[P], [R], [BOOK]], traj(0, 1)),
    ("everything-batched", REF2, [[P, R, BOOK]], traj(0, 1)),
    ("turns-reversed", REF2, [[BOOK], [P, R]], traj(0, 0)),
    ("extra-call-between", REF2, [[P, R], [T("Rome")], [BOOK]], traj(0, 1)),
    ("final-call-missing", REF2, [[P, R]], traj(0, 0)),
    ("wrong-arg-in-last-turn", REF2, [[P, R], [c("book_flight", flight_id="XX")]], traj(0, 0)),
    ("retry-then-correct", [[SEARCH], [BOOK]], [[SEARCH], [SEARCH], [BOOK]], traj(0, 1)),
    ("single-turn", [[P]], [[P]], traj(1, 1)),
    ("irrelevance-no-call", [], [], traj(1, 1)),
    # Fixed (was DESIGN-1): a call on an irrelevance case now fails in_order too.
    ("irrelevance-call-made", [], [[P]], traj(0, 0)),
    ("no-prediction", [[P]], [], traj(0, 0)),
    ("repeated-reference-calls", [[P], [P]], [[P]], traj(0, 0)),
]


@pytest.mark.parametrize("ref,pred,want", [x[1:] for x in TRAJ_CASES], ids=[x[0] for x in TRAJ_CASES])
def test_trajectory_match(ref, pred, want):
    assert scores(TrajectoryMatch(), ref, pred) == want


def test_trajectory_name_mode_suffix():
    got = scores(TrajectoryMatch(arg_mode="name"), REF2, [[W("x"), W("y")], [c("book_flight", flight_id="?")]])
    assert got == {"trajectory_strict_name": 1.0, "trajectory_in_order_name": 1.0}


# -- ParallelToolCalls ------------------------------------------------------------

def par(recall=None, precision=None, detection=None):
    out = {}
    if recall is not None:
        out["parallel_recall"] = recall
    if precision is not None:
        out["parallel_precision"] = precision
    if detection is not None:
        out["parallel_detection"] = detection
    return out


A1, A2, A3 = STOCK("AAPL"), STOCK("MSFT"), STOCK("NVDA")
PAR_CASES = [
    ("correct", REF2, [[P, R], [BOOK]], par(1, 1, 1)),
    ("serialized", REF2, [[P], [R], [BOOK]], par(0, None, 0)),
    ("dependent-call-batched", REF2, [[P, R, BOOK]], par(1, 0, 1)),
    ("mixed-wrong-batch", REF2, [[P], [R, BOOK]], par(0, 0, 1)),
    ("stopped-after-group", REF2, [[P, R]], par(1, 1, 1)),
    ("sequential-ref-batched", [[SEARCH], [BOOK]], [[SEARCH, BOOK]], par(None, 0, 0)),
    ("sequential-ref-sequential", [[SEARCH], [BOOK]], [[SEARCH], [BOOK]], par(None, None, 1)),
    ("single-call", [[P]], [[P]], par(None, None, 1)),
    ("three-way-split-2-1", [[A1, A2, A3]], [[A1, A2], [A3]], par(0, 1, 1)),
    ("three-way-correct", [[A1, A2, A3]], [[A3, A1, A2]], par(1, 1, 1)),
    ("two-groups-merged", [[P, R], [T("Paris"), T("Rome")]], [[P, R, T("Paris"), T("Rome")]], par(1, 0, 1)),
    ("two-groups-correct", [[P, R], [T("Paris"), T("Rome")]], [[P, R], [T("Paris"), T("Rome")]], par(1, 1, 1)),
    ("wrong-arg-inside-group", [[P, R]], [[P, W("Lyon")]], par(0, None, 1)),
    ("irrelevance-no-call", [], [], par(None, None, 1)),
    ("irrelevance-batched-calls", [], [[P, R]], par(None, None, 0)),
    ("group-expected-no-call", [[P, R]], [], par(0, None, 0)),
]


@pytest.mark.parametrize("ref,pred,want", [x[1:] for x in PAR_CASES], ids=[x[0] for x in PAR_CASES])
def test_parallel_tool_calls(ref, pred, want):
    assert scores(ParallelToolCalls(), ref, pred) == pytest.approx(want)


def test_parallel_partial_recall_two_groups():
    got = scores(ParallelToolCalls(), [[P, R], [A1, A2]], [[P, R], [A1], [A2]])
    assert got["parallel_recall"] == .5 and got["parallel_precision"] == 1.0


def test_parallel_recall_metadata():
    s = Sample(input="t", expected_tool_calls=[[P, R], [A1, A2]])
    out = {x.name: x for x in ParallelToolCalls().score(s, "", {"trace": {"tool_calls": [[P, R, A1, A2]]}})}
    assert out["parallel_recall"].metadata["n_groups"] == 2


# -- ToolCallValidity -------------------------------------------------------------

VALID_CASES = [
    ("valid", [[P]], 1.0),
    ("valid-with-enum", [[W("Paris", unit="celsius")]], 1.0),
    ("enum-violation", [[W("Paris", unit="kelvin")]], 0.0),
    ("unknown-tool", [[c("teleport", city="Paris")]], 0.0),
    ("unknown-argument", [[W("Paris", country="FR")]], 0.0),
    ("missing-required", [[c("get_weather", unit="celsius")]], 0.0),
    ("string-for-number", [[c("convert_currency", amount="5", from_currency="A", to_currency="B")]], 0.0),
    ("int-for-number-ok", [[c("convert_currency", amount=5, from_currency="A", to_currency="B")]], 1.0),
    ("float-for-integer", [[c("set_alarm", time="7", repeat_days=2.5)]], 0.0),
    ("bool-for-integer", [[c("set_alarm", time="7", repeat_days=True)]], 0.0),
    ("bool-ok", [[c("set_alarm", time="7", enabled=False)]], 1.0),
    ("array-and-object-ok", [[c("create_event", title="t", date="d", attendees=["a"], reminder={"minutes_before": 5})]], 1.0),
    ("array-given-string", [[c("create_event", title="t", date="d", attendees="a,b")]], 0.0),
    ("malformed-json", [[{"type": "function", "function": {"name": "get_weather", "arguments": "{oops"}}]], 0.0),
    ("half-valid", [[P, c("teleport")]], 0.5),
    ("one-of-three-invalid", [[P, R, W("x", unit="kelvin")]], 2 / 3),
]


@pytest.mark.parametrize("pred,want", [x[1:] for x in VALID_CASES], ids=[x[0] for x in VALID_CASES])
def test_tool_call_validity(pred, want):
    assert scores(ToolCallValidity(), None, pred, tools=TOOLS) == pytest.approx({"tool_call_validity": want})


def test_validity_reason_names_the_problem():
    s = Sample(input="t", tools=TOOLS)
    [out] = ToolCallValidity().score(s, "", {"trace": {"tool_calls": [[W("x", unit="kelvin"), c("teleport")]]}})
    assert "not in enum" in out.reason and "unknown tool" in out.reason
    assert out.metadata == {"n_calls": 2, "n_invalid": 2}


def test_validity_skips_when_no_calls_and_needs_tools():
    assert ToolCallValidity().score(Sample(input="t", tools=TOOLS), "just text", {}) == []
    assert not ToolCallValidity().applicable(Sample(input="t"))


def test_validity_anthropic_schema_and_nullable_type():
    tools = [{"name": "lookup", "input_schema": {"type": "object", "required": ["id"],
                                                 "properties": {"id": {"type": ["string", "null"]}}}}]
    assert scores(ToolCallValidity(), None, [[c("lookup", id=None)]], tools=tools) == {"tool_call_validity": 1.0}
    assert scores(ToolCallValidity(), None, [[c("lookup", id=3)]], tools=tools) == {"tool_call_validity": 0.0}


# -- RedundantToolCalls -----------------------------------------------------------

REDUNDANT_CASES = [
    ("none", [[P, R]], 0.0),
    ("exact-repeat-same-turn", [[P, P]], .5),
    ("same-tool-different-args", [[P, W("Lyon")]], 0.0),
    ("repeat-across-turns", [[P], [P], [P]], 2 / 3),
    ("key-order-still-a-repeat", [[c("f", a=1, b=2)], [{"name": "f", "arguments": {"b": 2, "a": 1}}]], .5),
    ("malformed-calls-not-counted", [[{"name": "", "arguments": "{x"}, {"name": "", "arguments": "{y"}]], 0.0),
]


@pytest.mark.parametrize("pred,want", [x[1:] for x in REDUNDANT_CASES], ids=[x[0] for x in REDUNDANT_CASES])
def test_redundant_tool_calls(pred, want):
    assert scores(RedundantToolCalls(), None, pred) == pytest.approx({"redundant_tool_calls": want})


def test_redundant_uses_same_equality_as_matching():
    """Regression for FINDING-2: 1 and 1.0 are one call to F1 AND a repeat here; True stays distinct from 1."""
    assert scores(ToolCallF1(), [[c("f", a=1)]], [[c("f", a=1.0)]])["tool_call_exact"] == 1.0
    assert scores(RedundantToolCalls(), None, [[c("f", a=1)], [c("f", a=1.0)]]) == {"redundant_tool_calls": .5}
    assert scores(RedundantToolCalls(), None, [[c("f", a=1)], [c("f", a=True)]]) == {"redundant_tool_calls": 0.0}
    assert scores(RedundantToolCalls(), None, [[c("f", o={"x": [2.0]})], [c("f", o={"x": [2]})]]) == \
        {"redundant_tool_calls": .5}


def test_redundant_skips_no_calls():
    assert RedundantToolCalls().score(Sample(input="t"), "text answer", {}) == []


# -- TaskCompletion (scripted judge) ----------------------------------------------

class Judge:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompts):
        self.prompts.extend(prompts)
        return [self.reply for _ in prompts]


@pytest.mark.parametrize("reply,value", [
    ("Booked as asked.\nCHOICE: complete", 1.0),
    ("Searched but never booked.\nCHOICE: partial", 0.5),
    ("Claims success with no actions.\nCHOICE: failed", 0.0),
    ("<think>draft CHOICE: failed</think>\nIt did book.\nCHOICE: complete", 1.0),   # last verdict wins
])
def test_task_completion_verdicts(reply, value):
    j = Judge(reply)
    s = TaskCompletion(judge_model=CallableModel(j)).score(
        Sample(input="Book the cheapest DEL-BOM flight", target="UK-102 booked"), "Booked UK-102.",
        {"trace": {"tool_calls": [[SEARCH], [c("book_flight", flight_id="UK-102")]]}})
    assert s.value == value


@pytest.mark.parametrize("reply", ["I am not sure what to say.", ""])
def test_task_completion_unreadable_verdict_raises(reply):
    """Regression for FINDING-3: no verdict -> error (Runner records it), never a 0.0."""
    with pytest.raises(ValueError):
        TaskCompletion(judge_model=CallableModel(Judge(reply))).score(Sample(input="q"), "done", None)


def test_task_completion_prompt_carries_everything():
    j = Judge("CHOICE: complete")
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "search_flights",
                                                                         "arguments": '{"origin": "DEL"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "UK-102 $80; AI-5 $120"}]
    TaskCompletion(judge_model=CallableModel(j)).score(
        Sample(input="Book the cheapest flight", target="UK-102"), "Booked UK-102.", {"trace": {"messages": msgs}})
    (prompt,) = j.prompts
    for needle in ("Book the cheapest flight", "UK-102 $80", "search_flights", "Booked UK-102."):
        assert needle in prompt


def test_task_completion_no_calls_is_visible_to_judge():
    j = Judge("CHOICE: failed")
    TaskCompletion(judge_model=CallableModel(j)).score(Sample(input="q"), "done!", None)
    assert "(no tool calls)" in j.prompts[0]
