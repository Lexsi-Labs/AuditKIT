"""Multi-path tool references: ``expected_tool_calls={"any_of": [...]}``.

An agent can reach one correct answer through several tool routes, and a
reference that named only one of them scored the alternatives wrong on every
reference metric. These tests pin the decisions in
``docs/notes/multipath-decisions.md``: which route a run is read against, how
ties break, what a single-route reference may NOT start carrying, that the three
reference metrics read the SAME route, and that an author who names one
alternative has not silently changed the meaning of every existing reference.
"""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.agent_eval import AgentCase, AgentEpisode, episode_from_openai_messages, rescore
from auditkit.agent_eval.runner import eligibility
from auditkit.agent_eval.types import OBSERVED
from auditkit.metrics.agent import (
    MAX_REFERENCE_PATHS, ParallelToolCalls, ToolCallF1, TrajectoryMatch, select_reference_path,
)
from auditkit.sample import Sample
from auditkit.trace import to_reference_paths, to_turns

REFERENCE_METRICS = (ToolCallF1, TrajectoryMatch, ParallelToolCalls)
REFERENCE_SCORERS = ("tool_call_f1", "trajectory_match", "parallel_tool_calls")
PATH_KEYS = ("matched_path", "n_paths", "path_f1", "path_decided_by")


def c(name, **args):
    return {"name": name, "arguments": args}


A, B, C, X, Y = c("A"), c("B"), c("C"), c("X"), c("Y")
D = c("D")
UNRELATED = [c(n) for n in "PQRSTUV"]

# The example from the decisions note: two independent lookups emitted together,
# or the one call that already does both. Same answer, two routes.
PARIS, ROME = c("get_weather", city="Paris"), c("get_weather", city="Rome")
COMPARE = c("compare_weather", cities=["Paris", "Rome"])
TWO_LOOKUPS, ONE_LOOKUP = [[PARIS, ROME]], [[COMPARE]]
EITHER_ROUTE = {"any_of": [TWO_LOOKUPS, ONE_LOOKUP]}

# A run of three calls matches one call of the 3-call route and two of the
# 9-call one, and both routes then score the same call-level F1 -- so only the
# recall tiebreak can separate them.
SHORT_ROUTE = [[A, X, Y]]
LONG_ROUTE = [[A, B] + UNRELATED]


def score_triples(metric, reference, predicted=None, output="", tools=None):
    """[(name, value, metadata)] for one sample: the whole score, metadata included."""
    s = Sample(input="task", expected_tool_calls=reference, tools=tools)
    ctx = {"trace": {"tool_calls": predicted}} if predicted is not None else {}
    out = metric.score(s, output, ctx)
    return [(x.name, x.value, x.metadata) for x in (out if isinstance(out, list) else [out])]


def without_path_keys(scores):
    """{name: (value, metadata)}, with the path keys (PATH_KEYS) dropped.

    Used to line a multi-route score up against the same reference written as a
    single route: what is left has to match, key for key.
    """
    return {name: (value, {k: v for k, v in meta.items() if k not in PATH_KEYS})
            for name, value, meta in scores}


def by_name(metric, reference, predicted=None):
    """{name: (value, metadata)} for one sample."""
    return {name: (value, meta) for name, value, meta in score_triples(metric, reference, predicted)}


def matched_path(metric, reference, predicted):
    """The route the metric read the run against, as reported on its own score."""
    return by_name(metric, reference, predicted)["tool_call_f1"][1]["matched_path"]


def episode(answer, calls, case_id):
    """One recorded run: `calls` as turns, each turn a list of (name, args)."""
    messages = [{"role": "user", "content": "task"}]
    for i, turn in enumerate(calls):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}-{j}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for j, (name, args) in enumerate(turn)]})
        for j in range(len(turn)):
            messages.append({"role": "tool", "tool_call_id": f"c{i}-{j}", "content": "ok"})
    messages.append({"role": "assistant", "content": answer})
    return episode_from_openai_messages(messages, final_answer=answer, case_id=case_id)


def parse_strict(text):
    """Parse as JSON, refusing the NaN/Infinity tokens json.dumps can emit."""
    def reject(token):
        raise ValueError(f"non-strict JSON token {token}")
    return json.loads(text, parse_constant=reject)


# -- the reference shape ------------------------------------------------------

@pytest.mark.parametrize("reference", [
    None,                                    # no reference given
    [],                                      # irrelevance: calling no tool is correct
    [A, B],                                  # a flat call list: one turn
    [[A, B], [], [C]],                       # turns, with an empty one dropped
    [c("f", x=1)],                           # a single call dict
    {"function": {"name": "f", "arguments": '{"x": 1}'}},                    # an OpenAI entry
    [{"role": "assistant", "content": None, "tool_calls": [{"name": "f"}]}],  # a transcript
    {"tool_calls": [[A], [B]]},
    {"messages": [{"role": "assistant", "tool_calls": [{"name": "f"}]}]},
    '[[{"name": "f", "arguments": {"x": 1}}]]',                              # a JSON cell
])
def test_every_single_route_reference_reads_back_as_exactly_one_route(reference):
    # A multi-route reference is only safe if nothing that used to be a
    # reference can start reading as more than one route.
    paths = to_reference_paths(reference)
    assert paths == [to_turns(reference)]
    assert len(paths) == 1


def test_a_reference_reader_cannot_take_a_multi_route_reference_for_a_single_call():
    # The backward-compatibility argument in one place: `to_turns` still refuses
    # {"any_of": ...} (its value is a list, not a call), so no reference shape
    # that used to be valid is now ambiguous.
    with pytest.raises(ValueError, match="cannot read a tool call"):
        to_turns({"any_of": [TWO_LOOKUPS]})


# -- backward compatibility ---------------------------------------------------

# The exact (name, metadata) a single-route reference produced before `any_of`
# existed, written out so that any change to any of them is a failure. Adding
# `matched_path: 0` to these would change the metadata of every run that has
# nothing to do with multi-route references, for no gain (decisions note, Q5).
BEFORE_F1 = [
    ("tool_call_precision", {"n_expected": 3, "n_predicted": 3, "n_matched": 3, "predicted": [A, B, C]}),
    ("tool_call_recall", {}),
    ("tool_call_f1", {}),
    ("tool_call_exact", {}),
]
BEFORE_F1_IRRELEVANCE = [
    ("tool_call_precision", {"n_expected": 0, "n_predicted": 0, "n_matched": 0, "predicted": []}),
    ("tool_call_recall", {}),
    ("tool_call_f1", {}),
    ("tool_call_exact", {}),
]
BEFORE_F1_IRRELEVANCE_CALLED = [
    ("tool_call_precision", {"n_expected": 0, "n_predicted": 1, "n_matched": 0, "predicted": [A]}),
    ("tool_call_recall", {}),
    ("tool_call_f1", {}),
    ("tool_call_exact", {}),
]
BEFORE_TRAJECTORY = [
    ("trajectory_strict", {"n_expected_turns": 2, "n_predicted_turns": 2}),
    ("trajectory_in_order", {}),
]
BEFORE_TRAJECTORY_IRRELEVANCE = [
    ("trajectory_strict", {"n_expected_turns": 0, "n_predicted_turns": 0}),
    ("trajectory_in_order", {}),
]
BEFORE_PARALLEL = [
    ("parallel_recall", {"n_groups": 1}),
    ("parallel_precision", {"n_batched_turns": 1}),
    ("parallel_detection", {"expected_parallel": True, "predicted_parallel": True}),
]
BEFORE_PARALLEL_IRRELEVANCE = [
    ("parallel_detection", {"expected_parallel": False, "predicted_parallel": False}),
]


@pytest.mark.parametrize("metric,reference,predicted,expected,before", [
    (ToolCallF1, [[A, B], [C]], [[A, B], [C]], [1.0] * 4, BEFORE_F1),
    (ToolCallF1, [], None, [1.0] * 4, BEFORE_F1_IRRELEVANCE),           # no tool called
    (ToolCallF1, [], [[A]], [0.0] * 4, BEFORE_F1_IRRELEVANCE_CALLED),   # irrelevance broken
    (TrajectoryMatch, [[A, B], [C]], [[A, B], [C]], [1.0] * 2, BEFORE_TRAJECTORY),
    (TrajectoryMatch, [], None, [1.0] * 2, BEFORE_TRAJECTORY_IRRELEVANCE),
    (ParallelToolCalls, [[A, B], [C]], [[A, B], [C]], [1.0] * 3, BEFORE_PARALLEL),
    (ParallelToolCalls, [], None, [1.0], BEFORE_PARALLEL_IRRELEVANCE),
])
def test_a_single_route_reference_scores_exactly_what_it_scored_before(metric, reference, predicted,
                                                                       expected, before):
    got = score_triples(metric(), reference, predicted)
    assert [(name, meta) for name, _, meta in got] == before
    assert [value for _, value, _ in got] == pytest.approx(expected)
    # the new keys are absent, not merely unread by this test
    assert not [k for _, _, meta in got for k in meta if k in PATH_KEYS]


@pytest.mark.parametrize("reference,predicted", [
    (None, None),                            # no reference at all: still scored as no calls expected
    ([A], [[A]]),
    ([[A, B], [C]], [[A], [B], [C]]),        # a partial run keeps its own counts
    ([[A, B], [C]], [[A, X], [C]]),          # a wrong run keeps them too
])
def test_no_single_route_reference_is_annotated_with_the_route_it_was_read_against(reference, predicted):
    for metric in REFERENCE_METRICS:
        for _, _, meta in score_triples(metric(), reference, predicted):
            assert set(meta).isdisjoint(PATH_KEYS)


# -- each route is a correct answer -------------------------------------------

@pytest.mark.parametrize("predicted,route,route_f1", [
    ([[PARIS, ROME]], 0, [1.0, 0.0]),   # the two independent lookups, batched
    ([[COMPARE]], 1, [0.0, 1.0]),       # the one call that does both
])
def test_a_run_that_took_a_listed_route_scores_perfectly_and_says_which_one(predicted, route, route_f1):
    scores = score_triples(ToolCallF1(), EITHER_ROUTE, predicted)
    assert [value for _, value, _ in scores] == [1.0] * 4  # precision, recall, f1, exact
    # every score of the family carries the route, so a reader comparing
    # tool_call_f1 with trajectory_match can see that both read the same one
    for _, _, meta in scores:
        assert (meta["matched_path"], meta["n_paths"], meta["path_f1"]) == (route, 2, route_f1)


@pytest.mark.parametrize("route", [TWO_LOOKUPS, ONE_LOOKUP])
def test_the_route_that_was_taken_scores_exactly_as_that_route_alone_would(route):
    # Routes are never expanded or combined: the run is measured against the one
    # route it took, so naming a second, wrong route cannot move a perfect run
    # off what its chosen route would have said on its own.
    listed = {"any_of": [route, [[c("never")]]]}
    assert without_path_keys(score_triples(ToolCallF1(), listed, route)) == \
        without_path_keys(score_triples(ToolCallF1(), route, route))


# -- mixed and partial runs ---------------------------------------------------

def test_a_run_mixing_parts_of_two_routes_is_scored_against_one_of_them_not_both():
    # get_weather(Paris) then compare_weather: half of each route. Crediting the
    # best part of every route would be precision 1.0 and recall 1.0; a route is
    # a whole trajectory, so the run is measured against the one it got furthest
    # along -- compare_weather alone, one call made of the one it expects.
    scores = by_name(ToolCallF1(), EITHER_ROUTE, [[PARIS, COMPARE]])
    assert scores["tool_call_f1"][0] == pytest.approx(2 / 3)
    assert scores["tool_call_recall"][0] == 1.0
    assert scores["tool_call_exact"][0] == 0.0
    assert scores["tool_call_f1"][1]["matched_path"] == 1
    assert scores["tool_call_precision"][1]["n_expected"] == 1  # route 1's single call


def test_a_partial_run_is_credited_with_the_route_it_got_furthest_along_on():
    # One call of three predicted matches the 3-call route; two of three match
    # the 9-call one. Both score the same call-level F1 (1/3), so the recall
    # tiebreak is the only thing that can decide -- and it must pick the route
    # the run progressed on (recall 1/3 against 2/9), not the one listed first.
    listed = {"any_of": [SHORT_ROUTE, LONG_ROUTE]}
    i, f1s = select_reference_path(to_reference_paths(listed), to_turns([[A, B, C]]), "exact")
    assert f1s == [0.333333, 0.333333] and i == 0
    j, _ = select_reference_path(to_reference_paths({"any_of": [LONG_ROUTE, SHORT_ROUTE]}),
                                 to_turns([[A, B, C]]), "exact")
    assert j == 1  # the listing is the only change, and it flips the answer
    # the tie is visible, not silent: a reader sees both routes scored alike
    f1 = by_name(ToolCallF1(), listed, [[A, B, C]])["tool_call_f1"][1]
    assert f1["matched_path"] == 0 and f1["path_f1"] == [0.333333, 0.333333]
    assert by_name(ToolCallF1(), listed, [[A, B, C]])["tool_call_recall"][0] == pytest.approx(1 / 3)


# -- the three reference metrics read one route -------------------------------

@pytest.mark.parametrize("predicted", [
    [[PARIS, ROME]],      # route 0 exactly
    [[COMPARE]],          # route 1 exactly
    [[PARIS, COMPARE]],   # one call from each route
    [[PARIS]],            # half of route 0
    [[X]],                # neither route
])
def test_the_three_reference_metrics_agree_on_which_route_the_run_took(predicted):
    # Were they to rank routes by their own thing, tool_call_f1 could take its
    # recall from one route while parallel_tool_calls took its structure from
    # another, and the scores would describe two different runs.
    chosen, f1s = select_reference_path(to_reference_paths(EITHER_ROUTE), to_turns(predicted), "exact")
    reported = set()
    for metric in REFERENCE_METRICS:
        scores = score_triples(metric(), EITHER_ROUTE, predicted)
        assert scores, f"{metric.__name__} emitted nothing to check"
        for _, _, meta in scores:
            reported.add(meta["matched_path"])
            assert (meta["n_paths"], meta["path_f1"]) == (2, f1s)
    assert reported == {chosen}


@pytest.mark.parametrize("predicted,route,strict,detection", [
    ([[PARIS, ROME]], 0, 1.0, 1.0),        # route 0's parallel group, batched
    ([[COMPARE]], 1, 1.0, 1.0),            # route 1 has no group, nothing batched
    ([[PARIS], [ROME]], 0, 0.0, 0.0),      # route 0, with its group serialized
    ([[PARIS]], 0, 0.0, 0.0),              # half of route 0
    ([[COMPARE], [X]], 1, 0.0, 1.0),       # route 1, plus a call it does not expect
])
def test_a_metric_value_only_follows_from_the_route_the_metrics_agreed_on(predicted, route, strict,
                                                                         detection):
    # Not a tautology: read against the OTHER route, parallel_detection says the
    # opposite for every one of these runs, so the value it reports is evidence
    # of which route that metric actually used.
    assert matched_path(ToolCallF1(), EITHER_ROUTE, predicted) == route
    assert by_name(TrajectoryMatch(), EITHER_ROUTE, predicted)["trajectory_strict"][0] == strict
    other = ONE_LOOKUP if route == 0 else TWO_LOOKUPS
    assert by_name(ParallelToolCalls(), EITHER_ROUTE, predicted)["parallel_detection"][0] == detection
    assert by_name(ParallelToolCalls(), other, predicted)["parallel_detection"][0] != detection


# -- [] is a route ------------------------------------------------------------

def test_calling_no_tool_is_perfect_when_the_listing_says_it_is():
    scores = by_name(ToolCallF1(), {"any_of": [[], TWO_LOOKUPS]}, None)
    assert [value for value, _ in scores.values()] == [1.0] * 4
    f1 = scores["tool_call_f1"][1]
    assert (f1["matched_path"], f1["n_paths"], f1["path_f1"]) == (0, 2, [1.0, 0.0])


def test_a_call_is_measured_against_the_best_route_when_no_call_is_only_one_option():
    f1 = by_name(ToolCallF1(), {"any_of": [[], TWO_LOOKUPS]}, [[PARIS, ROME]])["tool_call_f1"]
    assert f1[0] == 1.0
    assert (f1[1]["matched_path"], f1[1]["path_f1"]) == (1, [0.0, 1.0])


@pytest.mark.parametrize("reference", [
    {"any_of": [TWO_LOOKUPS, ONE_LOOKUP]},
    {"any_of": [[], TWO_LOOKUPS]},   # the empty route listed first still wins: the
    {"any_of": [[], ONE_LOOKUP]},    # rule is the listing, not the "nicer" route
])
def test_a_call_that_matches_no_route_is_read_against_the_first_one_listed(reference):
    # Nothing to rank on -- every route scores 0.0 -- so the last tiebreak,
    # listed order, decides. The score is 0.0 either way, and it says which route
    # it was read against rather than leaving the reader to guess.
    f1 = by_name(ToolCallF1(), reference, [[X]])["tool_call_f1"]
    assert f1[0] == 0.0
    assert f1[1]["matched_path"] == 0
    assert f1[1]["path_f1"] == [0.0] * len(reference["any_of"])


# -- invalid references -------------------------------------------------------

@pytest.mark.parametrize("reference,problem", [
    ({"any_of": []}, "non-empty list of routes"),
    ({"any_of": {}}, "non-empty list of routes"),
    ({"any_of": {"route": TWO_LOOKUPS}}, "non-empty list of routes"),
    ({"any_of": [TWO_LOOKUPS], "weight": 0.5}, "must carry no other keys"),
])
def test_a_malformed_multi_route_reference_is_rejected_with_a_readable_error(reference, problem):
    # An empty or non-list `any_of` would otherwise read as "no route" and score
    # every run 0.0; a key beside it would be silently ignored, and a silently
    # ignored key in a reference is a silently wrong score.
    with pytest.raises(ValueError, match=problem):
        to_reference_paths(reference)


def test_a_route_may_not_itself_name_alternative_routes():
    # Routes are alternatives, not a tree: nesting would hide which of the inner
    # routes actually matches, and the error has to say so rather than flatten.
    with pytest.raises(ValueError, match="route 0 nests 'any_of'"):
        to_reference_paths({"any_of": [{"any_of": [TWO_LOOKUPS]}]})


def test_an_unreadable_call_inside_a_route_names_the_route_it_sits_in():
    reference = {"any_of": [TWO_LOOKUPS, ONE_LOOKUP, [42]]}
    with pytest.raises(ValueError, match="route 2 of 'any_of'"):
        to_reference_paths(reference)
    # and the metric does not swallow it: a bad dataset cell is reported, not scored
    with pytest.raises(ValueError, match="route 2 of 'any_of'"):
        ToolCallF1().score(Sample(input="q", expected_tool_calls=reference), "", {})


# -- the route budget ---------------------------------------------------------

def test_a_reference_naming_more_routes_than_the_budget_is_rejected():
    too_wide = {"any_of": [[[A]]] * (MAX_REFERENCE_PATHS + 1)}
    with pytest.raises(ValueError, match=f"{MAX_REFERENCE_PATHS + 1} routes, over the "
                                         f"{MAX_REFERENCE_PATHS}-route budget"):
        select_reference_path(to_reference_paths(too_wide), to_turns([[A]]), "exact")
    with pytest.raises(ValueError, match="route budget"):
        ToolCallF1().score(Sample(input="q", expected_tool_calls=too_wide), "",
                           {"trace": {"tool_calls": [[A]]}})


def test_a_reference_naming_exactly_the_budget_of_routes_is_scored_in_full():
    # Only the last route matches, so a truncated budget would have reported the
    # first one: the whole listing is read.
    at_budget = {"any_of": [[[A]]] * (MAX_REFERENCE_PATHS - 1) + [[[B]]]}
    f1 = by_name(ToolCallF1(), at_budget, [[B]])["tool_call_f1"]
    assert f1[0] == 1.0
    assert f1[1]["n_paths"] == MAX_REFERENCE_PATHS
    assert f1[1]["matched_path"] == MAX_REFERENCE_PATHS - 1


def test_an_over_budget_reference_is_a_recorded_error_not_a_wrong_score(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    too_wide = {"any_of": [[[A]]] * (MAX_REFERENCE_PATHS + 1)}
    r = ak.evaluate([Sample(input="q", expected_tool_calls=too_wide, actual_output="x",
                            actual_trace={"tool_calls": [[A]]})],
                    model="precomputed", scorers=[ToolCallF1()])
    assert "route budget" in r.errors[0]["error"]
    assert "tool_call_f1" not in r.headline and r.failed_count == 1


# -- a reference that arrives as JSON -----------------------------------------

def test_a_reference_read_from_a_json_cell_still_names_several_routes():
    # A dataset cell is a string. If `any_of` did not survive the decode, every
    # alternative route in the file would quietly collapse into one gold route.
    cell = json.dumps(EITHER_ROUTE)
    assert to_reference_paths(cell) == to_reference_paths(EITHER_ROUTE)
    f1 = by_name(ToolCallF1(), cell, [[COMPARE]])["tool_call_f1"]
    assert f1[0] == 1.0 and f1[1]["matched_path"] == 1


# -- end to end ---------------------------------------------------------------

def test_evaluate_records_the_route_on_every_reference_score_it_emits(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    def sample(sid, pred):
        return Sample(input="weather in Paris and Rome", id=sid, expected_tool_calls=EITHER_ROUTE,
                      actual_output="Paris sunny, Rome rain.", actual_trace={"tool_calls": pred})
    # s1 read route 1, which has no parallel group and so emits no
    # parallel_recall/precision; s2 read route 0, which does
    r = ak.evaluate([sample("s1", [[COMPARE]]), sample("s2", [[PARIS, ROME]])],
                    model="precomputed", scorers=[m() for m in REFERENCE_METRICS])
    assert r.errors == []
    assert r.headline["tool_call_f1"] == 1.0 and r.headline["parallel_recall"] == 1.0
    for route, f1s, row in ((1, [0.0, 1.0], r.samples[0]), (0, [1.0, 0.0], r.samples[1])):
        for sc in row.metadata["scores"]:
            assert (sc["metadata"]["matched_path"], sc["metadata"]["n_paths"],
                    sc["metadata"]["path_f1"]) == (route, 2, f1s), sc["name"]
    # between the two runs every reference score is accounted for
    assert {sc["name"] for row in r.samples for sc in row.metadata["scores"]} == {
        "tool_call_precision", "tool_call_recall", "tool_call_f1", "tool_call_exact",
        "trajectory_strict", "trajectory_in_order",
        "parallel_recall", "parallel_precision", "parallel_detection"}

    p = tmp_path / "run.json"
    r.save(str(p))
    saved = parse_strict(p.read_text())
    assert {sc["metadata"]["matched_path"] for row in saved["predictions"]
            for sc in row["metadata"]["scores"]} == {0, 1}


def test_rescore_reports_the_route_on_every_reference_score():
    case = AgentCase(id="c1", task="weather in Paris and Rome",
                     allowed_tools=["get_weather", "compare_weather"], reference_turns=EITHER_ROUTE)
    ep = episode("Paris sunny, Rome rain.", [[("compare_weather", COMPARE["arguments"])]], "c1")
    r = rescore([ep], list(REFERENCE_SCORERS), cases=[case])
    row = r.rows[0]
    assert row.ineligible == {} and row.status == "completed"
    for name, diag in row.diagnostics.items():
        assert (diag["metadata"]["matched_path"], diag["metadata"]["n_paths"],
                diag["metadata"]["path_f1"]) == (1, 2, [0.0, 1.0]), name
    saved = parse_strict(r.to_json())
    assert saved["rows"][0]["diagnostics"]["tool_call_f1"]["metadata"]["matched_path"] == 1


def test_rescore_and_the_evaluation_harness_read_the_same_case_the_same_way(monkeypatch, tmp_path):
    # One reference, one run, two front doors: the numbers a reader compares
    # across a stored agent-eval report and a fresh run have to describe the
    # same route, or the two reports say different things about one agent.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    case = AgentCase(id="c1", task="weather in Paris and Rome", reference_turns=EITHER_ROUTE)
    ep = episode("Paris sunny, Rome rain.", [[("compare_weather", COMPARE["arguments"])]], "c1")
    rescored = rescore([ep], list(REFERENCE_SCORERS), cases=[case]).rows[0]
    run = ak.evaluate([Sample(input="weather in Paris and Rome", expected_tool_calls=case.reference_turns,
                              actual_output="Paris sunny, Rome rain.",
                              actual_trace={"tool_calls": [[COMPARE]]})],
                      model="precomputed", scorers=[m() for m in REFERENCE_METRICS])
    harness = {sc["name"]: sc["metadata"]["matched_path"] for sc in run.samples[0].metadata["scores"]}
    assert harness == {name: diag["metadata"]["matched_path"] for name, diag in rescored.diagnostics.items()}
    assert set(harness.values()) == {1}


# -- the case fingerprint -----------------------------------------------------

def test_route_order_is_part_of_the_case_identity():
    # The listed order is the last tiebreak, so a run that fits two routes
    # equally well is read against whichever was written first. That makes the
    # order part of what the reference says: the two listings below are the same
    # calls in a different order and must not share a digest.
    split, batched = [[A], [B]], [[A, B]]
    for first, second in ((split, batched), (batched, split)):
        assert AgentCase(id="c1", task="t", reference_turns={"any_of": [first, second]}).digest() != \
            AgentCase(id="c1", task="t", reference_turns={"any_of": [second, first]}).digest()
    # The order no longer decides WHICH structurally-distinct route an exactly
    # reproducing run is read against -- see
    # test_a_run_reproducing_one_route_structure_is_not_scored_against_the_other --
    # but it still decides when a run reproduces both, which is what keeps it
    # part of the identity.
    run = [[A], [B]]
    assert matched_path(ToolCallF1(), {"any_of": [split, batched]}, run) == 0
    assert matched_path(ToolCallF1(), {"any_of": [batched, split]}, run) == 1
    assert by_name(TrajectoryMatch(), {"any_of": [split, batched]}, run)["trajectory_strict"][0] == 1.0
    assert by_name(TrajectoryMatch(), {"any_of": [batched, split]}, run)["trajectory_strict"][0] == 1.0


# -- routes that differ only in turn structure --------------------------------

# "Batch these in one response" and "call them one at a time" are both correct for
# the same task, and the two routes hold the same calls, so call-level F1 cannot
# tell them apart. Ranking on F1 alone left the listing to decide -- and a run
# that did exactly one of them was scored against the other, taking a structural
# 0.0 for doing the right thing. The two structural keys fix that; listed order
# is still there for the runs that genuinely fit both.

def _route(*turns):
    """One route: its turns, each turn a list of calls."""
    return [list(t) for t in turns]


BATCHED = _route((PARIS, ROME))                      # both calls in one response
SERIAL = _route((PARIS,), (ROME,))                   # one call per response
TWO_BATCHED = _route((A, B), (C, D))                 # two turns, four calls
TWO_REORDERED = _route((B, A), (D, C))              # the same four calls, other order
DOUBLED = _route((PARIS, ROME), (PARIS, ROME))       # the same two calls, twice


@pytest.mark.parametrize("routes,run,route", [
    ([BATCHED, SERIAL], SERIAL, 1),        # serial, listed second
    ([SERIAL, BATCHED], BATCHED, 1),      # batched, listed second
    ([BATCHED, DOUBLED], DOUBLED, 1),      # one turn vs two
    ([TWO_BATCHED, TWO_REORDERED], TWO_REORDERED, 0),   # reordered WITHIN each turn
])
def test_a_run_reproducing_one_route_structure_is_not_scored_against_the_other(routes, run, route):
    reference = {"any_of": routes}
    assert select_reference_path(to_reference_paths(reference), to_turns(run), "exact")[0] == route
    # and every reference metric reads that same route, and scores it as the
    # correct run rather than a structural miss
    for metric in REFERENCE_METRICS:
        for _, value, meta in score_triples(metric(), reference, run):
            assert meta["matched_path"] == route
    assert by_name(TrajectoryMatch(), reference, run)["trajectory_strict"][0] == 1.0
    assert by_name(ToolCallF1(), reference, run)["tool_call_f1"][0] == 1.0


def test_reordering_calls_inside_one_turn_leaves_the_routes_tied_not_wrong():
    # Parallel calls have no order, so two turns holding the same calls in a
    # different order are the same route shape: the structural keys tie and the
    # listing decides -- and whichever it picks, the run still scores a perfect
    # trajectory, because the choice is not what makes the score right here.
    reference = {"any_of": [TWO_BATCHED, TWO_REORDERED]}
    assert matched_path(ToolCallF1(), reference, TWO_REORDERED) == 0
    assert by_name(TrajectoryMatch(), reference, TWO_REORDERED)["trajectory_strict"][0] == 1.0
    assert by_name(TrajectoryMatch(), reference, TWO_BATCHED)["trajectory_strict"][0] == 1.0


def test_f1_outranks_the_structural_keys():
    # One call in one response vs the same two calls in two: F1 already separates
    # them (the run makes four calls, so the one-turn route cannot match them
    # all), so the structural keys are never consulted. They are a tiebreak, not
    # a second opinion that can outvote the score.
    for reference, expected in (({"any_of": [BATCHED, DOUBLED]}, 1),
                                ({"any_of": [DOUBLED, BATCHED]}, 0)):
        assert matched_path(ToolCallF1(), reference, DOUBLED) == expected
    f1 = by_name(ToolCallF1(), {"any_of": [BATCHED, DOUBLED]}, DOUBLED)["tool_call_f1"]
    assert f1[0] == 1.0 and f1[1]["path_f1"] == [0.666667, 1.0]


def test_a_run_that_reproduces_neither_structure_still_falls_to_the_listing():
    # The structural keys rank what the run actually did; they do not make the
    # listing irrelevant. A run matching neither route equally is still read
    # against the first one written, and the digest still depends on the order.
    parallel, serial = [[PARIS, ROME]], [[PARIS], [ROME]]
    half = [[PARIS]]
    for reference, expected in (({"any_of": [parallel, serial]}, 0),
                                ({"any_of": [serial, parallel]}, 0)):
        assert matched_path(ToolCallF1(), reference, half) == expected


def test_a_partial_run_is_still_credited_to_the_route_it_got_furthest_along_on():
    # The structural keys sit AFTER recall, so they cannot outvote it: a run that
    # matched a third of one route and a third of another, but covered more of
    # one, still goes to the one it covered more of.
    short, long_ = [[A, C, D]], [[A, B] + UNRELATED]
    i, f1s = select_reference_path(to_reference_paths({"any_of": [short, long_]}),
                                   to_turns([[A, B, C]]), "exact")
    assert f1s[0] > f1s[1] and i == 0



def test_one_route_and_a_one_route_any_of_are_deliberately_different_cases():
    # They score the same -- a one-route listing is still a single route, so it
    # gets no path metadata either -- but they are not the same reference, and
    # collapsing them would change the identity of existing datasets silently.
    plain = AgentCase(id="c1", task="t", reference_turns=TWO_LOOKUPS)
    listed = AgentCase(id="c1", task="t", reference_turns={"any_of": [TWO_LOOKUPS]})
    assert plain.digest() != listed.digest()
    assert to_reference_paths(plain.reference_turns) == to_reference_paths(listed.reference_turns)
    assert by_name(ToolCallF1(), plain.reference_turns, [[PARIS, ROME]]) == \
        by_name(ToolCallF1(), listed.reference_turns, [[PARIS, ROME]])


# -- eligibility --------------------------------------------------------------

@pytest.mark.parametrize("reference", [
    [],                                  # irrelevance: calling no tool is correct
    {"any_of": [[]]},                    # the same thing, written the new way
    {"any_of": [[], TWO_LOOKUPS]},       # answering without a tool is one of the options
])
def test_a_no_call_route_is_a_reference_not_a_missing_one(reference):
    # An agent reply that OBSERVED zero calls is evidence, not a gap, so the only
    # thing that can make a reference metric ineligible here is a case with no
    # reference at all -- a truthiness test would drop this sample instead.
    ep = AgentEpisode(coverage={"tool_call_names": OBSERVED})
    assert ep.turns() == [] and not ep.calls_unnamed()
    case = AgentCase(id="c1", task="t", reference_turns=reference)
    for scorer in REFERENCE_SCORERS:
        assert eligibility(scorer, case, ep) is None, scorer


def test_only_an_absent_reference_is_reported_as_the_missing_field():
    ep = AgentEpisode(coverage={"tool_call_names": OBSERVED})
    case = AgentCase(id="c1", task="t")  # reference_turns is None: no reference given
    for scorer in REFERENCE_SCORERS:
        assert eligibility(scorer, case, ep) == "case.reference_turns"
    # the reference is checked first, so the report never blames the trace for
    # a case that simply has no reference
    assert eligibility("tool_call_f1", case, AgentEpisode()) == "case.reference_turns"
    assert eligibility("tool_call_f1", AgentCase(id="c1", task="t", reference_turns=[]),
                       AgentEpisode()) == "trace.tool_calls"
