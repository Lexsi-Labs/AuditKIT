"""#45 phase 1: which route each multi-route sample took, and how often listed order decided.

Scored from recorded traces (``model="precomputed"``), the way a user rescores agent runs.
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metrics.agent import ROUTE_DECIDERS, _select_reference_path
from auditkit.trace import to_turns

W = lambda c: {"name": "get_weather", "arguments": {"city": c}}
CMP = {"name": "compare_weather", "arguments": {"cities": ["Paris", "Rome"]}}
ROUTES = {"any_of": [[[W("Paris"), W("Rome")]], [[CMP]]]}          # two cities, or one comparison
SEQ_OR_BATCH = {"any_of": [[[W("Paris")], [W("Rome")]], [[W("Paris"), W("Rome")]]]}


@pytest.fixture(autouse=True)
def _fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def s(i, ref, turns):
    return ak.Sample(id=i, input="Weather in Paris vs Rome?", expected_tool_calls=ref,
                     actual_output="", actual_trace={"tool_calls": turns})


def run(samples, scorers=None):
    return ak.evaluate(samples, model="precomputed",
                       scorers=scorers or [ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls()])


def test_mixed_routes_give_the_histogram_and_what_decided():
    r = run([s("a", ROUTES, [[W("Paris"), W("Rome")]]),                 # route 0, clear F1 win
             s("b", ROUTES, [[CMP]]),                                   # route 1
             s("c", ROUTES, [[CMP]]),
             s("d", SEQ_OR_BATCH, [[W("Paris")], [W("Rome")]]),          # same calls: structure decides
             s("e", [[W("Oslo")]], [[W("Oslo")]])])                     # single route: not counted
    d = r.route_distribution()
    assert d["samples"] == 4
    assert d["by_route"] == {0: 2, 1: 2}
    assert d["decided_by"] == {"f1": 3, "turn_structure": 1}
    assert d["ties"] == 0
    line = next(l for l in r.summary().splitlines() if "routes taken" in l)
    assert "4 multi-route samples" in line and "ties broken by listed order: 0/4" in line


def test_a_run_matching_no_route_is_a_listed_order_tie_and_is_reported():
    r = run([s("x", ROUTES, [[{"name": "get_time", "arguments": {"city": "Paris"}}]])])
    assert r.route_distribution()["ties"] == 1
    assert r.route_distribution()["decided_by"] == {"listed_order": 1}


def test_single_route_runs_have_no_section():
    r = run([s("a", [[W("Paris")]], [[W("Paris")]]), s("b", [], [])])
    assert r.route_distribution() == {}
    assert "routes taken" not in r.summary()


def test_the_distribution_never_reaches_the_stats_or_headline():
    r = run([s("a", ROUTES, [[CMP]])])
    assert not any("route" in k or "path" in k for k in list(r.stats) + list(r.headline))


def test_every_reference_metric_carries_the_same_decision():
    r = run([s("d", SEQ_OR_BATCH, [[W("Paris"), W("Rome")]])])
    labels = {(d.get("metadata") or {}).get("path_decided_by")
              for d in r.predictions[0].metadata["scores"] if (d.get("metadata") or {}).get("n_paths")}
    assert labels == {"turn_structure"}


def test_a_json_string_any_of_from_a_csv_cell_is_counted_too():
    import json
    r = run([s("a", json.dumps(ROUTES), [[CMP]])])
    assert r.route_distribution()["by_route"] == {1: 1}


@pytest.mark.parametrize("pred,expected", [
    ([[W("Paris"), W("Rome")]], "f1"),
    ([[W("Paris")]], "f1"),                     # partial run: credited with the route it got furthest along
    ([], "f1"),                                 # no call: the [] route wins outright
])
def test_decider_labels(pred, expected):
    paths = [[[W("Paris"), W("Rome")]], [[CMP]], []]
    _, _, label = _select_reference_path([to_turns(p) for p in paths], to_turns(pred), "exact")
    assert label == expected and label in ROUTE_DECIDERS
