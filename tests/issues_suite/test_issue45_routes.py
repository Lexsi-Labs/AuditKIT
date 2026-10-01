"""#45 phase 1 on real BFCL parallel answers: a user accepts either the batched calls or the same
calls one per turn, and reads which route runs took and how often listed order decided.
"""

from __future__ import annotations

import json

import auditkit as ak


def two_route_samples(bfcl_root, n=12):
    """(sample, calls): a real parallel answer as {"any_of": [batched, sequential]}."""
    out = []
    for s in ak.load_bfcl(str(bfcl_root / "BFCL_v3_parallel.json"), limit=n):
        calls = [{"name": c["name"], "arguments": {k: next(a for a in v if a != "") for k, v in c["arguments"].items()
                                                   if any(a != "" for a in v)}}
                 for c in s.expected_tool_calls[0]]
        out.append((s, calls, {"any_of": [[calls], [[c] for c in calls]]}))
    return out


def recorded(s, ref, turns):
    return ak.Sample(id=s.id, input=s.input, tools=s.tools, expected_tool_calls=ref,
                     actual_output="", actual_trace={"tool_calls": turns})


def run(samples):
    return ak.evaluate(samples, model="precomputed",
                       scorers=[ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls()])


def test_batched_and_sequential_runs_are_told_apart_by_structure(bfcl_root):
    rows = two_route_samples(bfcl_root)
    runs = [recorded(s, ref, [calls] if k % 3 else [[c] for c in calls]) for k, (s, calls, ref) in enumerate(rows)]
    r = run(runs)
    d = r.route_distribution()
    sequential = sum(1 for k in range(len(rows)) if k % 3 == 0)
    assert d["samples"] == len(rows)
    assert d["by_route"] == {0: len(rows) - sequential, 1: sequential}
    assert d["decided_by"] == {"turn_structure": len(rows)} and d["ties"] == 0
    # every sample scores perfectly on the route it took, the whole point of any_of
    assert r.headline["trajectory_strict"] == 1.0 and r.headline["tool_call_f1"] == 1.0


def test_a_partial_run_is_credited_with_the_route_it_got_furthest_along(bfcl_root):
    s, calls, _ = two_route_samples(bfcl_root, 1)[0]
    other = {"name": "not_a_bfcl_tool", "arguments": {}}
    ref = {"any_of": [[[other]], [calls]]}                  # the listed-first route is the wrong one
    r = run([recorded(s, ref, [calls[:1]])])                # the run made one of the right calls
    assert r.route_distribution() == {"samples": 1, "by_route": {1: 1}, "decided_by": {"f1": 1}, "ties": 0}


def test_when_nothing_decides_the_listing_does_and_it_is_counted(bfcl_root):
    rows = two_route_samples(bfcl_root, 4)
    runs = [recorded(s, ref, [[{"name": "unrelated", "arguments": {}}]]) for s, _, ref in rows]
    d = run(runs).route_distribution()
    assert d["ties"] == 4 and d["by_route"] == {0: 4}
    assert "ties broken by listed order: 4/4" in run(runs).summary()


def test_the_denominator_is_multi_route_samples_only(bfcl_root):
    rows = two_route_samples(bfcl_root, 6)
    runs = [recorded(s, ref, [calls]) for s, calls, ref in rows[:2]] + \
           [recorded(s, [calls], [calls]) for s, calls, _ in rows[2:]]            # single-route references
    assert run(runs).route_distribution()["samples"] == 2


def test_routes_from_a_csv_cell_are_counted_the_same(bfcl_root):
    rows = two_route_samples(bfcl_root, 3)
    as_dicts = run([recorded(s, ref, [calls]) for s, calls, ref in rows]).route_distribution()
    as_cells = run([recorded(s, json.dumps(ref), [calls]) for s, calls, ref in rows]).route_distribution()
    assert as_dicts == as_cells


def test_nothing_reaches_the_stats(bfcl_root):
    rows = two_route_samples(bfcl_root, 3)
    r = run([recorded(s, ref, [calls]) for s, calls, ref in rows])
    assert not [k for k in r.headline if "route" in k or "path" in k]
