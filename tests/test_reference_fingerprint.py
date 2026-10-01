"""#44: a JSON-string reference (a CSV cell, a JSONL field) fingerprints like its decoded form.

Both identities that hash a reference: the dataset fingerprint (``Sample.expected_tool_calls``,
which decides cache reuse) and ``AgentCase.digest()``. Dict/list references keep their digests.
"""

from __future__ import annotations

import csv
import json

import auditkit as ak
from auditkit.agent_eval.types import AgentCase
from auditkit.runspec import RunSpec
from auditkit.scenario import ListScenario

REF = [[{"name": "get_weather", "arguments": {"city": "Paris"}}]]


def fp(samples):
    """The dataset fingerprint: ListScenario's content-hash name, part of every run's cache key."""
    return ListScenario(samples).name


def dataset_fp(ref):
    return fp([ak.Sample(id="a", input="Weather in Paris?", expected_tool_calls=ref)])


def test_a_string_reference_fingerprints_like_the_list():
    assert dataset_fp(json.dumps(REF)) == dataset_fp(REF)


def test_whitespace_and_key_order_in_the_cell_do_not_matter():
    pretty = '[[ {"arguments": {"city": "Paris"},\n  "name": "get_weather"} ]]'
    assert dataset_fp(pretty) == dataset_fp(REF)


def test_a_different_reference_still_fingerprints_differently():
    assert dataset_fp(json.dumps([[{"name": "get_weather", "arguments": {"city": "Rome"}}]])) != dataset_fp(REF)


def test_the_empty_reference_string_is_not_the_missing_reference():
    assert dataset_fp("[]") == dataset_fp([]) != dataset_fp(None)


def test_an_unparseable_cell_still_fingerprints_and_differs():
    assert dataset_fp("[[{oops") != dataset_fp(REF)            # never raises here; scoring reports it


def test_any_of_as_a_string_matches_the_dict():
    ref = {"any_of": [REF, []]}
    assert dataset_fp(json.dumps(ref)) == dataset_fp(ref)


def test_a_csv_round_trip_keeps_the_identity(tmp_path):
    # what a user does: keep the dataset in a spreadsheet, read the cell back as a string
    path = tmp_path / "refs.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "input", "expected_tool_calls"])
        w.writerow(["a", "Weather in Paris?", json.dumps(REF)])
    with path.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert isinstance(row["expected_tool_calls"], str)
    s = ak.Sample(id=row["id"], input=row["input"], expected_tool_calls=row["expected_tool_calls"])
    assert fp([s]) == dataset_fp(REF)


def test_a_string_reference_run_reuses_the_cached_dict_run(tmp_path, monkeypatch):
    # end to end: the second evaluate() is a cache hit, the point of the fix
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def model(prompts):
        calls.append(len(prompts))
        return ['<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'] * len(prompts)

    for ref in (REF, json.dumps(REF)):
        r = ak.evaluate([ak.Sample(id="a", input="Weather in Paris?", expected_tool_calls=ref)],
                        model=model, scorers=[ak.ToolCallF1()])
        assert r.headline["tool_call_f1"] == 1.0
    assert calls == [1]


def case(ref):
    return AgentCase(id="c", task="Weather in Paris?", reference_turns=ref)


def test_agent_case_digest_string_matches_list():
    assert case(json.dumps(REF)).digest() == case(REF).digest()
    assert case("[]").digest() == case([]).digest() != case(None).digest()


def test_agent_case_digest_of_a_list_reference_is_unchanged_by_the_fix():
    # the dict/list form is hashed exactly as before: decode_reference returns it untouched
    from auditkit.trace import decode_reference
    assert decode_reference(REF) is REF and decode_reference(None) is None
