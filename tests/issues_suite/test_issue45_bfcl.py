"""#45 on the real BFCL v3 files (pinned revision): the one_of count, the single-turn loader,
and the matcher, with the edge cases a user running BFCL hits.
"""

from __future__ import annotations

import json
import logging
import re
import shutil

import pytest

import auditkit as ak
from auditkit.bfcl import IRRELEVANCE, SINGLE_TURN

from .conftest import BFCL_REVISION

EXPECTED_ENTRIES = {"simple": 400, "multiple": 200, "parallel": 200, "parallel_multiple": 200,
                    "live_simple": 258, "live_multiple": 1053, "live_parallel": 16,
                    "live_parallel_multiple": 24, "java": 100, "javascript": 50, "sql": 100}
MULTI_TURN = ("base", "composite", "long_context", "miss_func", "miss_param")


def jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


# -- the count that decides phase 3 ------------------------------------------------------------------

def test_the_files_are_jsonl_not_json(bfcl_root):
    with pytest.raises(json.JSONDecodeError):
        json.load(open(bfcl_root / "possible_answer" / "BFCL_v3_simple.json"))


def test_single_turn_entry_counts_match_the_pinned_revision(bfcl_root):
    counts = {c: len(jsonl(bfcl_root / "possible_answer" / f"BFCL_v3_{c}.json")) for c in EXPECTED_ENTRIES}
    assert counts == EXPECTED_ENTRIES and sum(counts.values()) == 2601


def test_argument_alternatives_are_the_only_ambiguity_in_single_turn(bfcl_root):
    alternatives, multi_name_calls = 0, 0
    for c in EXPECTED_ENTRIES:
        for row in jsonl(bfcl_root / "possible_answer" / f"BFCL_v3_{c}.json"):
            for call in row["ground_truth"]:
                multi_name_calls += len(call) != 1          # a call naming two functions: never
                for args in call.values():
                    alternatives += sum(isinstance(v, list) and len(v) > 1 for v in args.values())
    assert (alternatives, multi_name_calls) == (3259, 0)


def test_multi_turn_never_offers_an_alternative_function(bfcl_root):
    # The count Pratinav asked for before any one_of work: every multi-turn answer is a plain list
    # of call strings per turn, with no way to name alternatives. So one_of has no BFCL use.
    entries = calls = 0
    blanks = []
    for c in MULTI_TURN:
        for row in jsonl(bfcl_root / "possible_answer" / f"BFCL_v3_multi_turn_{c}.json"):
            entries += 1
            for turn in row["ground_truth"]:
                assert isinstance(turn, list)
                for call in turn:
                    assert isinstance(call, str)
                    if not call.strip():
                        blanks.append(row["id"])            # a data defect, not an alternative
                        continue
                    assert re.match(r"^\s*[\w.]+\(", call), call
                    calls += 1
    assert (entries, calls, blanks) == (1000, 5961, ["multi_turn_composite_185"])


# -- the loader, every supported category ---------------------------------------------------------------

def replay(sample):
    """The first accepted value of every argument, as a model that got it right would send it."""
    def first(alts):
        return next((pick(a) for a in alts if a != ""), None)

    def pick(a):
        if isinstance(a, dict):
            return {k: first(v if isinstance(v, list) else [v]) for k, v in a.items()
                    if any(x != "" for x in (v if isinstance(v, list) else [v]))}
        return [pick(x) for x in a] if isinstance(a, list) else a

    turn = [{"name": c["name"], "arguments": {k: first(alts) for k, alts in c["arguments"].items()
                                              if any(a != "" for a in alts)}}
            for c in (sample.expected_tool_calls[0] if sample.expected_tool_calls else [])]
    return {"tool_calls": [turn] if turn else []}


def score(samples, traces, **kw):
    runs = [ak.Sample(id=s.id, input=s.input, tools=s.tools, expected_tool_calls=s.expected_tool_calls,
                      metadata=s.metadata, actual_output="", actual_trace=t) for s, t in zip(samples, traces)]
    return ak.evaluate(runs, model="precomputed", scorers=[ak.ToolCallF1(arg_match=ak.bfcl_arg_match, **kw)])


def f1_of(prediction):
    """A prediction's tool_call_f1 (``Prediction.score`` is the first score: precision)."""
    return next(d["value"] for d in prediction.metadata["scores"] if d["name"].startswith("tool_call_f1"))


def f1_one(sample, turns):
    return f1_of(score([sample], [{"tool_calls": turns}]).predictions[0])


@pytest.mark.parametrize("category", SINGLE_TURN + IRRELEVANCE)
def test_the_ground_truth_scores_perfectly_through_the_loader(bfcl_root, category):
    samples = ak.load_bfcl(str(bfcl_root / f"BFCL_v3_{category}.json"), revision=BFCL_REVISION[:7])
    assert samples and all(s.metadata["bfcl_revision"] == BFCL_REVISION[:7] for s in samples)
    r = score(samples, [replay(s) for s in samples])
    wrong = [p.sample_id for p in r.predictions if f1_of(p) != 1.0]
    unsatisfiable = [s.id for s in samples if s.metadata.get("bfcl_unsatisfiable")]
    assert wrong == unsatisfiable, "only an answer with no accepted value may fail its own replay"
    assert not r.errors


def test_the_two_unsatisfiable_answers_are_flagged_not_dropped(bfcl_root):
    samples = ak.load_bfcl(str(bfcl_root / "BFCL_v3_live_simple.json"))
    flagged = {s.id: s.metadata["bfcl_unsatisfiable"] for s in samples if s.metadata.get("bfcl_unsatisfiable")}
    assert len(flagged) == 2 and len(samples) == 258


def test_an_answer_whose_id_drifted_is_paired_by_index_and_said(bfcl_root, caplog):
    # BFCL data defect at this revision: answer live_multiple_1052-279-0 vs question live_multiple_1052-79-0
    with caplog.at_level(logging.WARNING, logger="auditkit.bfcl"):
        samples = ak.load_bfcl(str(bfcl_root / "BFCL_v3_live_multiple.json"))
    s = next(s for s in samples if s.id == "live_multiple_1052-79-0")
    assert s.metadata["bfcl_answer_id"] == "live_multiple_1052-279-0"
    assert len(samples) == 1053 and any("by index" in m for m in caplog.messages)


def test_system_prompts_reach_the_request(bfcl_root):
    samples = ak.load_bfcl(str(bfcl_root / "BFCL_v3_live_simple.json"))
    s = next(s for s in samples if "messages" in s.metadata)
    req = ak.ToolCallAdapter().adapt(s, ak.RunConfig())[0]
    assert req.params["messages"][0]["role"] == "system"
    assert req.params["messages"][-1] == {"role": "user", "content": s.input}
    assert req.params["tools"] == s.tools


def test_sql_questions_in_the_flat_shape_load(bfcl_root):
    samples = ak.load_bfcl(str(bfcl_root / "BFCL_v3_sql.json"))
    assert len(samples) == 100 and samples[0].input.startswith("What is the name of the student")


def test_the_tool_schemas_are_json_schema(bfcl_root):
    bfcl_types = {"dict", "float", "tuple", "any"}

    def walk(node):
        assert node.get("type") not in bfcl_types, node
        for v in (node.get("properties") or {}).values():
            walk(v)
        if isinstance(node.get("items"), dict):
            walk(node["items"])
    for c in ("simple", "parallel_multiple", "live_multiple", "java", "javascript"):
        for s in ak.load_bfcl(str(bfcl_root / f"BFCL_v3_{c}.json")):
            for t in s.tools:
                walk(t["function"]["parameters"])


def test_java_arguments_are_strings_with_the_type_named(bfcl_root):
    s = ak.load_bfcl(str(bfcl_root / "BFCL_v3_java.json"))[0]
    props = s.tools[0]["function"]["parameters"]["properties"]
    assert all(p["type"] == "string" and "Java" in p["description"] for p in props.values())


def test_dots_to_underscores_renames_tools_and_references_together(bfcl_root):
    s = ak.load_bfcl(str(bfcl_root / "BFCL_v3_simple.json"), dots_to_underscores=True)[1]
    assert s.tools[0]["function"]["name"] == "math_factorial"
    assert s.expected_tool_calls[0][0]["name"] == "math_factorial"
    assert f1_one(s, [[{"name": "math_factorial", "arguments": {"number": 5}}]]) == 1.0


@pytest.mark.parametrize("category,why", [("multi_turn_base", "multi-turn"), ("exec_simple", "no possible answers"),
                                          ("rest", "no possible answers"), ("live_relevance", "no possible answers")])
def test_unsupported_categories_are_refused_with_the_reason(bfcl_root, category, why):
    with pytest.raises(ValueError, match=why):
        ak.load_bfcl(str(bfcl_root / f"BFCL_v3_{category}.json"))


def test_a_missing_answer_file_names_the_path(bfcl_root, tmp_path):
    shutil.copy(bfcl_root / "BFCL_v3_simple.json", tmp_path)
    with pytest.raises(FileNotFoundError, match="possible_answer"):
        ak.load_bfcl(str(tmp_path / "BFCL_v3_simple.json"))


def test_a_file_resaved_as_one_json_array_still_loads(bfcl_root, tmp_path):
    (tmp_path / "possible_answer").mkdir()
    for sub in ("", "possible_answer/"):
        rows = jsonl(bfcl_root / f"{sub}BFCL_v3_parallel.json")
        (tmp_path / f"{sub}BFCL_v3_parallel.json").write_text(json.dumps(rows))
    assert len(ak.load_bfcl(str(tmp_path / "BFCL_v3_parallel.json"), limit=5)) == 5


def test_a_renamed_file_is_refused(tmp_path):
    (tmp_path / "my_simple.json").write_text("")
    with pytest.raises(ValueError, match="BFCL_v3_<category>"):
        ak.load_bfcl(str(tmp_path / "my_simple.json"))


# -- the matcher on real entries: what a model actually sends -------------------------------------------

def one(bfcl_root, category, sample_id):
    return next(s for s in ak.load_bfcl(str(bfcl_root / f"BFCL_v3_{category}.json")) if s.id == sample_id)


def call_score(sample, name, args):
    return f1_one(sample, [[{"name": name, "arguments": args}]])


def test_simple_0_accepts_what_bfcl_accepts(bfcl_root):
    s = one(bfcl_root, "simple", "simple_0")                # base [10], height [5], unit ["units", ""]
    area = "calculate_triangle_area"
    assert call_score(s, area, {"base": 10, "height": 5}) == 1.0                    # optional left out
    assert call_score(s, area, {"base": 10, "height": 5, "unit": "Units"}) == 1.0   # case ignored
    assert call_score(s, area, {"base": 10.0, "height": 5}) == 1.0                  # 10.0 is 10
    assert call_score(s, area, {"base": 10, "height": 6}) == 0.0                    # wrong value
    assert call_score(s, area, {"base": 10}) == 0.0                                 # required missing
    assert call_score(s, area, {"base": 10, "height": 5, "color": "red"}) == 0.0    # unexpected parameter
    assert call_score(s, area, {"base": "10", "height": 5}) == 0.0                  # a string is not a number


def test_booleans_are_never_numbers(bfcl_root):
    s = one(bfcl_root, "javascript", "javascript_0")        # isComplete: [True]
    assert call_score(s, "validateUserInput", {"inputField": "userInputField", "isComplete": True}) == 1.0
    assert call_score(s, "validateUserInput", {"inputField": "userInputField", "isComplete": 1}) == 0.0


def test_strings_ignore_spaces_case_and_bfcl_punctuation(bfcl_root):
    s = one(bfcl_root, "sql", "sql_0")
    good = {"sql_keyword": "select", "table_name": "Students", "columns": ["name"], "conditions": ["id=1234"]}
    assert call_score(s, "sql.execute", good) == 1.0
    assert call_score(s, "sql.execute", {**good, "conditions": ["id = 4321"]}) == 0.0


def test_nested_objects_use_their_own_accepted_values(bfcl_root):
    s = one(bfcl_root, "simple", "simple_96")               # conditions: list of {field, operation, value}
    conds = [{"field": "age", "operation": ">", "value": "25"}, {"field": "job", "operation": "=", "value": "engineer"}]
    assert call_score(s, "database.query", {"table": "user", "conditions": conds}) == 1.0
    assert call_score(s, "database.query", {"table": "user", "conditions": conds[::-1]}) == 0.0  # list order counts
    assert call_score(s, "database.query", {"table": "user", "conditions": conds[:1]}) == 0.0


def test_parallel_calls_match_in_any_order(bfcl_root):
    s = ak.load_bfcl(str(bfcl_root / "BFCL_v3_parallel.json"))[0]
    turn = replay(s)["tool_calls"][0]
    assert len(turn) >= 2
    assert f1_one(s, [turn[::-1]]) == 1.0
    assert f1_one(s, [turn[:1]]) < 1.0                  # one call missing


def test_irrelevance_is_one_only_for_no_call(bfcl_root):
    s = ak.load_bfcl(str(bfcl_root / "BFCL_v3_irrelevance.json"))[0]
    assert s.expected_tool_calls == []
    name = s.tools[0]["function"]["name"]
    assert f1_one(s, []) == 1.0
    assert f1_one(s, [[{"name": name, "arguments": {}}]]) == 0.0


def test_without_the_bfcl_matcher_the_lists_are_compared_literally(bfcl_root):
    # the documented trap: default exact matching reads the accepted-value lists as the values
    s = one(bfcl_root, "simple", "simple_1")
    run = [ak.Sample(id=s.id, input=s.input, tools=s.tools, expected_tool_calls=s.expected_tool_calls,
                     actual_output="", actual_trace={"tool_calls": [[{"name": "math.factorial", "arguments": {"number": 5}}]]})]
    assert ak.evaluate(run, model="precomputed", scorers=[ak.ToolCallF1()]).headline["tool_call_f1"] == 0.0
