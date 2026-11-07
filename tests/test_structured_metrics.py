"""JsonSchemaValid and FieldF1: structured-output metrics."""

from __future__ import annotations

import importlib.util
import json

import pytest

import auditkit as ak
from auditkit.metrics.structured import FieldF1, JsonSchemaValid
from auditkit.sample import Sample

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "items"],
    "properties": {
        "id": {"type": "string"},
        "paid": {"type": "boolean"},
        "items": {"type": "array", "minItems": 1,
                  "items": {"type": "object", "required": ["qty"],
                            "properties": {"qty": {"type": "integer", "minimum": 1}, "price": {"type": "number"},
                                           "kind": {"enum": ["a", "b"]}}}},
    },
}
GOOD = {"id": "X1", "paid": True, "items": [{"qty": 2, "price": 9.5, "kind": "a"}]}
S = Sample(input="q")


def valid(metric, obj_or_text) -> float:
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return metric.score(S, text).value


# ============================================================================
# JsonSchemaValid
# ============================================================================

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
BACKENDS = ["minimal", pytest.param("jsonschema", marks=pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed"))]


@pytest.mark.parametrize("backend", BACKENDS)
class TestJsonSchemaValid:
    def test_valid_object(self, backend):
        assert valid(JsonSchemaValid(SCHEMA, backend=backend), GOOD) == 1.0

    @pytest.mark.parametrize("bad", [
        {"items": [{"qty": 1}]},                                  # missing required id
        {"id": 5, "items": [{"qty": 1}]},                         # wrong type
        {"id": "x", "items": []},                                 # minItems
        {"id": "x", "items": [{"qty": 0}]},                       # minimum
        {"id": "x", "items": [{"qty": 1.5}]},                     # integer
        {"id": "x", "items": [{"qty": 1, "kind": "z"}]},          # enum
        {"id": "x", "items": [{"qty": 1}], "extra": 1},           # additionalProperties false
        {"id": "x", "paid": 1, "items": [{"qty": 1}]},            # bool is not a number or the reverse
        [],                                                       # wrong root type
    ])
    def test_invalid_instances(self, backend, bad):
        assert valid(JsonSchemaValid(SCHEMA, backend=backend), bad) == 0.0

    def test_integer_valued_float_is_an_integer(self, backend):
        assert valid(JsonSchemaValid(SCHEMA, backend=backend), {"id": "x", "items": [{"qty": 2.0}]}) == 1.0

    def test_not_json_scores_zero_with_reason(self, backend):
        s = JsonSchemaValid(SCHEMA, backend=backend).score(S, "not json")
        assert s.value == 0.0 and "not valid JSON" in s.reason

    def test_reason_names_the_failing_path(self, backend):
        s = JsonSchemaValid(SCHEMA, backend=backend).score(S, json.dumps({"id": "x", "items": [{"qty": "a"}]}))
        assert s.value == 0.0 and "items" in s.reason and "qty" in s.reason

    def test_strict_by_default_and_fences_opt_in(self, backend):
        fenced = "```json\n" + json.dumps(GOOD) + "\n```"
        assert valid(JsonSchemaValid(SCHEMA, backend=backend), fenced) == 0.0
        assert valid(JsonSchemaValid(SCHEMA, backend=backend, strip_fences=True), fenced) == 1.0

    def test_ref_defs(self, backend):
        schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/n"}}, "$defs": {"n": {"type": "number"}}}
        m = JsonSchemaValid(schema, backend=backend)
        assert valid(m, {"a": 1}) == 1.0 and valid(m, {"a": "x"}) == 0.0

    def test_combinators_and_const(self, backend):
        schema = {"anyOf": [{"type": "string", "pattern": "^a"}, {"const": 3}], "not": {"const": "ab"}}
        m = JsonSchemaValid(schema, backend=backend)
        assert [valid(m, v) for v in ('"ax"', "3", '"ab"', '"b"', "4")] == [1.0, 1.0, 0.0, 0.0, 0.0]

    def test_per_sample_schema_from_metadata(self, backend):
        m = JsonSchemaValid(backend=backend)
        with_schema = Sample(input="q", metadata={"schema": {"type": "array"}})
        assert m.applicable(with_schema) and not m.applicable(S)
        assert m.score(with_schema, "[1]").value == 1.0 and m.score(with_schema, "{}").value == 0.0


def test_minimal_backend_refuses_schemas_it_cannot_check():
    with pytest.raises(ValueError, match="cannot check"):
        JsonSchemaValid({"if": {"type": "string"}, "then": {"minLength": 3}}, backend="minimal")
    with pytest.raises(ValueError, match="cannot check"):
        JsonSchemaValid({"properties": {"a": {"prefixItems": [{"type": "string"}]}}}, backend="minimal")
    # a property that merely happens to be called like a keyword is fine
    assert valid(JsonSchemaValid({"properties": {"if": {"type": "string"}}}, backend="minimal"), {"if": "x"}) == 1.0


def test_minimal_backend_rejects_remote_ref():
    m = JsonSchemaValid({"$ref": "http://example.com/s.json"}, backend="minimal")
    with pytest.raises(ValueError, match="local"):
        m.score(S, "1")


def test_bad_schema_fails_at_construction_and_bad_backend_name():
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.SchemaError):
        JsonSchemaValid({"type": "nonsense"}, backend="jsonschema")
    with pytest.raises(ValueError):
        JsonSchemaValid(SCHEMA, backend="nope")


def test_schema_from_path_and_identity(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(SCHEMA))
    a, b = JsonSchemaValid(p, backend="minimal"), JsonSchemaValid({"type": "array"}, backend="minimal")
    assert valid(a, GOOD) == 1.0
    assert a.identity() != b.identity() and a.identity() == JsonSchemaValid(SCHEMA, backend="minimal").identity()
    assert JsonSchemaValid(SCHEMA, backend="minimal", strip_fences=True).identity() != a.identity()


def test_backends_agree_on_a_corpus():
    pytest.importorskip("jsonschema")
    mini, full = JsonSchemaValid(SCHEMA, backend="minimal"), JsonSchemaValid(SCHEMA, backend="jsonschema")
    corpus = [GOOD, {"id": "x"}, {"id": "x", "items": [{"qty": 1, "price": "1"}]}, {"id": 1, "items": []}, {}, [], "s", 3,
              {"id": "x", "items": [{"qty": 1, "kind": "b"}], "paid": False}]
    assert [valid(mini, c) for c in corpus] == [valid(full, c) for c in corpus]


def test_auto_backend_resolves_to_an_available_one():
    assert JsonSchemaValid(SCHEMA).identity()["backend"] in ("jsonschema", "minimal")


# ============================================================================
# FieldF1
# ============================================================================


def f1(output, target, **kw):
    scores = FieldF1(**kw).score(Sample(input="q", target=json.dumps(target) if not isinstance(target, str) else target),
                                 output if isinstance(output, str) else json.dumps(output))
    return {s.name: s.value for s in scores}, scores[0]


class TestFieldF1:
    def test_exact_match_is_one(self):
        v, _ = f1(GOOD, GOOD)
        assert v == {"field_f1": 1.0, "field_precision": 1.0, "field_recall": 1.0}

    def test_wrong_value_costs_precision_and_recall(self):
        v, s = f1({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert v["field_precision"] == v["field_recall"] == v["field_f1"] == 0.5
        assert "wrong: b" in s.reason

    def test_extra_key_costs_precision_missing_key_costs_recall(self):
        v, s = f1({"a": 1, "extra": 5}, {"a": 1})
        assert v["field_precision"] == 0.5 and v["field_recall"] == 1.0
        v, s = f1({"a": 1}, {"a": 1, "b": 2, "c": 3})
        assert v["field_precision"] == 1.0 and v["field_recall"] == pytest.approx(1 / 3)
        assert "missing: b, c" in s.reason

    def test_nested_paths_flatten(self):
        v, _ = f1({"c": {"n": "x", "m": {"z": 1}}}, {"c": {"n": "x", "m": {"z": 2}}})
        assert v["field_f1"] == 0.5

    def test_list_of_objects_alignment_is_order_independent(self):
        pred = {"items": [{"sku": "B", "q": 2}, {"sku": "A", "q": 1}]}
        truth = {"items": [{"sku": "A", "q": 1}, {"sku": "B", "q": 2}]}
        assert f1(pred, truth)[0]["field_f1"] == 1.0

    def test_greedy_alignment_pairs_best_matches(self):
        pred = {"items": [{"sku": "A", "q": 1, "p": 5}, {"sku": "B", "q": 9, "p": 6}]}
        truth = {"items": [{"sku": "B", "q": 2, "p": 6}, {"sku": "A", "q": 1, "p": 5}]}
        v, _ = f1(pred, truth)
        assert v["field_f1"] == pytest.approx(5 / 6)  # B's q is the only wrong leaf

    def test_extra_and_missing_list_items(self):
        v, _ = f1({"l": [{"a": 1}, {"a": 2}]}, {"l": [{"a": 1}]})
        assert v["field_precision"] == 0.5 and v["field_recall"] == 1.0
        v, _ = f1({"l": [{"a": 1}]}, {"l": [{"a": 1}, {"a": 2}]})
        assert v["field_precision"] == 1.0 and v["field_recall"] == 0.5

    def test_list_keys_do_not_fall_back_to_greedy_for_wrong_ids(self):
        pred = {"rows": [{"id": "X9", "v": 1}]}
        truth = {"rows": [{"id": "X1", "v": 1}]}
        assert f1(pred, truth)[0]["field_f1"] == 0.5  # greedy pairs them: v matches, id differs
        assert f1(pred, truth, list_keys="id")[0]["field_f1"] == 0.0  # keyed: unknown id is an unmatched row

    def test_list_keys_pattern_mapping(self):
        pred = {"orders": [{"lines": [{"sku": "B", "q": 1}, {"sku": "A", "q": 1}]}]}
        truth = {"orders": [{"lines": [{"sku": "A", "q": 1}, {"sku": "B", "q": 1}]}]}
        assert f1(pred, truth, list_keys={"orders[].lines": "sku"})[0]["field_f1"] == 1.0

    def test_scalar_lists_compare_as_multisets(self):
        assert f1({"t": ["b", "a", "a"]}, {"t": ["a", "a", "b"]})[0]["field_f1"] == 1.0
        v, _ = f1({"t": ["a", "a"]}, {"t": ["a", "b"]})
        assert v["field_f1"] == 0.5

    def test_number_normalisation_and_tolerance(self):
        assert f1({"p": "1,234.50"}, {"p": 1234.5})[0]["field_f1"] == 0.0
        assert f1({"p": "1,234.50"}, {"p": 1234.5}, normalize_numbers=True)[0]["field_f1"] == 1.0
        assert f1({"p": "$5"}, {"p": 5}, normalize_numbers=True)[0]["field_f1"] == 1.0
        assert f1({"p": 10.004}, {"p": 10.0}, normalize_numbers=True)[0]["field_f1"] == 0.0
        assert f1({"p": 10.004}, {"p": 10.0}, normalize_numbers=True, number_tol=0.01)[0]["field_f1"] == 1.0

    def test_bool_is_not_a_number(self):
        assert f1({"f": True}, {"f": 1}, normalize_numbers=True)[0]["field_f1"] == 0.0

    def test_date_formats(self):
        assert f1({"d": "05/03/2026"}, {"d": "2026-03-05"})[0]["field_f1"] == 0.0
        assert f1({"d": "05/03/2026"}, {"d": "2026-03-05"}, date_formats=["%Y-%m-%d", "%d/%m/%Y"])[0]["field_f1"] == 1.0

    def test_case_and_whitespace(self):
        assert f1({"n": "Acme  Ltd "}, {"n": "acme ltd"})[0]["field_f1"] == 0.0
        assert f1({"n": "Acme  Ltd "}, {"n": "acme ltd"}, case_sensitive=False)[0]["field_f1"] == 1.0
        assert f1({"n": "A  B"}, {"n": "A B"}, strip=False)[0]["field_f1"] == 0.0

    def test_normalizer_hook_by_path_pattern(self):
        keep_digits = lambda v: "".join(c for c in str(v) if c.isdigit())  # noqa: E731
        pred, truth = {"items": [{"hsn": "HSN 8471"}], "x": "a"}, {"items": [{"hsn": "8471"}], "x": "b"}
        v, _ = f1(pred, truth, normalizers={"items[].hsn": keep_digits})
        assert v["field_f1"] == 0.5

    def test_ignore_drops_leaves_on_both_sides(self):
        v, _ = f1({"a": 1, "ts": "now"}, {"a": 1, "ts": "then", "t2": 5}, ignore=["ts", "t2"])
        assert v["field_f1"] == 1.0

    def test_not_json_and_non_container_output_score_zero(self):
        v, s = f1("nope", {"a": 1})
        assert v == {"field_f1": 0.0, "field_precision": 0.0, "field_recall": 0.0} and "not valid JSON" in s.reason
        assert f1("3", {"a": 1})[0]["field_f1"] == 0.0

    def test_empty_both_is_one_empty_output_is_zero(self):
        assert f1({}, {})[0]["field_f1"] == 1.0
        assert f1({}, {"a": 1})[0]["field_f1"] == 0.0

    def test_type_mismatch_counts_no_true_positives(self):
        assert f1({"a": {"b": 1}}, {"a": 1})[0]["field_f1"] == 0.0

    def test_bad_target_raises(self):
        with pytest.raises(ValueError):
            FieldF1().score(Sample(input="q", target="not json"), "{}")
        with pytest.raises(ValueError):
            FieldF1().score(Sample(input="q", target="3"), "{}")

    def test_target_may_be_a_dict_and_requires_target(self):
        m = FieldF1()
        assert m.score(Sample(input="q", target={"a": 1}), '{"a": 1}')[0].value == 1.0
        assert not m.applicable(Sample(input="q"))

    def test_fenced_output_opt_in(self):
        assert f1("```json\n{\"a\": 1}\n```", {"a": 1})[0]["field_f1"] == 0.0
        assert f1("```json\n{\"a\": 1}\n```", {"a": 1}, strip_fences=True)[0]["field_f1"] == 1.0

    def test_metadata_counts(self):
        _, s = f1({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 0})
        assert s.metadata == {"tp": 1, "n_pred": 3, "n_target": 2}

    def test_identity_changes_with_config(self):
        ids = [FieldF1().identity(), FieldF1(normalize_numbers=True).identity(), FieldF1(list_keys="id").identity(),
               FieldF1(ignore=["x"]).identity(), FieldF1(date_formats=["%d"]).identity(), FieldF1(case_sensitive=False).identity()]
        assert len({json.dumps(i, sort_keys=True) for i in ids}) == len(ids)


# ============================================================================
# Through the pipeline
# ============================================================================


def test_metrics_in_evaluate_with_precomputed_outputs_and_registry():
    assert {"json_schema_valid", "field_f1"} <= set(ak.METRICS.names())
    truth = {"id": "X1", "items": [{"qty": 2}]}
    samples = [Sample(id="a", input="q", target=json.dumps(truth), actual_output=json.dumps(truth)),
               Sample(id="b", input="q", target=json.dumps(truth), actual_output=json.dumps({"id": "X1", "items": [{"qty": "2"}]}))]
    res = ak.evaluate(samples, model="precomputed", scorers=[JsonSchemaValid(SCHEMA, backend="minimal"), FieldF1()])
    assert res.headline["json_schema_valid"] == 0.5
    assert res.headline["field_f1"] == pytest.approx(0.75)
    assert res.headline["field_recall"] == pytest.approx(0.75)
    other = ak.evaluate(samples, model="precomputed", scorers=[JsonSchemaValid({"type": "object"}, backend="minimal"), FieldF1()])
    assert other.fingerprint != res.fingerprint


def test_by_name_scorer_strings_resolve():
    s = Sample(id="a", input="q", target="{}", actual_output="{}")
    assert ak.evaluate([s], model="precomputed", scorers="field_f1").headline["field_f1"] == 1.0
