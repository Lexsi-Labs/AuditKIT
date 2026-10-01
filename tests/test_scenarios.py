"""Tests for T1 pre-built benchmark scenarios."""

from __future__ import annotations


from auditkit.registry import SCENARIOS


class TestScenarioRegistry:
    def test_mmlu_registered(self):
        from auditkit.scenarios.mmlu import MMLUScenario
        assert SCENARIOS.get("mmlu") is MMLUScenario

    def test_gsm8k_registered(self):
        from auditkit.scenarios.gsm8k import GSM8KScenario
        assert SCENARIOS.get("gsm8k") is GSM8KScenario

    def test_arc_registered(self):
        from auditkit.scenarios.arc import ARCScenario
        assert SCENARIOS.get("arc") is ARCScenario

    def test_hellaswag_registered(self):
        from auditkit.scenarios.hellaswag import HellaSwagScenario
        assert SCENARIOS.get("hellaswag") is HellaSwagScenario

    def test_truthfulqa_registered(self):
        from auditkit.scenarios.truthfulqa import TruthfulQAScenario
        assert SCENARIOS.get("truthfulqa") is TruthfulQAScenario

    def test_humaneval_registered(self):
        from auditkit.scenarios.humaneval import HumanEvalScenario
        assert SCENARIOS.get("humaneval") is HumanEvalScenario

    def test_registered_count(self):
        assert len(SCENARIOS.names()) >= 6


class TestListScenarioMetadataHashing:
    """ListScenario now folds Sample.metadata into its fingerprint. It must
    stay robust to free-form metadata and stable across processes (findings
    34, 35), without changing the fingerprint of plain old datasets."""

    def test_old_style_dataset_fingerprint_unchanged(self):
        # Pin the pre-metadata formula: a plain dataset hashes byte-for-byte
        # as before (JSON-native base tuple, no _stable, no repr fallback).
        import hashlib
        import json

        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        # (input, target, choices, retrieval_context, actual_output) -- the
        # five fields hashed before this PR added tools/metadata/etc.
        samples = [
            Sample(input="q", target="a"),
            Sample(input="pick", target="a", choices=["a", "b"],
                   retrieval_context=["ctx"], actual_output="x"),
        ]
        rows = [
            ("q", "a", None, None, None),
            ("pick", "a", ["a", "b"], ["ctx"], "x"),
        ]
        h = hashlib.sha256(json.dumps(rows, default=str, sort_keys=True).encode()).hexdigest()[:8]
        assert ListScenario(samples).name == f"inline_2_{h}"

    def test_mixed_type_keys_do_not_raise(self):
        # [34] json.dumps(sort_keys=True) can't sort {int, str} keys.
        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        name = ListScenario([Sample(input="q", target="a",
                                    metadata={"m": {0: "x", "y": 1}})]).name
        assert name.startswith("inline_1_")

    def test_tuple_keys_and_cycle_do_not_raise(self):
        # [34] non-str keys and a self-reference both crash json.dumps.
        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        assert ListScenario([Sample(input="q", metadata={("a", 1): "x"})]).name.startswith("inline_1_")
        cyclic: dict = {}
        cyclic["self"] = cyclic
        assert ListScenario([Sample(input="q", metadata={"c": cyclic})]).name.startswith("inline_1_")

    def test_metadata_is_stable_across_calls(self):
        # [34]/[35] int+str keys, a set, and a plain object must each hash the
        # same on two independent calls (fresh objects prove no address leaks).
        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        def name(md):
            return ListScenario([Sample(input="q", target="a", metadata=md)]).name

        assert name({"m": {0: "x", "y": 1}}) == name({"m": {0: "x", "y": 1}})
        # different literal set orderings must fold to the same fingerprint
        assert name({"labels": {"a", "b", "c"}}) == name({"labels": {"c", "b", "a"}})
        # A bare object() has no __dict__, so it folds to its type qualname
        # (the __slots__-only/stateless ceiling) -- two fresh instances
        # (different addresses) still hash identically, with no address leak.
        assert name({"obj": object()}) == name({"obj": object()})

    def test_metadata_changes_the_fingerprint(self):
        # Not a wrong HIT: different metadata -> different name.
        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        a = ListScenario([Sample(input="q", target="a", metadata={"k": 1})]).name
        b = ListScenario([Sample(input="q", target="a", metadata={"k": 2})]).name
        assert a != b

    def test_opaque_object_state_changes_the_fingerprint(self):
        # [fix5] An opaque object (plain class, no __repr__) that differs only
        # in its attribute state must NOT collide onto one cache entry. Folding
        # in its __dict__ gives genuinely different metadata a different name,
        # yet two equal-state instances stay stable (no address leak).
        from auditkit.sample import Sample
        from auditkit.scenario import ListScenario

        class Cfg:
            def __init__(self, x):
                self.x = x

        def name(x):
            return ListScenario([Sample(input="q", target="a", metadata={"cfg": Cfg(x)})]).name

        assert name(1) != name(999)          # different state -> different fingerprint
        assert name(1) == name(1)            # same state, fresh objects -> stable
