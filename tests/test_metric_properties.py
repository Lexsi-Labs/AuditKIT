"""Property and edge-case checks across the metric library.

Three families:
* target-comparing text metrics: identical strings score the best value (also for the empty,
  article-only, repeated-token and non-Latin cases), scores stay in range, symmetric ones are symmetric;
* every registered metric: each constructor argument that changes the score is reflected in the
  metric's name or ``identity()``, so a run fingerprint never serves another setting's cached scores;
* deterministic metrics never return NaN or raise on odd input.
"""

from __future__ import annotations

import inspect
import math

import pytest

from auditkit.registry import METRICS
from auditkit.sample import Sample

# Text metrics that compare ``output`` with ``sample.target`` and need no model.
SIMILARITY = ["exact_match", "equals", "quasi_exact_match", "f1_score",
              "rouge_l", "token_overlap", "word_error_rate"]
SYMMETRIC = ["f1_score", "rouge_l", "token_overlap"]

TEXTS = [
    "", "paris", "Paris", "the cat sat on a mat", "to be or not to be", "paris paris paris",
    "a a a", "the", "नमस्ते दुनिया", "नमस्ते।", "مرحبا بالعالم", "باريس،", "日本語のテキスト", "パリ。",
    "😀😀", "1,000", "3.14", "  padded  \n", "tab\tseparated", "x" * 2000,
]


def build(name):
    return METRICS.get(name)()


def value(metric, target, output):
    r = metric.score(Sample(input="q", target=target), output)
    r = r[0] if isinstance(r, list) else r
    return r.value


@pytest.mark.parametrize("name", SIMILARITY)
@pytest.mark.parametrize("text", TEXTS)
def test_identical_strings_score_the_best_value(name, text):
    assert value(build(name), text, text) == pytest.approx(1.0)


@pytest.mark.parametrize("name", SIMILARITY)
@pytest.mark.parametrize("target", TEXTS)
@pytest.mark.parametrize("output", ["", "paris", "the cat sat", "paris paris", "नमस्ते", "x" * 50])
def test_scores_are_finite_and_in_range(name, target, output):
    v = value(build(name), target, output)
    assert isinstance(v, (int, float)) and not math.isnan(v) and 0.0 <= v <= 1.0


@pytest.mark.parametrize("name", SIMILARITY)
def test_scoring_is_deterministic(name):
    m = build(name)
    first = [value(m, t, "the cat sat") for t in TEXTS]
    assert first == [value(m, t, "the cat sat") for t in TEXTS]


@pytest.mark.parametrize("name", SYMMETRIC)
@pytest.mark.parametrize("a,b", [("red red blue", "red blue blue green"), ("paris", "paris paris paris"),
                                 ("the cat sat", "a dog ran"), ("नमस्ते दुनिया", "दुनिया")])
def test_symmetric_metrics_are_symmetric(name, a, b):
    m = build(name)
    assert value(m, a, b) == pytest.approx(value(m, b, a))


@pytest.mark.parametrize("name", ["f1_score"])
def test_a_padded_answer_does_not_get_full_credit(name):
    assert value(build(name), "paris", "paris paris paris") < 1.0


@pytest.mark.parametrize("name", ["f1_score", "quasi_exact_match"])
@pytest.mark.parametrize("plain,marked", [("नमस्ते", "नमस्ते।"), ("باريس", "باريس،"), ("パリ", "パリ。"),
                                          ("paris", "Paris."), ("paris", "(paris)")])
def test_punctuation_in_any_script_is_ignored(name, plain, marked):
    m = build(name)
    assert value(m, plain, marked) == pytest.approx(1.0)


def test_f1_score_joins_digits_split_by_a_separator_like_squad():
    assert value(build("f1_score"), "1000", "1,000") == pytest.approx(1.0)


# ---- identity() covers the constructor ------------------------------------

REQUIRED_ARGS = {"contains": {"substring": "x"}, "ends_with": {"suffix": "x"}, "starts_with": {"prefix": "x"},
                 "regex": {"pattern": "x"}, "keyword_detector": {"blacklist": ["x"]}}
# Known gaps, each fixed by an open PR (#12 for ends_with, #14 for the judges). Delete the entries once they merge.
KNOWN_GAPS = {("ends_with", "ignore_case"), ("llm_judge", "direction"), ("structured_judge", "direction")}


def _mutate(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.25
    if isinstance(v, str) and v:
        return v + "_x"
    return None


def _key(m):
    return m.name, repr(m.identity())


def _identity_cases():
    out = []
    for name in sorted(METRICS.names()):
        cls = METRICS.get(name)
        try:
            params = inspect.signature(cls.__init__).parameters.values()
        except (TypeError, ValueError):
            continue
        for p in params:
            if p.name == "self" or p.default is p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if name in REQUIRED_ARGS and p.name in REQUIRED_ARGS[name]:
                continue
            new = _mutate(p.default)
            if new is not None and new != p.default:
                out.append(pytest.param(name, p.name, new, id=f"{name}.{p.name}"))
    return out


# Arguments that only take a fixed set of values: another allowed value to build the changed metric with.
CHOICES = {
    ("encoder_judge", "aggregation"): "argmax",
    ("factuality_encoder_judge", "aggregation"): "argmax",
    ("sentiment_encoder_judge", "aggregation"): "argmax",
    ("guard_judge", "assess"): "prompt",
    ("guard_judge", "profile"): "wildguard",
    # "auto" resolves to the jsonschema package when it is installed, so "minimal" is a different backend.
    ("json_schema_valid", "backend"): "minimal",
    ("parallel_tool_calls", "arg_mode"): "subset",
    ("tool_call_f1", "arg_mode"): "subset",
    ("trajectory_match", "arg_mode"): "subset",
}
# Arguments the default profile rejects: extra keyword arguments to build the changed metric with.
EXTRA: dict = {}


@pytest.mark.parametrize("name,param,new", _identity_cases())
def test_every_constructor_argument_is_in_the_identity(name, param, new):
    if (name, param) in KNOWN_GAPS:
        pytest.skip("a known gap, fixed by an open PR")
    cls = METRICS.get(name)
    kw = {**REQUIRED_ARGS.get(name, {}), **EXTRA.get((name, param), {})}
    try:
        base = cls(**kw)
    except Exception:
        pytest.skip("needs a model or network to construct")
    try:
        changed = cls(**{**kw, param: new})
    except Exception:
        # The mutated value was rejected: try another allowed choice, and fail if there is none.
        alt = CHOICES.get((name, param))
        assert alt is not None, f"{name}({param}={new!r}) is rejected; add an allowed value to CHOICES"
        changed = cls(**{**kw, param: alt})
    assert _key(base) != _key(changed), f"{name}({param}=...) does not change the name or identity()"


# ---- specific edge cases found by the sweep -------------------------------

def test_token_overlap_two_empty_sides_match():
    assert value(build("token_overlap"), "", "") == 1.0
    assert value(build("token_overlap"), "", "paris") == 0.0


def test_perplexity_skips_empty_text_instead_of_scoring_nan():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from auditkit.metrics.generation import Perplexity
    m = Perplexity.__new__(Perplexity)
    m._model_name = "unused"

    class Tok:
        bos_token_id = eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            import torch
            return type("E", (), {"input_ids": torch.zeros((1, 0), dtype=torch.long)})()

    m._model, m._tokenizer = object(), Tok()
    assert m.score(Sample(input="q", target="t"), "") == []
