"""Tests for T2 code/deterministic metrics and weighted scoring."""

from __future__ import annotations


from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.metric import ExactMatch
from auditkit.metrics.code import (
    Equals, Contains, StartsWith, Regex, EndsWith,
    Levenshtein, WordCount, IsJson, F1Score,
)
from auditkit.scoring import ScoreGate, WeightedSum


# ============================================================================
# Equals
# ============================================================================

class TestEquals:
    def test_exact_match(self):
        m = Equals()
        s = Sample(input="q", target="hello")
        assert m.score(s, "hello").value == 1.0

    def test_mismatch(self):
        m = Equals()
        s = Sample(input="q", target="hello")
        assert m.score(s, "world").value == 0.0

    def test_case_insensitive(self):
        m = Equals(ignore_case=True)
        s = Sample(input="q", target="Hello")
        assert m.score(s, "hello").value == 1.0

    def test_case_sensitive_by_default(self):
        m = Equals()
        s = Sample(input="q", target="Hello")
        assert m.score(s, "hello").value == 0.0

    def test_strips_whitespace(self):
        m = Equals()
        s = Sample(input="q", target="hello")
        assert m.score(s, "  hello  ").value == 1.0

    def test_requires_target(self):
        m = Equals()
        assert "target" in m.required_fields
        assert m.applicable(Sample(input="q")) is False

    def test_name(self):
        assert Equals().name == "equals"
        assert Equals(ignore_case=True).name == "equals_ci"


# ============================================================================
# Contains
# ============================================================================

class TestContains:
    def test_contains_substring(self):
        m = Contains("world")
        s = Sample(input="q")
        assert m.score(s, "hello world").value == 1.0

    def test_not_contains(self):
        m = Contains("world")
        s = Sample(input="q")
        assert m.score(s, "hello there").value == 0.0

    def test_case_insensitive(self):
        m = Contains("WORLD", ignore_case=True)
        s = Sample(input="q")
        assert m.score(s, "hello World").value == 1.0

    def test_case_sensitive_by_default(self):
        m = Contains("WORLD")
        s = Sample(input="q")
        assert m.score(s, "hello world").value == 0.0

    def test_name_contains_substring(self):
        assert Contains("x").name == "contains(x)"
        assert Contains("x", ignore_case=True).name == "contains_ci(x)"

    def test_requires_no_fields(self):
        m = Contains("x")
        assert m.applicable(Sample(input="q")) is True


# ============================================================================
# StartsWith
# ============================================================================

class TestStartsWith:
    def test_starts_with(self):
        m = StartsWith("hello")
        s = Sample(input="q")
        assert m.score(s, "hello world").value == 1.0

    def test_not_starts_with(self):
        m = StartsWith("hello")
        s = Sample(input="q")
        assert m.score(s, "world hello").value == 0.0

    def test_case_insensitive(self):
        m = StartsWith("HELLO", ignore_case=True)
        s = Sample(input="q")
        assert m.score(s, "Hello world").value == 1.0

    def test_name(self):
        assert StartsWith("x").name == "startswith(x)"
        assert StartsWith("x", ignore_case=True).name == "startswith_ci(x)"


# ============================================================================
# EndsWith
# ============================================================================

class TestEndsWith:
    def test_ends_with(self):
        m = EndsWith("world")
        s = Sample(input="q")
        assert m.score(s, "hello world").value == 1.0

    def test_not_ends_with(self):
        m = EndsWith("world")
        s = Sample(input="q")
        assert m.score(s, "world hello").value == 0.0

    def test_case_insensitive(self):
        m = EndsWith("WORLD", ignore_case=True)
        assert m.score(Sample(input="q"), "hello World").value == 1.0

    def test_name(self):
        assert EndsWith("x").name == "endswith(x)"


# ============================================================================
# Regex
# ============================================================================

class TestRegex:
    def test_regex_match(self):
        m = Regex(r"\d{3}-\d{4}")
        s = Sample(input="q")
        assert m.score(s, "Call 555-1234 now").value == 1.0

    def test_no_match(self):
        m = Regex(r"\d{3}-\d{4}")
        s = Sample(input="q")
        assert m.score(s, "No digits here").value == 0.0

    def test_full_match_required(self):
        m = Regex(r"^\d+$")
        s = Sample(input="q")
        assert m.score(s, "12345").value == 1.0
        assert m.score(s, "abc").value == 0.0

    def test_name(self):
        assert Regex(r"\d+").name == "regex(\\d+)"

    def test_requires_no_fields(self):
        m = Regex(r".+")
        assert m.applicable(Sample(input="q")) is True


# ============================================================================
# Levenshtein (edit distance, inverted to [0,1])
# ============================================================================

class TestLevenshtein:
    def test_identical_is_1(self):
        m = Levenshtein()
        s = Sample(input="q", target="hello")
        assert m.score(s, "hello").value == 1.0

    def test_completely_different_is_0(self):
        m = Levenshtein()
        s = Sample(input="q", target="hello")
        assert m.score(s, "").value == 0.0  # empty vs non-empty

    def test_partial_match(self):
        m = Levenshtein()
        s = Sample(input="q", target="kitten")
        # "kitten" vs "sitting" = 3 edits, max_len = 7
        score = m.score(s, "sitting")
        assert 0.0 < score.value < 1.0
        assert isinstance(score, Score)

    def test_name(self):
        assert Levenshtein().name == "levenshtein"


# ============================================================================
# WordCount
# ============================================================================

class TestWordCount:
    def test_within_range(self):
        m = WordCount(min_words=1, max_words=5)
        s = Sample(input="q")
        assert m.score(s, "one two three").value == 1.0

    def test_too_few(self):
        m = WordCount(min_words=3)
        s = Sample(input="q")
        assert m.score(s, "one two").value == 0.0

    def test_too_many(self):
        m = WordCount(max_words=3)
        s = Sample(input="q")
        assert m.score(s, "one two three four").value == 0.0

    def test_exact(self):
        m = WordCount(min_words=3, max_words=3)
        s = Sample(input="q")
        assert m.score(s, "one two three").value == 1.0
        assert m.score(s, "one two").value == 0.0

    def test_empty(self):
        m = WordCount(min_words=0, max_words=5)
        s = Sample(input="q")
        assert m.score(s, "").value == 1.0

    def test_name(self):
        assert WordCount(min_words=2).name == "wordcount(2-)"
        assert WordCount(max_words=5).name == "wordcount(-5)"
        assert WordCount(min_words=2, max_words=5).name == "wordcount(2-5)"


# ============================================================================
# IsJson
# ============================================================================

class TestIsJson:
    def test_valid_json_object(self):
        m = IsJson()
        s = Sample(input="q")
        assert m.score(s, '{"a": 1, "b": 2}').value == 1.0

    def test_valid_json_array(self):
        m = IsJson()
        assert m.score(Sample(input="q"), "[1, 2, 3]").value == 1.0

    def test_invalid_json(self):
        m = IsJson()
        assert m.score(Sample(input="q"), "not json").value == 0.0

    def test_validates_schema_keys(self):
        m = IsJson(require_keys=["name", "age"])
        s = Sample(input="q")
        assert m.score(s, '{"name": "Alice", "age": 30}').value == 1.0
        assert m.score(s, '{"name": "Alice"}').value == 0.0

    def test_name(self):
        assert IsJson().name == "is_json"
        assert IsJson(require_keys=["x"]).name == "is_json(keys=x)"

    def test_requires_no_fields(self):
        m = IsJson()
        assert m.applicable(Sample(input="q")) is True


# ============================================================================
# F1Score (token-overlap F1)
# ============================================================================

class TestF1Score:
    def test_exact_match_is_1(self):
        m = F1Score()
        s = Sample(input="q", target="the cat sat")
        assert m.score(s, "the cat sat").value == 1.0

    def test_no_overlap_is_0(self):
        m = F1Score()
        s = Sample(input="q", target="the cat sat")
        assert m.score(s, "dog ran").value == 0.0

    def test_partial_overlap(self):
        m = F1Score()
        s = Sample(input="q", target="the cat sat")
        score = m.score(s, "the dog sat")
        assert 0.0 < score.value < 1.0

    def test_precision_and_recall(self):
        m = F1Score()
        s = Sample(input="q", target="a b c")
        score = m.score(s, "a b")
        assert score.value > 0.0

    def test_requires_target(self):
        m = F1Score()
        assert "target" in m.required_fields
        assert m.applicable(Sample(input="q")) is False

    def test_name(self):
        assert F1Score().name == "f1_score"


# ============================================================================
# ScoreGate (weighted metric with threshold)
# ============================================================================

class TestScoreGate:
    def test_gate_delegates_to_inner_metric(self):
        inner = ExactMatch()
        gate = ScoreGate(metric=inner, weight=2.0, threshold=0.5)
        s = Sample(input="q", target="ok")
        scores = gate.score(s, "ok")
        assert len(scores) == 1
        assert scores[0].value == 1.0
        assert scores[0].weight == 2.0

    def test_gate_sets_threshold(self):
        inner = ExactMatch()
        gate = ScoreGate(metric=inner, threshold=0.5)
        s = Sample(input="q", target="ok")
        scores = gate.score(s, "wrong")
        assert scores[0].passed is False

    def test_gate_required_fields_from_inner(self):
        inner = ExactMatch()
        gate = ScoreGate(metric=inner)
        assert "target" in gate.required_fields
        assert gate.applicable(Sample(input="q")) is False
        assert gate.applicable(Sample(input="q", target="a")) is True

    def test_gate_name_includes_inner(self):
        gate = ScoreGate(metric=ExactMatch())
        assert "exact_match" in gate.name


# ============================================================================
# WeightedSum
# ============================================================================

class TestWeightedSum:
    def test_weighted_sum_computes_correctly(self):
        ws = WeightedSum()
        scores = [
            Score(name="a", value=1.0, weight=1.0),
            Score(name="b", value=0.0, weight=3.0),
        ]
        result = ws.compute(scores)
        assert result.name == "weighted_sum"
        assert result.value == 0.25  # (1*1 + 0*3) / (1+3) = 0.25

    def test_equal_weight_is_mean(self):
        ws = WeightedSum()
        scores = [
            Score(name="a", value=1.0, weight=1.0),
            Score(name="b", value=0.5, weight=1.0),
        ]
        result = ws.compute(scores)
        assert result.value == 0.75

    def test_empty_scores(self):
        ws = WeightedSum()
        result = ws.compute([])
        assert result.value == 0.0

    def test_name_customizable(self):
        ws = WeightedSum(name="overall")
        scores = [Score(name="a", value=1.0)]
        assert ws.compute(scores).name == "overall"
