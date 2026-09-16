"""Tests for embedding similarity metrics."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.metrics.embedding import CosineSimilarity, TokenOverlap, BM25Similarity
from auditkit.errors import ExtraNotInstalled


S = Sample(input="x", target="the cat sat on the mat")


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


NEEDS_TRANSFORMERS = pytest.mark.skipif(
    not _importable("transformers"), reason="transformers not installed"
)


class TestCosineSimilarity:
    def test_extra_not_installed(self):
        if _importable("transformers"):
            pytest.skip("transformers is installed in this environment -- nothing to assert")
        with pytest.raises(ExtraNotInstalled):
            CosineSimilarity().score(S, "the cat sat on the mat")

    def test_name(self):
        assert CosineSimilarity().name == "cosine_similarity"

    def test_no_sentence_transformers_dependency(self):
        """The whole point of this rewrite: no sentence-transformers import
        anywhere in this metric, so it can never trigger sentence-
        transformers' real, live-confirmed transitive torchcodec/FFmpeg
        failure on Colab (unrelated to text embeddings entirely). Checks
        for a real `import sentence_transformers` statement specifically,
        not just any mention of the name (the module's own docstring
        explains the removal by name, which would otherwise false-fail
        a bare substring check)."""
        import ast
        import inspect
        from auditkit.metrics import embedding
        tree = ast.parse(inspect.getsource(embedding))
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        assert "sentence_transformers" not in imported_names

    @NEEDS_TRANSFORMERS
    def test_identical_text_scores_near_one(self):
        s = Sample(input="q", target="The capital of France is Paris.")
        score = CosineSimilarity().score(s, "The capital of France is Paris.")
        assert score.value > 0.99

    @NEEDS_TRANSFORMERS
    def test_real_paraphrase_scores_high(self):
        s = Sample(input="q", target="The capital of France is Paris.")
        score = CosineSimilarity().score(s, "Paris is the capital city of France.")
        assert score.value > 0.9

    @NEEDS_TRANSFORMERS
    def test_unrelated_text_scores_low(self):
        s = Sample(input="q", target="The capital of France is Paris.")
        identical = CosineSimilarity().score(s, "The capital of France is Paris.")
        unrelated = CosineSimilarity().score(s, "The stock market fell sharply today.")
        assert unrelated.value < identical.value
        assert unrelated.value < 0.3

    @NEEDS_TRANSFORMERS
    def test_model_reused_across_calls_not_reloaded(self):
        m = CosineSimilarity()
        s = Sample(input="q", target="hello")
        m.score(s, "hello")
        loaded_model = m._model
        m.score(s, "world")
        assert m._model is loaded_model


class TestTokenOverlap:
    def test_identical(self):
        assert TokenOverlap().score(S, "the cat sat on the mat").value == 1.0
    def test_no_overlap(self):
        assert TokenOverlap().score(S, "xyz abc").value == 0.0
    def test_partial(self):
        v = TokenOverlap().score(S, "the cat").value
        assert 0.0 < v < 1.0
    def test_name(self):
        assert TokenOverlap().name == "token_overlap"


class TestBM25Similarity:
    def test_identical(self):
        assert BM25Similarity().score(S, "the cat sat on the mat").value == 1.0
    def test_no_match(self):
        assert BM25Similarity().score(S, "xyz").value == 0.0
    def test_partial(self):
        v = BM25Similarity().score(S, "the cat sat").value
        assert 0.0 < v < 1.0
    def test_name(self):
        assert BM25Similarity().name == "bm25_similarity"
