"""Embedding-based similarity metrics."""
from __future__ import annotations

from typing import Any

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.errors import ExtraNotInstalled


@METRICS.register("cosine_similarity")
class CosineSimilarity(Metric):
    """Embedding cosine similarity via plain ``transformers`` -- no
    ``sentence-transformers`` dependency.

    A previous version of this metric used the ``sentence-transformers``
    package, which transitively imports ``transformers``' audio/video
    processing modules and, in turn, ``torchcodec`` -- a real, live-
    confirmed environment failure on Colab (``torch``/``torchcodec``
    version mismatch, unrelated ``libavutil``/FFmpeg shared libraries
    missing) that has nothing to do with text embeddings at all and took
    this metric down with it. ``sentence-transformers`` was used for
    exactly this one metric in the whole library (confirmed: no other
    file imports it), so removing it entirely and reimplementing the
    same computation directly on ``transformers.AutoModel`` -- already a
    core dependency for :class:`EncoderJudge`/``HFGenModel``, and
    confirmed *not* to trigger the ``torchcodec`` import path -- avoids
    the whole problem rather than working around it.

    Mean-pooling + L2-normalization is the exact recipe
    ``sentence-transformers`` itself uses internally for MiniLM-family
    checkpoints (documented on the model card) -- this reproduces the
    same embeddings, not an approximation.
    """

    name = "cosine_similarity"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = False
    required_fields = frozenset({"target"})

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._torch = None

    def identity(self) -> dict:
        return {"name": self.name, "model_name": self._model_name}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ExtraNotInstalled("transformers", "pip install auditkit[transformers]")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()

    def _embed(self, text: str) -> Any:
        torch = self._torch
        inputs = self._tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs)
        # Mean pooling over real (non-padding) token embeddings -- the
        # attention mask zeroes out padding positions before averaging,
        # exactly as sentence-transformers does for this checkpoint family.
        token_embeddings = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts)[0]
        norm = embedding.norm(p=2)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        self._ensure_model()
        emb_out = self._embed(output)
        emb_tgt = self._embed(sample.target or "")
        similarity = float(self._torch.dot(emb_out, emb_tgt).item())
        return Score(name=self.name, value=similarity, kind=self.kind)


@METRICS.register("token_overlap")
class TokenOverlap(Metric):
    name = "token_overlap"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        tokens_out = set(output.split())
        tokens_tgt = set((sample.target or "").split())
        if not tokens_out and not tokens_tgt:
            return Score(name=self.name, value=0.0, kind=self.kind)
        intersection = tokens_out & tokens_tgt
        union = tokens_out | tokens_tgt
        return Score(name=self.name, value=len(intersection) / len(union), kind=self.kind)


@METRICS.register("bm25_similarity")
class BM25Similarity(Metric):
    name = "bm25_similarity"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        tokens_out = output.split()
        tokens_tgt = (sample.target or "").split()
        freq_out: dict[str, int] = {}
        for t in tokens_out:
            freq_out[t] = freq_out.get(t, 0) + 1
        freq_tgt: dict[str, int] = {}
        for t in tokens_tgt:
            freq_tgt[t] = freq_tgt.get(t, 0) + 1
        all_tokens = set(freq_out) | set(freq_tgt)
        score_val = sum(min(freq_out.get(t, 0), freq_tgt.get(t, 0)) for t in all_tokens)
        max_len = max(len(tokens_out), len(tokens_tgt), 1)
        return Score(name=self.name, value=score_val / max_len, kind=self.kind)
