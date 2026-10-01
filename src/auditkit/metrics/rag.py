"""Deterministic, model-free lexical RAG signals (word overlap heuristics).

These are cheap proxies, not the LLM-judged RAGAS metrics of the same family
(see :mod:`auditkit.metrics.rag_judge` for those). All four share one
tokenizer, :func:`_tokens` (lowercase + strip punctuation), so ``"Paris."``,
``"paris"`` and ``"Paris"`` tokenize identically. The retrieved context is the
trace's ``retrieved_contexts`` (what a RAG pipeline actually retrieved at run
time) when present, else ``Sample.retrieval_context`` -- shared with the ranking
and judge RAG metrics via :func:`auditkit.metrics.retrieval.retrieved_contexts`.
A sample with no retrieved context at all is skipped (no score), not scored 0.
"""

from __future__ import annotations

import re
from typing import Any, Union

from auditkit.metric import Metric
from auditkit.metrics.retrieval import retrieved_contexts
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokens(text: str) -> list[str]:
    """Normalized word tokens shared by every lexical RAG metric: lowercased and
    punctuation-stripped, so ``"Paris."``, ``"paris"`` and ``"Paris"`` all
    tokenize to ``"paris"`` (finding 14)."""
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _covered(token: str, vocabulary: set[str]) -> bool:
    """True when a vocabulary word equals *token* or begins with it (so ``"cat"``
    matches ``"cats"``) -- a word-prefix match on normalized tokens, never a
    substring match (which would match ``"an"`` inside ``"banana"``)."""
    return any(word.startswith(token) for word in vocabulary)


def _context_vocab(contexts: list[str]) -> set[str]:
    vocab: set[str] = set()
    for c in contexts:
        vocab.update(_tokens(c))
    return vocab


@METRICS.register("lexical_groundedness")
class LexicalGroundedness(Metric):
    """Fraction of the output's tokens that are backed by the retrieved context,
    via word-prefix matching (:func:`_covered`) on the shared tokenizer.

    Named for what it computes (a lexical, word-overlap check), not RAGAS's
    ``Faithfulness``, which decomposes the output into claims via an LLM and
    checks NLI entailment. This is a cheaper, unrelated-in-method proxy for the
    same question (is the output grounded in the retrieved context). The
    retrieved context comes from the trace's ``retrieved_contexts`` when present,
    else ``Sample.retrieval_context``; a sample with no context is skipped.
    """

    name = "lexical_groundedness"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE

    def score(self, sample: Sample, output: str, context: Any = None) -> Union[Score, list[Score]]:
        contexts = retrieved_contexts(sample, context)
        if not contexts:
            return []  # nothing retrieved to ground against -> skip, not a fake 0
        tokens = _tokens(output)
        if not tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        vocab = _context_vocab(contexts)
        supported = sum(1 for t in tokens if _covered(t, vocab))
        return Score(name=self.name, value=supported / len(tokens), kind=self.kind)


@METRICS.register("context_coverage")
class ContextCoverage(Metric):
    """Fraction of the target answer's vocabulary present in the retrieved context.

    Deliberately independent of ``output`` -- it measures *retrieval* quality
    (did the retriever bring back enough to construct the correct answer), not
    what the model did with it. Distinct from RAGAS's ``context_recall`` (same
    intent, but a raw vocabulary-overlap heuristic here). ``output`` is accepted
    only to satisfy the :class:`Metric` interface. Needs ``Sample.target``; the
    retrieved context comes from the trace when present, else the sample, and a
    sample with no context is skipped.
    """

    name = "context_coverage"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Union[Score, list[Score]]:
        target_tokens = set(_tokens(sample.target or ""))
        if not target_tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        contexts = retrieved_contexts(sample, context)
        if not contexts:
            return []
        intersection = target_tokens & _context_vocab(contexts)
        return Score(name=self.name, value=len(intersection) / len(target_tokens), kind=self.kind)


@METRICS.register("context_overlap")
class ContextOverlap(Metric):
    """Fraction of the retrieved context chunks that share a token with the output,
    via word-prefix matching (:func:`_covered`) on the shared tokenizer.

    Distinct from RAGAS's ``ContextPrecision`` (same intent, LLM-based there). A
    coarse proxy (any single shared word marks a whole chunk "relevant"). The
    retrieved context comes from the trace's ``retrieved_contexts`` when present,
    else ``Sample.retrieval_context``; a sample with no context is skipped.
    """

    name = "context_overlap"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE

    def score(self, sample: Sample, output: str, context: Any = None) -> Union[Score, list[Score]]:
        contexts = retrieved_contexts(sample, context)
        if not contexts:
            return []
        output_tokens = _tokens(output)
        relevant = sum(1 for ctx in contexts
                       if any(_covered(t, set(_tokens(ctx))) for t in output_tokens))
        return Score(name=self.name, value=relevant / len(contexts), kind=self.kind)


@METRICS.register("answer_overlap")
class AnswerOverlap(Metric):
    """Fraction of the reference answer's vocabulary also present in the output,
    via the shared tokenizer's token-set overlap.

    Distinct from RAGAS's ``ResponseRelevancy`` (which compares the answer back
    to the question via embeddings): this compares the output directly to the
    ``target`` answer's vocabulary.
    """

    name = "answer_overlap"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        output_tokens = set(_tokens(output))
        target_tokens = set(_tokens(sample.target or ""))
        if not target_tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        intersection = output_tokens & target_tokens
        return Score(name=self.name, value=len(intersection) / len(target_tokens), kind=self.kind)
