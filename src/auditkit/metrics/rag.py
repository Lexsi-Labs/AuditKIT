from __future__ import annotations

import re
from typing import Any

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind

_PUNCT_RE = re.compile(r"[^\w\s]")


def _word_prefix_match(needle: str, haystack: str) -> bool:
    """True iff *needle* appears in *haystack* starting at a word boundary
    (so it matches as a whole word or as the start of a longer word, e.g.
    "cat" against "cats") -- but never mid-word, unlike plain substring
    containment (e.g. "an" against "banana", which starts one character
    into the word, not at a boundary).
    """
    return re.search(rf"\b{re.escape(needle)}", haystack) is not None


@METRICS.register("lexical_groundedness")
class LexicalGroundedness(Metric):
    """Fraction of the output's words that are backed by the retrieved
    context, via word-boundary matching -- not substring containment.

    Named for what it actually computes (a lexical, word-overlap check),
    not RAGAS's ``Faithfulness``: RAGAS decomposes the output into claims
    via an LLM and checks each claim's NLI entailment against the context;
    this metric checks raw word-boundary presence instead, a cheaper,
    unrelated-in-method signal that happens to answer a similar question
    (is the output grounded in the retrieved context).

    A prior version checked ``claim in ctx`` with plain Python substring
    containment, so a short output word could match as a substring buried
    inside an unrelated context word (e.g. "an" matching inside "banana").
    Fixed to match only at a word boundary (:func:`_word_prefix_match`),
    which still allows a benign prefix relationship (e.g. "cat" matching
    the start of "cats") without matching arbitrary mid-word substrings.
    """

    name = "lexical_groundedness"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"retrieval_context"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        claims = output.split()
        retrieval_context = sample.retrieval_context or []
        if not claims or not retrieval_context:
            return Score(name=self.name, value=0.0, kind=self.kind)
        supported = sum(
            1 for claim in claims
            if any(_word_prefix_match(claim, ctx) for ctx in retrieval_context)
        )
        return Score(name=self.name, value=supported / len(claims), kind=self.kind)


@METRICS.register("context_coverage")
class ContextCoverage(Metric):
    """Fraction of the target answer's vocabulary present somewhere in the
    retrieved context.

    Deliberately independent of ``output`` -- this measures *retrieval*
    quality (did the retriever bring back enough material to construct the
    correct answer at all), not what the model actually did with it. Named
    distinctly from RAGAS's ``context_recall`` (same scoping intent, but a
    raw vocabulary-overlap heuristic here, not RAGAS's LLM-based
    claim-decomposition method). ``output`` is accepted only to satisfy the
    :class:`Metric` interface; see :class:`LexicalGroundedness` (checks the
    output's claims against context) and :class:`AnswerOverlap` (checks the
    output against the target) for output-dependent RAG checks.
    """

    name = "context_coverage"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"retrieval_context", "target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        target_tokens = set((sample.target or "").split())
        if not target_tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        context_tokens: set[str] = set()
        for ctx in (sample.retrieval_context or []):
            context_tokens.update(ctx.split())
        intersection = target_tokens & context_tokens
        return Score(name=self.name, value=len(intersection) / len(target_tokens), kind=self.kind)


@METRICS.register("context_overlap")
class ContextOverlap(Metric):
    """Fraction of the retrieved context chunks that share a word with the
    output, via word-boundary matching -- not substring containment.

    Named distinctly from RAGAS's ``ContextPrecision`` (same scoping
    intent, but a raw word-overlap heuristic here, not RAGAS's
    LLM-based method). A prior version matched raw substrings (a single
    output token like ``"a"`` matching inside almost any context string),
    inflating precision. Fixed to word-boundary matching
    (:func:`_word_prefix_match`, shared with :class:`LexicalGroundedness`).
    Still a coarse proxy (any single shared word counts a whole context
    chunk as "relevant"), but no longer a false-positive-prone substring
    bug.
    """

    name = "context_overlap"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"retrieval_context"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        contexts = sample.retrieval_context or []
        if not contexts:
            return Score(name=self.name, value=0.0, kind=self.kind)
        output_tokens = output.split()
        relevant = sum(
            1 for ctx in contexts
            if any(_word_prefix_match(tok, ctx) for tok in output_tokens)
        )
        return Score(name=self.name, value=relevant / len(contexts), kind=self.kind)


@METRICS.register("answer_overlap")
class AnswerOverlap(Metric):
    """Fraction of the real reference answer's vocabulary also present in
    the output, via raw token-set overlap.

    Named distinctly from RAGAS's ``ResponseRelevancy`` (which compares the
    answer back to the *question* via embedding similarity): this metric
    compares the output directly to the *target* answer's vocabulary, a
    different and more literal comparison than the name it replaces implied.
    """

    name = "answer_overlap"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        output_tokens = set(_PUNCT_RE.sub(" ", output).split())
        target_tokens = set(_PUNCT_RE.sub(" ", sample.target or "").split())
        if not target_tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        intersection = output_tokens & target_tokens
        return Score(name=self.name, value=len(intersection) / len(target_tokens), kind=self.kind)
