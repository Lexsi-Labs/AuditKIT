"""LLM-as-judge via a real encoder classifier — classification, not generation.

Every judge in ``metrics/judge.py`` (``LLMJudge``, ``GEval``, ...) needs a
*generative* model: it asks a second LLM to produce free text, then parses a
verdict out of that text with a regex/marker. :class:`EncoderJudge` is a
genuinely different mechanism: any `AutoModelForSequenceClassification`
encoder — BERT, RoBERTa, DeBERTa, ALBERT, ELECTRA, XLNet, MPNet, and any
other architecture Hugging Face's ``Auto*`` classes cover — used directly as
the judge. It never generates anything; it classifies a (candidate,
reference) pair in one forward pass and reads the verdict straight off the
model's own output distribution, with no free-text parsing at all.

**Templating for an encoder** works differently than a chat/prompt template:
encoder classifiers built for pair-input tasks (NLI, similarity, cross-encoder
ranking, ...) take exactly *two* text spans and encode them together via the
tokenizer's native pair-sequence support (``tokenizer(text, text_pair)`` →
``[CLS] text [SEP] text_pair [SEP]`` for BERT-style models, with an
equivalent for every other encoder architecture) — there is no chat-template
string to write by hand. ``text_template=``/``text_pair_template=`` here
just decide *which rendered fields* fill those two slots; the actual
pair-encoding is handled by ``transformers``' own pipeline, not by manually
concatenating a literal ``"[SEP]"`` into one string (which does not
reliably match what any given tokenizer's real special-token handling would
produce — see ``FactualConsistency`` in ``hallucination.py`` for the
older, less reliable version of that mistake).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from auditkit.metric import Metric
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.registry import METRICS
from auditkit.errors import ExtraNotInstalled

# Label names commonly used across NLI/entailment/similarity-style encoder
# classifiers, for auto-detecting a sensible label_map when the checkpoint's
# own config.id2label uses real words. Deliberately conservative -- a
# checkpoint with meaningless labels (LABEL_0/LABEL_1/LABEL_2, the actual
# default for a lot of fine-tuned classifiers) can't be guessed at all, and
# EncoderJudge raises rather than fabricating a mapping in that case.
_POSITIVE_LABEL_NAMES = frozenset({
    "entailment", "entail", "positive", "pos", "yes", "true", "consistent",
    "supported", "match", "duplicate", "similar", "1", "label_1",
})
_NEGATIVE_LABEL_NAMES = frozenset({
    "contradiction", "contradict", "negative", "neg", "no", "false",
    "inconsistent", "unsupported", "mismatch", "not_duplicate",
    "dissimilar", "0", "label_0",
})
_NEUTRAL_LABEL_NAMES = frozenset({"neutral", "unrelated"})


def _resolve_device(device: Optional[str]) -> int | str:
    """Best available device for transformers.pipeline(device=...), which
    wants an int (-1=cpu, 0=cuda:0, ...) or a device string -- unlike
    HFGenModel's backend, which takes a plain device string throughout.
    """
    if device is not None:
        return device
    import torch
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


@METRICS.register("encoder_judge")
class EncoderJudge(Metric):
    """LLM-as-judge backed by a real encoder classifier, not a prompted LLM.

    Works with **any** `AutoModelForSequenceClassification`-compatible
    checkpoint — BERT, RoBERTa, DeBERTa, ALBERT, ELECTRA, XLNet, MPNet, and
    so on; genericity comes for free from ``transformers``' own ``Auto*``
    classes, no architecture-specific code here.

    Parameters
    ----------
    model_name
        Any encoder sequence-classification checkpoint
        (``"microsoft/deberta-base-mnli"``, ``"roberta-large-mnli"``,
        ``"textattack/bert-base-uncased-MNLI"``, a fine-tuned similarity/
        toxicity/sentiment classifier, ...).
    text_template, text_pair_template
        Placeholder templates (``{input}``/``{output}``/``{target}``,
        alias ``{expected}``/``{context}``, same field set as
        :class:`~auditkit.metrics.judge.LLMJudge`) rendered into the two
        text spans handed to the tokenizer's real pair-sequence encoding.
        Default: candidate output as the first span, reference target as
        the second (the standard NLI-as-judge setup — "does the output
        entail the target"). Set ``text_pair_template=None`` for a
        single-sequence classifier (e.g. a standalone toxicity/quality
        classifier with no reference to compare against).
    label_map
        ``{label_name_or_index: score}`` — maps the checkpoint's own
        output labels to a 0–1 score. Keys match case-insensitively against
        ``model.config.id2label`` values, or against the numeric index
        (as ``int`` or the literal string a generic checkpoint uses, e.g.
        ``"LABEL_1"``). If omitted, tries to auto-detect a sensible
        mapping from common NLI/sentiment-style label names (entailment/
        positive/yes → 1.0, contradiction/negative/no → 0.0, neutral →
        0.5) — **raises** rather than guessing when the checkpoint's
        labels aren't recognizable (e.g. bare ``LABEL_0``/``LABEL_1``),
        since fabricating a mapping for unknown labels would silently
        misjudge every single sample.
    aggregation
        ``"probability"`` (default): the score is the softmax-weighted sum
        of every label's probability times its mapped score — a
        continuous value reflecting the classifier's actual confidence,
        not just its single best guess (e.g. an entailment probability of
        0.85 scores 0.85, not a rounded 1.0). ``"argmax"``: score is just
        the mapped value of the single highest-probability label (a
        cruder, binary-flavored pass/fail).
    device
        ``None`` (default) auto-detects CUDA/MPS/CPU, same as every other
        real local backend in this repo. Pass an explicit
        ``transformers.pipeline``-style device otherwise.

    The returned :class:`Score` always carries ``reason`` (the predicted
    label + its probability) and ``metadata`` (the full label→probability
    distribution), so you can inspect exactly what the classifier saw and
    decided, the same transparency :class:`~auditkit.metrics.judge.LLMJudge`
    gives via its raw judge-model reply.

    **Two real gotchas, confirmed live across multiple checkpoints:**

    - **A generic checkpoint's label *order* is not standardized, even for
      the nominally same task.** Two real MNLI-finetuned checkpoints tested
      against each other disagreed: ``howey/electra-base-mnli`` uses
      ``LABEL_0=entailment, LABEL_1=neutral, LABEL_2=contradiction``, while
      a naively-assumed "standard" order (0=contradiction, 1=entailment,
      2=neutral, matching several other real checkpoints) is exactly
      backwards for it. There is no way to know a generic checkpoint's real
      label semantics without checking — this is exactly why
      auto-detection refuses to guess for labels like ``LABEL_0``/
      ``LABEL_1`` rather than assuming any particular index convention.
      Determine the real order empirically (a couple of known
      obviously-true/obviously-false example pairs, checking which index
      lights up) before trusting a ``label_map`` you haven't verified.
    - **``aggregation="probability"`` assumes single-label, mutually-
      exclusive classification** (NLI/entailment, sentiment — probabilities
      across all labels sum to ~1, so the weighted sum is a genuine soft
      interpolation between verdicts). For an independent **multi-label**
      classifier (e.g. `unitary/toxic-bert`, sigmoid outputs, several
      labels can be true at once), mapping a label to ``0.0`` means "this
      label contributes nothing to the score" — **not** "the complement of
      this label." There is no way to express "1 − P(toxic)" through the
      weighted-sum formula alone; if you want that framing specifically,
      use a dedicated metric built for it (e.g. this repo's own
      `ToxicityScore`) rather than `EncoderJudge`.
    """

    name = "encoder_judge"
    kind = ScoreKind.JUDGE
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(
        self,
        model_name: str = "microsoft/deberta-base-mnli",
        *,
        text_template: str = "{output}",
        text_pair_template: Optional[str] = "{target}",
        label_map: Optional[dict[Any, float]] = None,
        aggregation: Literal["probability", "argmax"] = "probability",
        device: Optional[str] = None,
        name: str = "encoder_judge",
    ) -> None:
        self._model_name = model_name
        self.text_template = text_template
        self.text_pair_template = text_pair_template
        self._label_map_arg = dict(label_map) if label_map else None
        if aggregation not in ("probability", "argmax"):
            raise ValueError(f"EncoderJudge: aggregation must be 'probability' or 'argmax', got {aggregation!r}")
        self.aggregation = aggregation
        self._device = device
        self.name = name
        self._pipe = None
        self._label_map: Optional[dict[str, float]] = None  # resolved lazily, needs id2label

    def identity(self) -> dict:
        return {
            "name": self.name,
            "model_name": self._model_name,
            "text_template": self.text_template,
            "text_pair_template": self.text_pair_template,
            "label_map": self._label_map_arg,
            "aggregation": self.aggregation,
        }

    # -- model / label-map resolution --------------------------------------

    def _ensure_pipeline(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            raise ExtraNotInstalled("transformers", "pip install auditkit[transformers]") from None
        self._pipe = pipeline(
            "text-classification",
            model=self._model_name,
            device=_resolve_device(self._device),
            top_k=None,  # return every label's probability, not just the top one
        )
        self._label_map = self._resolve_label_map()

    def _resolve_label_map(self) -> dict[str, float]:
        id2label: dict[int, str] = dict(self._pipe.model.config.id2label)
        if self._label_map_arg is not None:
            return self._normalize_explicit_label_map(self._label_map_arg, id2label)
        return self._auto_detect_label_map(id2label)

    @staticmethod
    def _normalize_explicit_label_map(raw: dict[Any, float], id2label: dict[int, str]) -> dict[str, float]:
        """Resolve a user-supplied label_map (keyed by label name, case-
        insensitive, or by numeric index) against the real label strings
        the pipeline will actually emit."""
        by_lower_name = {v.lower(): v for v in id2label.values()}
        resolved: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(key, int) and key in id2label:
                resolved[id2label[key]] = value
                continue
            key_str = str(key).lower()
            if key_str in by_lower_name:
                resolved[by_lower_name[key_str]] = value
                continue
            # Bare numeric string ("0", "1") against index-derived generic
            # labels (LABEL_0, LABEL_1) that id2label itself may already use.
            if key_str.isdigit() and int(key_str) in id2label:
                resolved[id2label[int(key_str)]] = value
                continue
            raise ValueError(
                f"EncoderJudge: label_map key {key!r} doesn't match any of this "
                f"checkpoint's real labels {list(id2label.values())!r} (by name, "
                f"case-insensitive, or by index)."
            )
        return resolved

    @staticmethod
    def _auto_detect_label_map(id2label: dict[int, str]) -> dict[str, float]:
        detected: dict[str, float] = {}
        for label in id2label.values():
            norm = label.lower()
            if norm in _POSITIVE_LABEL_NAMES:
                detected[label] = 1.0
            elif norm in _NEGATIVE_LABEL_NAMES:
                detected[label] = 0.0
            elif norm in _NEUTRAL_LABEL_NAMES:
                detected[label] = 0.5
        if len(detected) < len(id2label):
            raise ValueError(
                f"EncoderJudge: couldn't auto-detect a label_map for this "
                f"checkpoint's labels {list(id2label.values())!r} -- pass "
                f"label_map= explicitly, e.g. label_map={{{list(id2label.values())[0]!r}: 1.0, ...}}. "
                f"Auto-detection only recognizes common NLI/sentiment-style "
                f"names (entailment/positive/yes -> 1.0, contradiction/"
                f"negative/no -> 0.0, neutral -> 0.5); a checkpoint with "
                f"generic labels (LABEL_0/LABEL_1/...) can't be guessed "
                f"without fabricating what they mean."
            )
        return detected

    # -- templating ----------------------------------------------------------

    def _render(self, template: str, sample: Sample, output: str) -> str:
        fields = {
            "input": getattr(sample, "input_text", None) or str(sample.input),
            "output": output,
            "expected": sample.target or "",
            "target": sample.target or "",
            "context": "\n".join(sample.retrieval_context or []),
        }
        for k, v in (sample.metadata or {}).items():
            fields.setdefault(k, v)
        text = template
        for key, val in fields.items():
            text = text.replace("{" + key + "}", str(val))
        return text

    # -- Metric API ------------------------------------------------------------

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        self._ensure_pipeline()
        text = self._render(self.text_template, sample, output)
        if self.text_pair_template is not None:
            text_pair = self._render(self.text_pair_template, sample, output)
            pipeline_input = {"text": text, "text_pair": text_pair}
        else:
            pipeline_input = text
        # truncation=True is required, not optional: BERT/RoBERTa/ELECTRA-
        # style checkpoints use absolute position embeddings with a hard
        # 512-token limit and crash outright (RuntimeError: tensor size
        # mismatch) on longer combined input -- confirmed live with two
        # long real paragraphs on a BERT checkpoint. DeBERTa (relative
        # position embeddings, no hard limit -- its tokenizer even reports
        # model_max_length as the classic "unbounded" sentinel value)
        # doesn't strictly need this, but truncation=True is confirmed
        # harmless for it too.
        results = self._pipe(pipeline_input, truncation=True)
        # transformers' pipeline has an inconsistent return shape depending
        # on input form: a bare string is treated as "a batch of one" and
        # comes back nested ([[{label,score}, ...]]), while a single
        # {"text":..., "text_pair":...} dict comes back flat
        # ([{label,score}, ...]) -- confirmed live, both forms used here
        # depending on whether text_pair_template is set. Normalize either
        # way to a flat list of per-label dicts.
        if results and isinstance(results[0], list):
            results = results[0]
        probs = {r["label"]: r["score"] for r in results}
        predicted_label = max(results, key=lambda r: r["score"])["label"]
        predicted_prob = probs[predicted_label]

        if self.aggregation == "argmax":
            value = self._label_map.get(predicted_label, 0.0)
        else:
            value = sum(probs.get(label, 0.0) * mapped for label, mapped in self._label_map.items())

        return Score(
            name=self.name,
            value=value,
            kind=self.kind,
            reason=f"predicted={predicted_label!r} probability={predicted_prob:.3f} aggregation={self.aggregation}",
            metadata={"probabilities": probs, "predicted_label": predicted_label},
        )


# ---------------------------------------------------------------------------
# Prebuilt judges — ready-to-use EncoderJudge subclasses for 6 real,
# live-verified checkpoints spanning 5 architecture families (see
# docs/encoder_judge.md's checkpoint table for the full verification
# detail). Every constructor argument still has the exact same meaning and
# is fully overridable -- these only supply the *correct, checkpoint-
# specific defaults* (model_name, label_map, template shape) so using one
# doesn't require knowing that checkpoint's real label semantics up front,
# mirroring how Factuality/ClosedQA/Relevance are LLMJudge with a
# pre-filled prompt/choices, not a different mechanism.
# ---------------------------------------------------------------------------

@METRICS.register("factuality_encoder_judge")
class FactualityEncoderJudge(EncoderJudge):
    """Factual-consistency-as-judge via DeBERTa (v1) --
    ``microsoft/deberta-base-mnli``. Real label names
    (ENTAILMENT/NEUTRAL/CONTRADICTION); label_map is auto-detected
    correctly with no configuration needed.

    The one kept from what was originally three near-identical NLI
    prebuilts (DeBERTa v1/v3, RoBERTa) -- all three used checkpoints with
    real, auto-detectable labels, so they added a class name but no real
    behavior difference over passing ``model_name=`` directly. DeBERTa v1
    is the strongest of the three in practice: relative position
    embeddings (no hard input-length limit, unlike RoBERTa/BERT/ELECTRA's
    512-token absolute limit) and empirically more confident/discriminative
    than DeBERTa-v3 on ambiguous pairs (see the removed DebertaV3NLIJudge's
    old docstring: 0.33 vs ~0.001 for the same "unrelated sentences" test).
    Named for its actual use case (checking whether a claim/answer is
    factually consistent with a reference), mirroring ``Factuality`` on
    ``LLMJudge`` -- same use case, two different judge mechanisms."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-base-mnli",
        *,
        text_template: str = "{output}",
        text_pair_template: Optional[str] = "{target}",
        label_map: Optional[dict[Any, float]] = None,
        aggregation: Literal["probability", "argmax"] = "probability",
        device: Optional[str] = None,
        name: str = "factuality_encoder_judge",
    ) -> None:
        super().__init__(
            model_name=model_name, text_template=text_template, text_pair_template=text_pair_template,
            label_map=label_map, aggregation=aggregation, device=device, name=name,
        )


@METRICS.register("sentiment_encoder_judge")
class SentimentEncoderJudge(EncoderJudge):
    """Sentiment-as-judge via DistilBERT --
    ``distilbert-base-uncased-finetuned-sst-2-english``. The only
    remaining prebuilt with a genuinely different *task* from
    :class:`FactualityEncoderJudge` (2-way sentiment, not NLI entailment)
    and single-sequence (no reference needed) rather than pair mode --
    demonstrates ``text_pair_template=None`` as a prebuilt, not just a
    base-class option. Judges ``output`` alone; ``target`` is unused.

    Originally shipped alongside ``BertNLIJudge``/``ElectraNLIJudge``,
    both since removed: both did the exact same task as
    ``FactualityEncoderJudge`` (entailment/factual-consistency) on
    checkpoints whose only distinguishing feature was an unrecoverable,
    empirically-verified generic label order (``textattack/bert-base-
    uncased-MNLI``: ``LABEL_0=contradiction, LABEL_1=entailment,
    LABEL_2=neutral``; ``howey/electra-base-mnli``: the OPPOSITE order,
    ``LABEL_0=entailment, LABEL_1=neutral, LABEL_2=contradiction``) --
    real, checkpoint-specific knowledge worth documenting (see
    ``docs/encoder_judge.md``), but not a distinct enough task to justify
    two more classes with no behavior difference from
    ``FactualityEncoderJudge`` beyond which checkpoint they pin."""

    required_fields = frozenset()

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        *,
        text_template: str = "{output}",
        text_pair_template: Optional[str] = None,
        label_map: Optional[dict[Any, float]] = None,
        aggregation: Literal["probability", "argmax"] = "probability",
        device: Optional[str] = None,
        name: str = "sentiment_encoder_judge",
    ) -> None:
        super().__init__(
            model_name=model_name, text_template=text_template, text_pair_template=text_pair_template,
            label_map=label_map if label_map is not None else {"POSITIVE": 1.0, "NEGATIVE": 0.0},
            aggregation=aggregation, device=device, name=name,
        )
