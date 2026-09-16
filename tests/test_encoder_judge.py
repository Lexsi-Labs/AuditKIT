"""Tests for EncoderJudge -- LLM-as-judge via a real encoder classifier.

Real, functional tests against real checkpoints (no mocks) -- consistent
with this repo's established pattern for metrics that load a real model
(bert_score, factual_consistency, cosine_similarity). Two real checkpoints
are used deliberately: `microsoft/deberta-base-mnli` has real,
human-readable id2label values (ENTAILMENT/NEUTRAL/CONTRADICTION), so it
exercises the auto-detected label_map path; `textattack/bert-base-uncased-MNLI`
has only generic LABEL_0/1/2 labels, so it exercises the "auto-detection
must raise, not guess" path and the explicit label_map override.
"""
from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.errors import ExtraNotInstalled
from auditkit.metrics.encoder_judge import EncoderJudge
from auditkit.sample import Sample


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


NEEDS_TRANSFORMERS = pytest.mark.skipif(
    not _importable("transformers"), reason="needs the transformers package"
)


class TestEncoderJudgeExtraNotInstalled:
    def test_extra_not_installed(self):
        if _importable("transformers"):
            pytest.skip("transformers is installed in this environment -- nothing to assert")
        with pytest.raises(ExtraNotInstalled):
            EncoderJudge().score(Sample(input="x", target="y"), "y")


class TestEncoderJudgeBasics:
    def test_name_and_kind(self):
        from auditkit.types import ScoreKind
        j = EncoderJudge()
        assert j.name == "encoder_judge"
        assert j.kind == ScoreKind.JUDGE

    def test_rejects_invalid_aggregation(self):
        with pytest.raises(ValueError, match="aggregation"):
            EncoderJudge(aggregation="bogus")

    def test_registered_in_metrics_registry(self):
        from auditkit.registry import METRICS
        assert "encoder_judge" in METRICS.names()


@NEEDS_TRANSFORMERS
class TestEncoderJudgeAutoDetection:
    """microsoft/deberta-base-mnli has real ENTAILMENT/NEUTRAL/CONTRADICTION
    labels -- auto-detection should resolve a correct label_map with no
    label_map= passed at all."""

    def test_identical_text_scores_high(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat on the mat.")
        score = j.score(s, "The cat sat on the mat.")
        assert score.value > 0.9
        assert "ENTAIL" in score.reason.upper()

    def test_unrelated_text_scores_low(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat on the mat.")
        score = j.score(s, "A completely unrelated sentence about rockets.")
        assert score.value < 0.1

    def test_metadata_carries_full_probability_distribution(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat on the mat.")
        score = j.score(s, "The cat sat on the mat.")
        assert set(score.metadata["probabilities"]) == {"ENTAILMENT", "NEUTRAL", "CONTRADICTION"}
        assert score.metadata["predicted_label"] == "ENTAILMENT"

    def test_argmax_aggregation_is_binary_flavored(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli", aggregation="argmax")
        s = Sample(input="q", target="The cat sat on the mat.")
        assert j.score(s, "The cat sat on the mat.").value == 1.0
        assert j.score(s, "A completely unrelated sentence about rockets.").value == 0.0

    def test_via_real_evaluate_pipeline(self):
        r = ak.evaluate(
            [Sample(input="q", target="The cat sat on the mat.")],
            model=lambda p: ["The cat sat on the mat."],
            scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
        )
        assert r.headline["encoder_judge"] > 0.9

    def test_via_real_compare_models(self):
        res = ak.compare_models(
            [lambda p: ["The cat sat on the mat."], lambda p: ["A completely different sentence."]],
            [Sample(input="q", target="The cat sat on the mat.")],
            scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
            model_names=["a", "b"],
        )
        assert res.winner("encoder_judge") == "a"


@NEEDS_TRANSFORMERS
class TestEncoderJudgeGenericLabels:
    """textattack/bert-base-uncased-MNLI only exposes LABEL_0/1/2 -- no
    real names to auto-detect from at all."""

    def test_auto_detection_raises_rather_than_guessing(self):
        j = EncoderJudge(model_name="textattack/bert-base-uncased-MNLI")
        with pytest.raises(ValueError, match="auto-detect"):
            j.score(Sample(input="q", target="x"), "y")

    def test_explicit_label_map_by_string_key(self):
        j = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
        )
        s = Sample(input="q", target="The cat sat on the mat.")
        score = j.score(s, "The cat sat on the mat.")
        assert score.value > 0.5  # identical text should lean toward entailment (LABEL_1)

    def test_explicit_label_map_by_int_index_matches_string_key(self):
        s = Sample(input="q", target="The cat sat on the mat.")
        j_str = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
        )
        j_int = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={0: 0.0, 1: 1.0, 2: 0.5},
        )
        assert j_str.score(s, "The cat sat on the mat.").value == j_int.score(s, "The cat sat on the mat.").value

    def test_unrecognized_label_map_key_raises(self):
        j = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={"NOT_A_REAL_LABEL": 1.0},
        )
        with pytest.raises(ValueError, match="doesn't match any"):
            j.score(Sample(input="q", target="x"), "y")


@NEEDS_TRANSFORMERS
class TestEncoderJudgeSingleSequenceMode:
    """text_pair_template=None -- a standalone classifier with no reference
    to compare against (e.g. sentiment), not an NLI-style pair judge."""

    def test_single_sequence_sentiment_classification(self):
        j = EncoderJudge(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            text_template="{output}", text_pair_template=None,
            label_map={"POSITIVE": 1.0, "NEGATIVE": 0.0},
        )
        s = Sample(input="q", target="")
        assert j.score(s, "I love this product, it is amazing!").value > 0.9
        assert j.score(s, "This is the worst thing I have ever bought.").value < 0.1


@NEEDS_TRANSFORMERS
class TestEncoderJudgeCrossArchitecture:
    """Genericity isn't assumed -- verified against real checkpoints from
    multiple distinct encoder architectures, not just BERT/DeBERTa."""

    def test_roberta_architecture_with_real_labels(self):
        j = EncoderJudge(model_name="cross-encoder/nli-distilroberta-base")
        s = Sample(input="q", target="The cat sat on the mat.")
        assert j.score(s, "The cat sat on the mat.").value > 0.9
        assert j.score(s, "A completely unrelated sentence about rockets.").value < 0.1

    def test_roberta_minilm_variant(self):
        j = EncoderJudge(model_name="cross-encoder/nli-MiniLM2-L6-H768")
        s = Sample(input="q", target="The cat sat on the mat.")
        assert j.score(s, "The cat sat on the mat.").value > 0.9
        assert j.score(s, "A completely unrelated sentence about rockets.").value < 0.1

    def test_debertav3_architecture(self):
        """DebertaV2ForSequenceClassification -- a genuinely different
        implementation from the DeBERTa-v1 checkpoint used elsewhere in
        this file. Weaker discrimination on the unrelated-sentence pair
        (0.33 vs the ~0.001 seen on other checkpoints) is a real, honest
        model-quality difference, not a bug -- still correctly directional
        (predicted label is contradiction, well below the entailment case)."""
        j = EncoderJudge(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        s = Sample(input="q", target="The cat sat on the mat.")
        identical = j.score(s, "The cat sat on the mat.")
        unrelated = j.score(s, "A completely unrelated sentence about rockets.")
        assert identical.value > 0.99
        assert unrelated.value < identical.value
        assert unrelated.metadata["predicted_label"] == "contradiction"

    def test_electra_architecture_label_order_is_not_standardized(self):
        """Real, live-confirmed finding: howey/electra-base-mnli's label
        order (LABEL_0=entailment, LABEL_1=neutral, LABEL_2=contradiction)
        is the OPPOSITE convention from deberta/roberta-style checkpoints
        used elsewhere in this file -- exactly why auto-detection refuses
        to guess for generic labels, and why a real label_map has to be
        determined empirically per checkpoint, not assumed from another
        checkpoint's convention."""
        j = EncoderJudge(model_name="howey/electra-base-mnli", label_map={0: 1.0, 1: 0.5, 2: 0.0})
        s = Sample(input="q", target="The weather today is sunny and warm.")
        assert j.score(s, "The weather today is sunny and warm.").value > 0.9
        assert j.score(s, "I went to the store to buy some groceries.").value == pytest.approx(0.5, abs=0.1)

    def test_multi_label_classifier_toxic_bert(self):
        """unitary/toxic-bert: independent sigmoid outputs (multiple labels
        can be true simultaneously), not a single-label softmax over
        mutually exclusive classes like NLI -- a genuinely different kind
        of encoder classifier than every other test in this file."""
        j = EncoderJudge(model_name="unitary/toxic-bert")
        with pytest.raises(ValueError, match="auto-detect"):
            j.score(Sample(input="q", target=""), "you are all idiots")

        j_explicit = EncoderJudge(
            model_name="unitary/toxic-bert",
            text_template="{output}", text_pair_template=None,
            label_map={"toxic": 1.0},
        )
        s = Sample(input="q", target="")
        clean = j_explicit.score(s, "Thank you so much for your help today, I really appreciate it.")
        toxic = j_explicit.score(s, "You are all idiots and I hate everyone here.")
        assert clean.value < 0.01
        assert toxic.value > 0.9


@NEEDS_TRANSFORMERS
class TestEncoderJudgeIdentity:
    def test_identity_reflects_all_config(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli", aggregation="argmax")
        identity = j.identity()
        assert identity["model_name"] == "microsoft/deberta-base-mnli"
        assert identity["aggregation"] == "argmax"

    def test_differently_configured_instances_have_different_identity(self):
        j1 = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        j2 = EncoderJudge(model_name="microsoft/deberta-base-mnli", aggregation="argmax")
        j3 = EncoderJudge(model_name="microsoft/deberta-base-mnli", text_pair_template=None)
        assert j1.identity() != j2.identity()
        assert j1.identity() != j3.identity()

    def test_identically_configured_instances_have_identical_identity(self):
        j1 = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        j2 = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        assert j1.identity() == j2.identity()


@NEEDS_TRANSFORMERS
class TestEncoderJudgeWithAnnotators:
    """The Annotator pipeline (RegexAnnotator/ThinkingStripAnnotator/...)
    extracts a clean value from the raw output of the model UNDER
    EVALUATION, entirely before any Metric -- including EncoderJudge --
    is ever invoked. This makes it fully metric-agnostic: identical
    wiring whether the downstream scorer is EncoderJudge, LLMJudge, or a
    plain deterministic metric. Verified live across two different
    encoder architectures and two different annotator types."""

    def test_regex_annotator_extraction_reaches_encoder_judge(self):
        from auditkit.annotator import RegexAnnotator

        def rambling_model(prompts):
            return ["Let me think about this step by step... "
                    "the answer must be: The cat sat on the mat." for _ in prompts]

        sample = Sample(input="What did the cat do?", target="The cat sat on the mat.")
        answer = RegexAnnotator(r"answer must be:\s*(.+)", group=1, name="answer")

        r_raw = ak.evaluate(
            [sample], model=rambling_model,
            scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
        )
        r_extracted = ak.evaluate(
            [sample], model=rambling_model,
            scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
            annotators=answer, extract_with="answer",
        )
        # Extraction must have genuinely changed what was scored -- not
        # just wired up but inert.
        assert r_extracted.headline["encoder_judge"] > r_raw.headline["encoder_judge"]
        assert r_extracted.predictions[0].parsed_answer == "The cat sat on the mat."
        assert r_extracted.predictions[0].raw_output != r_extracted.predictions[0].parsed_answer

    def test_thinking_strip_annotator_with_a_different_architecture(self):
        """Same mechanism, RoBERTa instead of DeBERTa, a different
        annotator type -- confirms this isn't specific to one checkpoint
        or one annotator implementation."""
        from auditkit.annotator import ThinkingStripAnnotator

        def thinking_model(prompts):
            return ["<think>hmm the cat probably sat somewhere</think>"
                    "The cat sat on the mat." for _ in prompts]

        sample = Sample(input="What did the cat do?", target="The cat sat on the mat.")
        strip = ThinkingStripAnnotator(pattern=r"(.+)", name="stripped")

        r = ak.evaluate(
            [sample], model=thinking_model,
            scorers=[EncoderJudge(model_name="cross-encoder/nli-distilroberta-base")],
            annotators=strip, extract_with="stripped",
        )
        assert r.headline["encoder_judge"] > 0.99
        assert r.predictions[0].parsed_answer == "The cat sat on the mat."


@NEEDS_TRANSFORMERS
class TestEncoderJudgeVariableLengthInputs:
    """Real bug found and fixed: BERT/RoBERTa/ELECTRA-style checkpoints use
    absolute position embeddings with a hard 512-token limit and crashed
    outright (RuntimeError: tensor size mismatch) on long combined input,
    since truncation wasn't enabled. DeBERTa (relative position
    embeddings, no hard limit) never hit this, which is why it wasn't
    caught by the smaller examples used in every earlier test in this
    file -- confirmed live before the fix, then confirmed fixed after."""

    _LONG_TEXT = (
        "The cat, a small domesticated feline that has lived alongside humans "
        "for thousands of years, decided on this particular afternoon to sit "
        "down comfortably on the soft woven mat that had been placed near the door."
    ) * 15  # comfortably exceeds 512 tokens combined with itself

    _LONG_TEXT_DIFFERENT_TOPIC = (
        "Rockets are vehicles designed to travel through space using powerful "
        "engines that expel exhaust gases at high speed to generate thrust via "
        "Newton's third law."
    ) * 15  # same rough length as _LONG_TEXT, but a genuinely unrelated subject

    # -- Moderate length imbalance: one side a full short sentence, the
    #    other a genuinely long passage. Distinct from the "extreme"
    #    single-word test below by using a real, complete sentence rather
    #    than pushing to the minimum possible token count. ----------------

    def test_short_output_vs_long_reference(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target=self._LONG_TEXT)
        score = j.score(s, "The cat sat.")
        assert 0.0 <= score.value <= 1.0

    def test_long_output_vs_short_reference(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat.")
        score = j.score(s, self._LONG_TEXT)
        assert 0.0 <= score.value <= 1.0

    # -- Extreme length imbalance: pushed to the minimum possible token
    #    count (a single word) on one side, vs. hundreds of words on the
    #    other -- a much larger length ratio than the "short vs long" pair
    #    above, which still used a full grammatical sentence. -------------

    def test_extreme_asymmetry_single_word_vs_long_paragraph(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target=self._LONG_TEXT)
        score = j.score(s, "Cat.")
        assert 0.0 <= score.value <= 1.0

    # -- Degenerate zero-length edge cases: empty output, empty target, and
    #    both empty -- three DISTINCT combinations, since the pipeline
    #    could plausibly behave asymmetrically depending on which of the
    #    two slots (text vs. text_pair) receives the empty string. --------

    def test_empty_output_does_not_crash(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat on the mat.")
        score = j.score(s, "")
        assert 0.0 <= score.value <= 1.0

    def test_empty_target_does_not_crash(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="")
        score = j.score(s, "The cat sat on the mat.")
        assert 0.0 <= score.value <= 1.0

    def test_both_empty_does_not_crash(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="")
        score = j.score(s, "")
        assert 0.0 <= score.value <= 1.0

    # -- Both very long, SAME text on both sides: a pure crash/range check
    #    (no discrimination claim, since there's nothing to discriminate
    #    between -- output and target are identical). Distinct from the
    #    "truncation preserves signal" tests below, which use two
    #    DIFFERENT long texts specifically to verify truncation doesn't
    #    destroy the classifier's ability to tell them apart. ------------

    def test_both_very_long_does_not_crash_on_absolute_position_embedding_model(self):
        """This exact case raised RuntimeError before truncation=True was added."""
        j = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
        )
        s = Sample(input="q", target=self._LONG_TEXT)
        score = j.score(s, self._LONG_TEXT)
        assert 0.0 <= score.value <= 1.0

    def test_both_very_long_on_relative_position_embedding_model(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target=self._LONG_TEXT)
        score = j.score(s, self._LONG_TEXT)
        assert score.value > 0.9  # identical (if truncated) text either way

    # -- Both very long, DIFFERENT topics: unlike the identical-text tests
    #    above, this actually exercises whether truncation preserves
    #    enough real content to still tell two long, unrelated passages
    #    apart -- not just "doesn't crash," but "still correct after
    #    losing whatever got cut off." Run against BOTH an
    #    absolute-position model (which genuinely truncates at 512 tokens,
    #    discarding real content) and DeBERTa (which doesn't truncate at
    #    all, so this instead just confirms long-input discrimination
    #    generally works) -- two different mechanisms, same assertion. ---

    def test_truncation_preserves_discriminative_signal_on_absolute_position_model(self):
        j = EncoderJudge(
            model_name="textattack/bert-base-uncased-MNLI",
            label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
        )
        s = Sample(input="q", target=self._LONG_TEXT)
        same_topic = j.score(s, self._LONG_TEXT)
        diff_topic = j.score(s, self._LONG_TEXT_DIFFERENT_TOPIC)
        assert same_topic.value > diff_topic.value
        assert same_topic.value > 0.9
        assert diff_topic.value < 0.5

    def test_long_input_discrimination_on_relative_position_model(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target=self._LONG_TEXT)
        same_topic = j.score(s, self._LONG_TEXT)
        diff_topic = j.score(s, self._LONG_TEXT_DIFFERENT_TOPIC)
        assert same_topic.value > diff_topic.value

    # -- Single-sequence mode (text_pair_template=None) with long input --
    #    a genuinely different code path from every test above: the
    #    pipeline receives a bare string, not a {"text", "text_pair"}
    #    dict, which triggers the separate nested-vs-flat return-shape
    #    normalization in score() (see the comment in encoder_judge.py). -

    def test_single_sequence_mode_handles_long_input(self):
        j = EncoderJudge(
            model_name="distilbert-base-uncased-finetuned-sst-2-english",
            text_template="{output}", text_pair_template=None,
            label_map={"POSITIVE": 1.0, "NEGATIVE": 0.0},
        )
        long_positive = "This is absolutely wonderful and amazing, I love it so much. " * 40
        s = Sample(input="q", target="")
        score = j.score(s, long_positive)
        assert score.value > 0.9


@NEEDS_TRANSFORMERS
class TestPrebuiltEncoderJudges:
    """2 ready-to-use EncoderJudge subclasses, each with the correct,
    checkpoint-specific config (model_name/label_map/template shape)
    baked in as defaults -- mirrors the Factuality/ClosedQA/Relevance
    pattern already used for LLMJudge. Every default is still a normal
    constructor argument, fully overridable.

    Originally 6, reduced in two rounds: DebertaNLIJudge/DebertaV3NLIJudge/
    RobertaNLIJudge collapsed into FactualityEncoderJudge (all three used
    checkpoints with real, auto-detectable labels -- three class names for
    no real behavior difference over passing model_name= directly). Then
    BertNLIJudge/ElectraNLIJudge removed entirely: both did the exact same
    task as FactualityEncoderJudge (entailment/factual-consistency) on
    checkpoints whose only distinguishing feature was an unrecoverable,
    empirically-verified generic label order -- real knowledge, but not a
    distinct task, so it's kept as a documented example in
    docs/encoder_judge.md instead of two more classes. The generic-label
    mechanism itself is still fully covered via bare EncoderJudge(model_name=...)
    in TestEncoderJudgeGenericLabels above, independent of any prebuilt.
    SentimentEncoderJudge (renamed from DistilBertSentimentJudge) is the
    only one left with a genuinely different task (sentiment, not NLI)."""

    def test_zero_config_factuality_judge_discriminates_correctly(self):
        j = ak.FactualityEncoderJudge()
        s = Sample(input="q", target="The cat sat on the mat.")
        identical = j.score(s, "The cat sat on the mat.")
        unrelated = j.score(s, "A completely unrelated sentence about rockets.")
        assert identical.value > 0.9
        assert unrelated.value < identical.value

    def test_zero_config_sentiment_judge(self):
        j = ak.SentimentEncoderJudge()
        s = Sample(input="q", target="")
        assert j.score(s, "I love this!").value > 0.9
        assert j.score(s, "I hate this!").value < 0.1

    def test_registered_in_metrics_registry(self):
        from auditkit.registry import METRICS
        for key in ("factuality_encoder_judge", "sentiment_encoder_judge"):
            assert key in METRICS.names()

    def test_aggregation_override_still_works(self):
        j = ak.FactualityEncoderJudge(aggregation="argmax")
        s = Sample(input="q", target="The cat sat on the mat.")
        assert j.score(s, "The cat sat on the mat.").value == 1.0

    def test_label_map_override_still_works(self):
        j = ak.SentimentEncoderJudge(label_map={"POSITIVE": 1.0, "NEGATIVE": 1.0})  # both count as pass
        s = Sample(input="q", target="")
        default_score = ak.SentimentEncoderJudge().score(s, "This is terrible.").value
        override_score = j.score(s, "This is terrible.").value
        assert override_score >= default_score

    def test_via_real_evaluate_pipeline(self):
        r = ak.evaluate(
            [Sample(input="q", target="The cat sat on the mat.")],
            model=lambda p: ["The cat sat on the mat."],
            scorers=[ak.FactualityEncoderJudge()],
        )
        assert r.headline["factuality_encoder_judge"] > 0.9

    def test_both_prebuilt_judges_score_the_same_run_together(self):
        r = ak.evaluate(
            [Sample(input="q", target="The cat sat on the mat.")],
            model=lambda p: ["The cat sat on the mat."],
            scorers=[ak.FactualityEncoderJudge(), ak.SentimentEncoderJudge()],
        )
        assert r.headline["factuality_encoder_judge"] > 0.9
        assert "sentiment_encoder_judge" in r.headline

    def test_all_prebuilt_classes_have_distinct_identities(self):
        judges = [ak.EncoderJudge(), ak.FactualityEncoderJudge(), ak.SentimentEncoderJudge()]
        identities = [str(j.identity()) for j in judges]
        assert len(identities) == len(set(identities))


@NEEDS_TRANSFORMERS
class TestEncoderJudgeInputContentEdgeCases:
    """Not length-related -- these probe unusual *content*, not size:
    whitespace-only strings, non-English/non-ASCII text, emoji, and text
    that happens to contain a literal special-token-looking substring
    (the exact scenario the older FactualConsistency's manual "[SEP]"
    string-concat approach handled unreliably -- see the module
    docstring's explanation of why EncoderJudge uses real tokenizer
    pair-encoding instead)."""

    def test_whitespace_only_strings_do_not_crash(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="   ")
        score = j.score(s, "   ")
        assert 0.0 <= score.value <= 1.0

    def test_non_english_text(self):
        """Distinct from the ASCII-only text used everywhere else in this
        file -- confirms tokenization/classification isn't ASCII-specific,
        using two different real scripts (French, Japanese)."""
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s_fr = Sample(input="q", target="le chat s'est assis sur le tapis")
        assert j.score(s_fr, "le chat s'est assis sur le tapis").value > 0.9
        s_ja = Sample(input="q", target="猫はマットの上に座った")
        assert j.score(s_ja, "猫はマットの上に座った").value > 0.9

    def test_emoji_does_not_break_tokenization(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        s = Sample(input="q", target="The cat sat on the mat 😺")
        assert j.score(s, "The cat sat on the mat 😺").value > 0.9

    def test_literal_special_token_text_in_real_content(self):
        """Real content that happens to contain the literal substring
        "[SEP]" (e.g. discussing a data format) must be treated as
        ordinary text, not accidentally interpreted as a real special
        token boundary -- exactly the ambiguity real tokenizer
        pair-encoding avoids that a manual string-concat approach
        wouldn't reliably handle."""
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        text = "The output format uses [SEP] to separate fields."
        s = Sample(input="q", target=text)
        assert j.score(s, text).value > 0.9


@NEEDS_TRANSFORMERS
class TestEncoderJudgeTemplatingEdgeCases:
    """Distinct from TestEncoderJudgeInputContentEdgeCases above -- these
    probe the _render() templating mechanism itself (which fields get
    substituted, what happens with an unrecognized placeholder), not the
    classifier's handling of unusual text content."""

    def test_text_pair_template_can_reference_retrieval_context(self):
        j = EncoderJudge(
            model_name="microsoft/deberta-base-mnli",
            text_template="{output}", text_pair_template="{context}",
        )
        s = Sample(input="q", target="unused", retrieval_context=["The cat sat on the mat."])
        assert j.score(s, "The cat sat on the mat.").value > 0.9

    def test_unrecognized_placeholder_is_left_literal_not_a_crash(self):
        """Matches LLMJudge._render()'s own established behavior for the
        same situation -- an unknown {field} name is left as literal text
        in the rendered string rather than raising, so a typo produces a
        real (if likely low-quality) score instead of an exception."""
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli", text_pair_template="{no_such_field}")
        s = Sample(input="q", target="x")
        score = j.score(s, "y")
        assert 0.0 <= score.value <= 1.0


@NEEDS_TRANSFORMERS
class TestEncoderJudgeMultiSampleAndPerformance:
    """Distinct from every single-sample .score() call used elsewhere in
    this file -- exercises the real multi-sample evaluate() batch path
    (Runner calling score() once per sample) and confirms the classifier
    pipeline is loaded once and reused, not reloaded per call."""

    def test_multi_sample_batch_via_evaluate_scores_each_sample_independently(self):
        samples = [
            Sample(input="q1", target="The cat sat on the mat."),
            Sample(input="q2", target="The dog ran in the park."),
            Sample(input="q3", target="Water boils at 100 degrees."),
        ]

        def matching_model(prompts):
            return ["The cat sat on the mat.", "A totally unrelated sentence.", "Water boils at 100 degrees."]

        r = ak.evaluate(samples, model=matching_model, scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")])
        scores = [p.score for p in r.predictions]
        assert scores[0] > 0.9    # matched
        assert scores[1] < 0.1    # mismatched
        assert scores[2] > 0.9    # matched
        assert r.headline["encoder_judge"] == pytest.approx(sum(scores) / 3, abs=1e-6)

    def test_pipeline_is_loaded_once_and_reused_across_calls(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli")
        assert j._pipe is None
        j.score(Sample(input="q", target="a"), "a")
        pipe_after_first_call = j._pipe
        assert pipe_after_first_call is not None
        j.score(Sample(input="q", target="b"), "b")
        assert j._pipe is pipe_after_first_call  # same object, not reloaded


@NEEDS_TRANSFORMERS
class TestEncoderJudgeDeviceAndErrorHandling:
    def test_explicit_device_override(self):
        j = EncoderJudge(model_name="microsoft/deberta-base-mnli", device="cpu")
        score = j.score(Sample(input="q", target="a"), "a")
        assert 0.0 <= score.value <= 1.0

    def test_invalid_model_name_raises_a_real_error_not_a_silent_failure(self):
        j = EncoderJudge(model_name="this-model-does-not-exist-12345")
        with pytest.raises(Exception):  # a real, upstream huggingface_hub/OSError, not swallowed
            j.score(Sample(input="q", target="x"), "y")
