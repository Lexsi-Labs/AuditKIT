"""Every registered metric, run through the real evaluate()/compare_models()
pipeline -- not just a direct Metric.score() call.

Direct .score() tests (tests/test_*_metrics.py, test_integration.py's
TestAllMetricFamilies) confirm each metric's own math is correct, but bypass
Runner/Adapter/RunSpec.fingerprint()/Prediction construction/context wiring
entirely. This file closes that gap: each metric gets at least one real
ak.evaluate() call, and a representative handful also go through
ak.compare()/compare_models() to confirm the comparison layer handles every
metric family (deterministic, generative, judge, RAG, pairwise, security)
correctly, not just exact_match.
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metric import Acc, AccNorm
from auditkit.metrics.code import Equals, Contains, StartsWith, EndsWith, Regex, Levenshtein, WordCount, IsJson
from auditkit.metrics.embedding import CosineSimilarity, TokenOverlap, BM25Similarity
from auditkit.metrics.generation import Bleu, RogueL, ChrF, WordErrorRate, Perplexity, BertScore
from auditkit.metrics.hallucination import FactualConsistency
from auditkit.metrics.judge import LLMJudge, GEval, RubricItem, Factuality, ClosedQA, Relevance
from auditkit.metrics.pairwise import WinRate, EloScore, PreferenceAccuracy
from auditkit.metrics.rag import LexicalGroundedness, ContextCoverage, ContextOverlap, AnswerOverlap
from auditkit.metrics.security import KeywordDetector
from auditkit.metrics.toxicity import ToxicityScore, RepresentationSkew, HateSpeechScore
from auditkit.model import Model, Result_, Generated
from auditkit.sample import Sample


class Canned(Model):
    """A model that always replies with a fixed string -- used as a judge_model."""
    name = "canned"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, requests):
        return [Result_(completions=[Generated(text=self.reply)]) for _ in requests]


def real_model(prompts):
    # No repeated words -- a repeated token (e.g. "the" appearing twice in
    # "the cat sat on the mat") makes set-based F1/precision imperfect even
    # for an identical string, which is real behavior but not what these
    # smoke tests are checking.
    return ["cat sat on mat quietly" for _ in prompts]


BASIC = [Sample(input="describe the scene", target="cat sat on mat quietly")]
MCQ = [Sample(input="pick one", target=1, choices=["wrong", "right", "also wrong"])]
RAG_DATA = [Sample(
    input="where did the cat sit?", target="cat sat on mat quietly",
    retrieval_context=["cat sat on mat quietly"],
)]


class TestDeterministicMetricsViaEvaluate:
    """equals/contains/starts_with/ends_with/regex/levenshtein/word_count/
    is_json/f1_score -- all zero-config, resolvable by bare string name."""

    def test_exact_and_quasi_exact_match(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=["exact_match", "quasi_exact_match"])
        assert r.headline["exact_match"] == 1.0
        assert r.headline["quasi_exact_match"] == 1.0

    def test_f1_score(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=["f1_score"])
        assert r.headline["f1_score"] == 1.0

    def test_token_overlap_and_bm25(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=["token_overlap", "bm25_similarity"])
        assert r.headline["token_overlap"] == 1.0
        assert r.headline["bm25_similarity"] == 1.0

    def test_equals_contains_starts_ends(self):
        r = ak.evaluate(
            BASIC, model=real_model,
            scorers=[Equals(), Contains("cat"), StartsWith("cat"), EndsWith("quietly")],
        )
        assert r.headline["equals"] == 1.0
        assert r.headline["contains(cat)"] == 1.0
        assert r.headline["startswith(cat)"] == 1.0
        assert r.headline["endswith(quietly)"] == 1.0

    def test_regex_levenshtein_word_count(self):
        r = ak.evaluate(
            BASIC, model=real_model,
            scorers=[Regex(r"cat.*mat"), Levenshtein(), WordCount(min_words=3)],
        )
        assert r.headline["regex(cat.*mat)"] == 1.0
        assert r.headline["levenshtein"] == 1.0
        assert r.headline["wordcount(3-)"] == 1.0

    def test_is_json(self):
        def json_model(prompts):
            return ['{"a": 1}' for _ in prompts]
        r = ak.evaluate(BASIC, model=json_model, scorers=[IsJson()])
        assert r.headline["is_json"] == 1.0

    def test_acc_and_acc_norm_mcq(self):
        def mcq_model(prompts):
            return ["right"]
        r = ak.evaluate(MCQ, model=mcq_model, scorers=[Acc(), AccNorm()], adapter="mcq")
        assert r.headline["acc"] == 1.0
        assert r.headline["acc_norm"] == 1.0


class TestGenerationQualityMetricsViaEvaluate:
    def test_bleu_rougel_chrf_wer(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[Bleu(), RogueL(), ChrF(), WordErrorRate()])
        assert r.headline["bleu"] == 1.0
        assert r.headline["rouge_l"] == 1.0
        assert r.headline["chrf"] == 1.0
        assert r.headline["wer"] == 1.0

    def test_perplexity(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[Perplexity()])
        assert r.headline["perplexity"] > 0.0

    def test_bert_score(self):
        # bert_score's default tokenizer (microsoft/deberta-xlarge-mnli)
        # reports model_max_length as HuggingFace's "no limit configured"
        # sentinel (~1e30), which transformers>=5's Rust-backed truncation
        # setup can't convert to a bounded int (OverflowError). BertScore.score()
        # works around this with a scoped monkeypatch of bert_score's own
        # get_tokenizer that clamps model_max_length before use (see
        # docs/BUGS.md) -- confirmed fixed, not just caught-and-skipped.
        r = ak.evaluate(BASIC, model=real_model, scorers=[BertScore()])
        assert not r.errors
        assert r.headline["bert_score"] > 0.99

    def test_bert_score_discriminates_unrelated_text(self):
        r = ak.evaluate(
            [Sample(input="x", target="cat sat on mat quietly")],
            model=lambda p: ["completely unrelated topic about rockets"],
            scorers=[BertScore()],
        )
        assert not r.errors
        assert 0.0 <= r.headline["bert_score"] < 0.7


class TestEmbeddingMetricsViaEvaluate:
    def test_cosine_similarity(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[CosineSimilarity()])
        assert r.headline["cosine_similarity"] > 0.99


class TestHallucinationMetricsViaEvaluate:
    def test_factual_consistency(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[FactualConsistency()])
        assert r.headline["factual_consistency"] == 1.0


class TestToxicityMetricsViaEvaluate:
    def test_toxicity_bias_hate_speech(self):
        r = ak.evaluate(
            [Sample(input="x", target="clean")], model=lambda p: ["have a nice day"],
            scorers=[ToxicityScore(), RepresentationSkew(), HateSpeechScore()],
        )
        assert r.headline["toxicity_score"] > 0.9
        # "have a nice day" mentions no demographic groups -> nothing to be
        # skewed -> 0.0 (representation_skew is MINIMIZE, 0.0 = balanced/none).
        assert r.headline["representation_skew"] == 0.0
        assert r.headline["hate_speech_score"] > 0.9


class TestPairwiseMetricsViaEvaluate:
    """win_rate/elo_score/preference_accuracy -- real (if proxy) behavior,
    including their documented fallback when context isn't populated."""

    def test_win_rate_elo_preference_accuracy(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[WinRate(), EloScore(), PreferenceAccuracy()])
        assert "win_rate" in r.headline
        assert "elo_score" in r.headline
        assert "preference_accuracy" in r.headline


class TestRagMetricsViaEvaluate:
    def test_lexical_groundedness_context_coverage_precision_relevancy(self):
        r = ak.evaluate(
            RAG_DATA, model=real_model,
            scorers=[LexicalGroundedness(), ContextCoverage(), ContextOverlap(), AnswerOverlap()],
            adapter="rag",
        )
        assert r.headline["lexical_groundedness"] == 1.0
        assert r.headline["context_coverage"] == 1.0
        assert r.headline["context_overlap"] == 1.0
        assert r.headline["answer_overlap"] == 1.0


class TestJudgeMetricsViaEvaluate:
    """LLMJudge/GEval/Factuality/ClosedQA/Relevance all need construction
    args (a judge_model at minimum) -- a Canned model stands in for the
    real judge so this runs with no GPU/API key, while still exercising
    the real judge-call-then-parse pipeline end to end."""

    def test_llm_judge_with_choices(self):
        judge = LLMJudge(judge_model=Canned("CHOICE: yes"), choices={"yes": 1.0, "no": 0.0})
        r = ak.evaluate(BASIC, model=real_model, scorers=[judge])
        assert r.headline["llm_judge"] == 1.0

    def test_g_eval_rubric(self):
        geval = GEval(rubric=[RubricItem("correctness", weight=2.0)], judge_model=Canned("Reasoning.\nSCORE: 5"))
        r = ak.evaluate(BASIC, model=real_model, scorers=[geval])
        assert r.headline["g-eval"] == 1.0

    def test_factuality(self):
        f = Factuality(judge_model=Canned("Reasoning.\nCHOICE: C"))
        r = ak.evaluate(BASIC, model=real_model, scorers=[f])
        assert r.headline["factuality"] == 1.0

    def test_closed_qa(self):
        qa = ClosedQA(judge_model=Canned("CHOICE: yes"))
        r = ak.evaluate(BASIC, model=real_model, scorers=[qa])
        assert r.headline["closed_qa"] == 1.0

    def test_relevance(self):
        rel = Relevance(judge_model=Canned("CHOICE: relevant"))
        r = ak.evaluate(BASIC, model=real_model, scorers=[rel])
        assert r.headline["relevance"] == 1.0


class TestSecurityMetricsViaEvaluate:
    def test_keyword_detector(self):
        r = ak.evaluate(BASIC, model=real_model, scorers=[KeywordDetector(blacklist=["forbidden"])])
        assert r.headline["keyword_detector"] == 1.0


# ============================================================================
# Representative metrics through compare()/compare_models() specifically --
# not re-testing each metric's math (that's the classes above), just
# confirming the comparison layer (leaderboard, RunComparison, fingerprint,
# direction-aware grading) correctly handles every metric *family*, not
# just exact_match.
# ============================================================================

class TestMetricFamiliesViaCompareModels:
    def test_generation_quality_family(self):
        res = ak.compare_models(
            [real_model, real_model], BASIC, scorers=[Bleu()],
            model_names=["a", "b"],
        )
        assert res.winner("bleu") is not None

    def test_embedding_family(self):
        res = ak.compare_models(
            [real_model, real_model], BASIC, scorers=[CosineSimilarity()],
            model_names=["a", "b"],
        )
        assert res.winner("cosine_similarity") is not None

    def test_toxicity_family(self):
        res = ak.compare_models(
            [lambda p: ["have a nice day"], lambda p: ["have a nice day"]],
            [Sample(input="x", target="clean")], scorers=[ToxicityScore()],
            model_names=["a", "b"],
        )
        assert res.winner("toxicity_score") is not None

    def test_rag_family(self):
        res = ak.compare_models(
            [real_model, real_model], RAG_DATA, scorers=[LexicalGroundedness()],
            model_names=["a", "b"], adapter="rag",
        )
        assert res.winner("lexical_groundedness") is not None

    def test_judge_family(self):
        judge = LLMJudge(judge_model=Canned("CHOICE: yes"), choices={"yes": 1.0, "no": 0.0})
        res = ak.compare_models(
            [real_model, real_model], BASIC, scorers=[judge],
            model_names=["a", "b"],
        )
        assert res.winner("llm_judge") is not None

    def test_deterministic_family_via_ak_compare(self):
        # ak.compare() (not compare_models()) with a non-exact_match metric.
        base = ak.evaluate(BASIC, model=real_model, scorers=[Levenshtein()])
        cand = ak.evaluate(BASIC, model=real_model, scorers=[Levenshtein()])
        cmp = ak.compare(base, cand)
        assert cmp.grade() is not None
