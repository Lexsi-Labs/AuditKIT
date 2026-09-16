"""Text generation quality metrics (BLEU, ROUGE, chrF, WER, Perplexity, BERTScore)."""
from __future__ import annotations

import math
from typing import Any

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.errors import ExtraNotInstalled


@METRICS.register("bleu")
class Bleu(Metric):
    name = "bleu"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(self, max_n: int = 4, smooth: bool = True) -> None:
        self._max_n = max_n
        self._smooth = smooth

    def identity(self) -> dict:
        return {"name": self.name, "max_n": self._max_n, "smooth": self._smooth}

    @staticmethod
    def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _brevity_penalty(ref_len: int, hyp_len: int) -> float:
        if hyp_len == 0:
            return 0.0
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1.0 - ref_len / hyp_len)

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        ref = (sample.target or "").split()
        hyp = output.split()
        if not ref or not hyp:
            return Score(name=self.name, value=0.0, kind=self.kind)

        max_n = min(self._max_n, max(len(ref), len(hyp)))
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = self._ngrams(hyp, n)
            ref_ngrams = self._ngrams(ref, n)
            if not hyp_ngrams:
                precisions.append(1e-12 if self._smooth else 0.0)
                continue

            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1

            count = 0
            clipped = 0
            for ng in hyp_ngrams:
                count += 1
                if ref_counts.get(ng, 0) > 0:
                    clipped += 1
                    ref_counts[ng] -= 1

            p = clipped / count if count > 0 else 0.0
            if self._smooth and n > 1 and p == 0.0:
                p = 1e-12
            precisions.append(p)

        if not precisions:
            return Score(name=self.name, value=0.0, kind=self.kind)
        if any(p <= 0.0 for p in precisions):
            if self._smooth:
                precisions = [max(p, 1e-12) for p in precisions]
            else:
                return Score(name=self.name, value=0.0, kind=self.kind)
        log_sum = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_sum / len(precisions))
        bp = self._brevity_penalty(len(ref), len(hyp))
        return Score(name=self.name, value=bp * geo_mean, kind=self.kind)


@METRICS.register("rouge_l")
class RogueL(Metric):
    name = "rouge_l"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    @staticmethod
    def _lcs_length(a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        ref = (sample.target or "").split()
        hyp = output.split()
        if not ref and not hyp:
            return Score(name=self.name, value=1.0, kind=self.kind)
        if not ref or not hyp:
            return Score(name=self.name, value=0.0, kind=self.kind)

        lcs_len = self._lcs_length(hyp, ref)
        precision = lcs_len / len(hyp)
        recall = lcs_len / len(ref)
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return Score(name=self.name, value=f1, kind=self.kind)


@METRICS.register("chrf")
class ChrF(Metric):
    name = "chrf"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(self, n: int = 6, beta: float = 1.0) -> None:
        self._n = n
        self._beta = beta

    def identity(self) -> dict:
        return {"name": self.name, "n": self._n, "beta": self._beta}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        ref = sample.target or ""
        hyp = output
        if not ref or not hyp:
            return Score(name=self.name, value=0.0, kind=self.kind)

        # Cap by the shorter string's length, same as Bleu does for its word
        # n-grams -- without this, a short string (e.g. "hi"/"hi", 2 chars)
        # still gets n=3..6 character n-gram orders folded into the average,
        # each contributing a spurious 0 precision/recall since no n-gram of
        # that length can even exist, understating an otherwise-perfect match.
        max_n = min(self._n, max(len(ref), len(hyp)))
        total_p = 0.0
        total_r = 0.0
        for n in range(1, max_n + 1):
            ref_ngrams: dict[str, int] = {}
            for i in range(len(ref) - n + 1):
                ng = ref[i:i + n]
                ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

            hyp_ngrams: dict[str, int] = {}
            for i in range(len(hyp) - n + 1):
                ng = hyp[i:i + n]
                hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

            match_count = 0
            for ng, count in hyp_ngrams.items():
                match_count += min(count, ref_ngrams.get(ng, 0))

            hyp_count = sum(hyp_ngrams.values())
            ref_count = sum(ref_ngrams.values())

            p = match_count / hyp_count if hyp_count > 0 else 0.0
            r = match_count / ref_count if ref_count > 0 else 0.0
            total_p += p
            total_r += r

        avg_p = total_p / max_n
        avg_r = total_r / max_n
        beta2 = self._beta ** 2
        if avg_p == 0.0 and avg_r == 0.0:
            f_beta = 0.0
        else:
            f_beta = (1.0 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
        return Score(name=self.name, value=f_beta, kind=self.kind)


@METRICS.register("word_error_rate")
class WordErrorRate(Metric):
    name = "wer"
    kind = ScoreKind.BENCHMARK
    # MAXIMIZE, not MINIMIZE despite the class name: .score() below already
    # inverts to `max(0.0, 1.0 - wer)`, a 0-1 "higher is better" value, not
    # raw word-error-rate. Direction describes the returned VALUE's
    # convention, not the underlying named phenomenon.
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    @staticmethod
    def _word_edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
        m, n = len(ref_words), len(hyp_words)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] * (n + 1)
            for j in range(1, n + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev = curr
        return prev[n]

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        ref = (sample.target or "").split()
        hyp = output.split()
        if not ref and not hyp:
            return Score(name=self.name, value=1.0, kind=self.kind)
        if not ref:
            return Score(name=self.name, value=0.0, kind=self.kind)

        distance = self._word_edit_distance(ref, hyp)
        wer = distance / len(ref)
        # Word error rate itself is unbounded above (a much-longer/garbled
        # hyp can need more edits than ref has words), so 1.0 - wer can go
        # arbitrarily negative -- clamp to the [0, 1] range every other
        # normalized metric in this module promises, rather than leaking an
        # unbounded value into aggregation/thresholding code that assumes it.
        return Score(name=self.name, value=max(0.0, 1.0 - wer), kind=self.kind)


@METRICS.register("perplexity")
class Perplexity(Metric):
    name = "perplexity"
    kind = ScoreKind.BENCHMARK
    # Genuinely MINIMIZE: .score() returns raw exp(loss), unbounded and
    # not inverted -- lower perplexity is a more confident/correct model.
    direction = Direction.MINIMIZE
    is_deterministic = False
    required_fields = frozenset({"target"})

    def __init__(self, model_name: str = "gpt2") -> None:
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def identity(self) -> dict:
        return {"name": self.name, "model_name": self._model_name}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        if self._model is None:
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
            except ImportError:
                raise ExtraNotInstalled("transformers", "pip install auditkit[transformers]") from None
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_name)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(self._model_name)

        import torch
        encodings = self._tokenizer(output, return_tensors="pt")
        input_ids = encodings.input_ids
        # A causal-LM loss needs at least 2 tokens (one to predict from, one
        # to predict) -- a single-token `output` (a bare word/number/label,
        # exactly what an annotator's extraction commonly produces) has
        # nothing to shift against and silently returns nan. Prepend a
        # BOS-equivalent token so there's always a predecessor; GPT-2-family
        # tokenizers have no distinct bos_token, so eos_token_id doubles as
        # the conventional sequence start there.
        bos_id = self._tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self._tokenizer.eos_token_id
        if bos_id is not None:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)
        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        ppl = math.exp(loss)
        return Score(name=self.name, value=ppl, kind=self.kind)


@METRICS.register("bert_score")
class BertScore(Metric):
    name = "bert_score"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = False
    required_fields = frozenset({"target"})

    def __init__(self, model_name: str = "microsoft/deberta-xlarge-mnli") -> None:
        self._model_name = model_name

    def identity(self) -> dict:
        return {"name": self.name, "model_name": self._model_name}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        try:
            import sys
            import bert_score.score  # noqa: F401 -- ensures sys.modules entry exists
            from bert_score import score as bert_scorer
        except ImportError:
            raise ExtraNotInstalled("bert-score", "pip install bert-score") from None
        ref = [sample.target or ""]
        hyp = [output]
        # bert_score's own package __init__ does `from .score import score`,
        # which shadows the `bert_score.score` attribute with that function --
        # so the submodule must be reached via sys.modules, not attribute access.
        _bs_score_mod = sys.modules["bert_score.score"]
        original_get_tokenizer = _bs_score_mod.get_tokenizer

        def _patched_get_tokenizer(model_type, use_fast_tokenizer=False):
            # Some tokenizers (e.g. microsoft/deberta-xlarge-mnli, this
            # metric's own default) report model_max_length as HuggingFace's
            # "no limit configured" sentinel (~1e30). transformers>=5's
            # Rust-backed truncation setup tries to convert that into a
            # bounded int and raises OverflowError. bert_score itself never
            # clamps this, so we do it here before it's used for truncation.
            tok = original_get_tokenizer(model_type, use_fast_tokenizer)
            if getattr(tok, "model_max_length", 0) > 100_000:
                tok.model_max_length = 512
            return tok

        _bs_score_mod.get_tokenizer = _patched_get_tokenizer
        try:
            P, R, F1 = bert_scorer(hyp, ref, model_type=self._model_name, verbose=False)
        finally:
            _bs_score_mod.get_tokenizer = original_get_tokenizer
        f1 = F1.item()
        return Score(name=self.name, value=f1, kind=self.kind)
