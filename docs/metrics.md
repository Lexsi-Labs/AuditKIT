# Metrics

AuditKIT provides several metric families with a range of individual metrics.
All metrics use zero required dependencies unless otherwise noted.

For exact mechanics, constructor arguments, and honest caveats per metric,
see the [Scorer Reference](scorers_reference.md) and [Known Issues](known_issues.md).
To extract a clean value from raw output before scoring against any of
these, see [Annotators](annotators.md).

## Code / Deterministic

| Metric | Description | Example |
|---|---|---|
| `Equals()` | Exact string match | `Equals().score(s, "hello")` |
| `Contains(sub)` | Substring check | `Contains("cat", ignore_case=True).score(s, s2)` |
| `StartsWith(prefix)` | Prefix check | `StartsWith("he").score(s, "hello")` |
| `EndsWith(suffix)` | Suffix check | `EndsWith("lo").score(s, "hello")` |
| `Regex(pattern)` | Regex pattern | `Regex(r"\d+").score(s, "123")` |
| `Levenshtein()` | Edit-distance similarity [0,1] | `Levenshtein().score(s, "sitting")` |
| `WordCount(min_words, max_words)` | Word count range | `WordCount(min_words=2, max_words=10).score(s, text)` |
| `IsJson(require_keys)` | Valid JSON + optional keys | `IsJson(require_keys=["name"]).score(s, json_str)` |
| `F1Score()` | Token-overlap F1 | `F1Score().score(s, "the cat sat")` |

## Generation

| Metric | Description | Extra |
|---|---|---|
| `Bleu(max_n=4)` | BLEU score | none |
| `RogueL()` | ROUGE-L F1 | none |
| `ChrF(n=6)` | Character n-gram F-score | none |
| `WordErrorRate()` | 1 - WER | none |
| `Perplexity()` | Exponential cross-entropy | `[transformers]` |
| `BertScore()` | BERTScore F1 | `[bert-score]` |

## Embedding Similarity

| Metric | Description | Extra |
|---|---|---|
| `CosineSimilarity()` | Cosine similarity of embeddings (plain `transformers`, no `sentence-transformers` package) | `[transformers]` |
| `TokenOverlap()` | Jaccard token overlap | none |
| `BM25Similarity()` | BM25 ranking score | none |

## Hallucination Detection

| Metric | Description | Extra |
|---|---|---|
| `FactualConsistency()` | NLI-based fact checking | `[transformers]` |

## Toxicity / Bias

| Metric | Description |
|---|---|
| `ToxicityScore()` | 1.0 = safe, lower = toxic |
| `RepresentationSkew()` | Demographic-representation balance (0 = balanced, 1 = one-sided) — not a bias verdict |
| `BiasJudge()` | LLM-as-judge: fraction of the output's own opinions that are biased (needs a `judge_model`) |
| `HateSpeechScore()` | 1.0 = safe |

## Pairwise / Preference

| Metric | Description |
|---|---|
| `WinRate()` | Fraction of candidates the output beats |
| `EloScore()` | Elo rating from pairwise results |
| `PreferenceAccuracy()` | Accuracy on preference pairs |

## RAG

| Metric | Description |
|---|---|
| `LexicalGroundedness()` | Output claims supported by context |
| `ContextCoverage()` | Target tokens found in context |
| `ContextOverlap()` | Context relevant to output |
| `AnswerOverlap()` | Token overlap with target |

## Security / Performance

| Metric | Description |
|---|---|
| `KeywordDetector(blacklist)` | Flag blacklisted terms |
| `DefconGrade.from_score()` | Maps score to DEFCON 1-5 |
| `GuardJudge()` | Safety scoring via a guard model (Llama Guard, WildGuard, HarmBench, ShieldGemma, Granite Guardian); 0 = safe, 1 = unsafe, harm categories in metadata |
| `LatencyStats()` | Latency tracking (ms) |
| `Throughput()` | RPS tracking |

## LLM-as-Judge

```python
judge = ak.GEval(
    rubric=[ak.RubricItem(criterion="correctness", weight=2.0)],
    judge_model=judge_model,
)
judge.score(sample, output)
```

## Encoder Judge (BERT/RoBERTa/DeBERTa/ELECTRA/...)

A judge backed by a real encoder classifier instead of a prompted
generative model — no API key, deterministic, works with any
`AutoModelForSequenceClassification`-compatible checkpoint. See
[Encoder Judge](encoder_judge.md) for the full guide (templating for
encoders, label-map auto-detection vs. explicit mapping, aggregation
modes, and real gotchas confirmed across 5 checkpoints/4 architectures).

```python
from auditkit.metrics.encoder_judge import EncoderJudge

judge = EncoderJudge(model_name="microsoft/deberta-base-mnli")
judge.score(sample, output)  # NLI-as-judge: does output entail sample.target?
```

## Choosing a model-backed judge

AuditKIT has three ways to score with another model. They look similar but
answer different questions. Pick by *what the grader model is*:

| Use | When | Grader | Needs |
|---|---|---|---|
| **`LLMJudge` / `GEval`** | open-ended quality, correctness, relevance, rubric grading | a **generative** LLM you prompt | a chat/completions model + API key or local weights |
| **`EncoderJudge`** | is the output entailed by / consistent with / classified as the target? | a **generative-free encoder classifier** (BERT/RoBERTa/DeBERTa NLI or sentiment) | local `transformers`; deterministic, no API key |
| **`GuardJudge`** | is the output (or prompt) **unsafe**? | a **purpose-built safety guard** (Llama Guard, WildGuard, HarmBench) | any `AutoModel` spec (local `hf:` or hosted) |

Rule of thumb: **quality → `LLMJudge`/`GEval`**, **agreement/entailment →
`EncoderJudge`**, **safety → `GuardJudge`**. `EncoderJudge` and `GuardJudge`
are both "wrap a non-generative model as a scorer," but `EncoderJudge` runs an
encoder classification head over a text pair, while `GuardJudge` sends a
role-structured conversation (or a classifier prompt) to a safety model and
parses its safe/unsafe verdict + harm categories.
