# Metrics

AuditKIT has 68 registered metrics in 11 families. Most need no third-party
packages; the ones that do say which extra, and the judge metrics need a judge model.

For exact mechanics, constructor arguments, and caveats per metric,
see the [Scorer Reference](scorers_reference.md) and [Known Issues](known_issues.md).
To extract a clean value from raw output before scoring against any of
these, see [Annotators](annotators.md).

## Code / Deterministic

| Metric | Description | Example |
|---|---|---|
| `ExactMatch()` | `output` equals `target` after trimming whitespace (the default when samples have a `target`) | `scorers="exact_match"` |
| `QuasiExactMatch()` | Exact match after lowercasing, stripping punctuation and articles, collapsing whitespace | `scorers="quasi_exact_match"` |
| `Acc()` / `AccNorm()` | Multiple choice: the output resolves to the same choice as `target` (a letter, an index or the choice text) | `scorers="acc"` |
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
| `ToxicityScore()` | 1.0 = safe, lower = toxic. Runs a classifier (`unitary/toxic-bert`, `[transformers]`) by default; `use_model=False` uses a keyword list instead |
| `RepresentationSkew()` | Demographic-representation balance (0 = balanced, 1 = one-sided) — not a bias verdict |
| `BiasJudge()` | LLM-as-judge: fraction of the output's own opinions that are biased (needs a `judge_model`) |
| `HateSpeechScore()` | 1.0 = safe; a fixed blend of `ToxicityScore` and `RepresentationSkew` |

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
| `RetrievalMetrics(k=None)` | Ranked retrieval vs. gold `reference_contexts`: hit rate, precision, recall, MRR, average precision, nDCG (`@k` names with `k`) |
| `Faithfulness(judge_model=...)` | LLM judge: fraction of the answer's claims supported by the retrieved context |
| `ContextPrecision(judge_model=...)` | LLM judge: are retrieved chunks relevant and ranked relevant-first (`context_precision`, `context_relevance`) |
| `ContextRecall(judge_model=...)` | LLM judge: fraction of the reference answer's sentences the retrieved context covers |
| `ContextRelevance(judge_model=...)` | LLM judge: fraction of retrieved chunks relevant to the question |
| `AnswerRelevancy(judge_model=...)` | LLM judge: fraction of the answer's statements that address the question |
| `ResponseGroundedness(judge_model=...)` | LLM judge: fraction of the answer's sentences supported by the retrieved context |
| `Hallucination(judge_model=...)` | LLM judge: fraction of the answer's claims not supported by the context (lower is better) |

The first four are lexical heuristics. `RetrievalMetrics` is deterministic; the
judge metrics work with any text judge. See [Agents & RAG](agents_and_rag.md).

Deterministic RAG stress checks (no judge), scored against a gold case and a
versioned corpus snapshot; see [RAG stress testing](rag_stress.md):

| Metric | Description |
|---|---|
| `EvidenceSetRecall()` | Retrieval covered one complete sufficient evidence set |
| `Freshness()` | Fraction of retrieved docs current as of the decision date |
| `AclCompliance()` | Fraction of retrieved docs the case identity may see |
| `CitationSupport()` | Citations exist in the current version and match the gold citations |
| `Abstention()` | Abstains on unanswerable cases, answers answerable ones |
| `ContextRetention()` | A complete evidence set survived into the final context |
| `NumericAccuracy()` | The answer carries every number in the gold answer |

## Agents / Tool Use

| Metric | Description |
|---|---|
| `ToolCallF1(arg_mode="exact")` | Multiset precision / recall / F1 of tool calls, plus `tool_call_exact` |
| `TrajectoryMatch()` | Turn sequence vs. reference: `trajectory_strict`, `trajectory_in_order` |
| `ParallelToolCalls()` | Parallel calling: `parallel_recall`, `parallel_precision`, `parallel_detection` |
| `ToolCallValidity()` | Fraction of calls valid against `Sample.tools` schemas (no reference needed) |
| `RedundantToolCalls()` | Fraction of calls that repeat an earlier call (lower is better) |
| `TaskCompletion(judge_model=...)` | LLM judge of task + trajectory + final answer: complete / partial / failed |
| `ToolSelectionJudge(judge_model=...)` | LLM judge: was each tool call justified at that point (no reference needed) |
| `ToolPermission(denied_tools=None)` | Fraction of calls within the allowed/denied tool policy |
| `AgentLoopDetection()` | 1.0 = loop-free: no repeated call, stalled reasoning or call cycle |

These read the run's tool calls from the trace (native `api:` tool calls, an
`agent:` endpoint, or `Sample.actual_trace`) and fall back to parsing the text
output. Tool evals need `adapter=ak.ToolCallAdapter()` (or `adapter="auto"`).
See [Agents & RAG](agents_and_rag.md) for the data model, score semantics and a
worked parallel-call example.

## Security / Performance

| Metric | Description |
|---|---|
| `KeywordDetector(blacklist)` | Flag blacklisted terms |
| `DefconGrade.from_score()` | Maps score to DEFCON 1-5 |
| `GuardJudge()` | Safety scoring via a guard model (Llama Guard, WildGuard, HarmBench, ShieldGemma, Granite Guardian); 0 = safe, 1 = unsafe, harm categories in metadata |
| `LatencyStats()` | Latency statistics (ms) for `RunResult.perf`, not a per-sample scorer |
| `Throughput()` | Requests and tokens per second for `RunResult.perf`, not a per-sample scorer |

## LLM-as-Judge

```python
judge = ak.GEval(
    rubric=[ak.RubricItem(criterion="correctness", weight=2.0)],
    judge_model=judge_model,
)
judge.score(sample, output)
```

| Metric | Description |
|---|---|
| `LLMJudge(prompt=..., judge_model=...)` | The fully custom judge: your prompt, with `choices=` (classifier) or `scale=` (numeric) |
| `GEval(rubric=[...], judge_model=...)` | Rubric grading with weighted criteria |
| `Factuality(judge_model=...)` | A–E relationship classifier comparing `output` with the expected answer |
| `ClosedQA(judge_model=...)` | Does `output` correctly answer `input` (no gold `target` needed) |
| `Relevance(judge_model=...)` | Relevant / partially relevant / irrelevant to `input` |

## Encoder Judge (BERT/RoBERTa/DeBERTa/ELECTRA/...)

A judge backed by a real encoder classifier instead of a prompted
generative model — no API key, deterministic, works with any
`AutoModelForSequenceClassification`-compatible checkpoint. See
[Encoder Judge](encoder_judge.md) for the full guide (templating for
encoders, label-map auto-detection vs. explicit mapping, aggregation
modes, and checkpoint gotchas). Two presets: `FactualityEncoderJudge`
(NLI, `microsoft/deberta-base-mnli`) and `SentimentEncoderJudge`
(`distilbert-base-uncased-finetuned-sst-2-english`).

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
