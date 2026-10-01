# Scorer reference — every registered metric

All scorers registered in `METRICS` (`src/auditkit/registry.py`), grouped by
module (68 in all). Every scorer implements the same
fixed `score(sample, output, context=None) -> Score` interface described in
[LLM-as-Judge Rendering](llm_judge_rendering.md) — the differences below are
in constructor arguments, `required_fields` (which fields the runner checks
before it runs a scorer against a sample — see `Metric.applicable()`), and
what `context` keys (if any) they read.

**Kind** is the `ScoreKind` the scorer reports (`BENCHMARK`, `JUDGE`, `CODE`,
`SECURITY`, `RAG`, `AGENT`) — used for grouping/filtering results, not a hard
constraint on what the scorer measures.

## Deterministic / core (`src/auditkit/metric.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `exact_match` | `ExactMatch` | BENCHMARK | `target` | 1.0 iff `output` equals `target` after trimming outer whitespace. |
| `quasi_exact_match` | `QuasiExactMatch` | BENCHMARK | `target` | Exact match after normalizing both strings: lowercase, strip punctuation, drop articles (`a`/`an`/`the`), collapse whitespace. |
| `acc` | `Acc` | BENCHMARK | `target`, `choices` | 1.0 iff `output` resolves to the same 0-based choice index as `target`. Accepts the output as a bare letter (`"B"`), a 0-based index (`"1"`/`1`), or the literal choice text — see `_resolve_choice_index`. |
| `acc_norm` | `AccNorm` | BENCHMARK | `target`, `choices` | Same index-resolution scoring as `acc` in the current implementation (identical logic, separate registration for tasks that report both names). |

## Code / string matching (`src/auditkit/metrics/code.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `equals` | `Equals` | CODE | `target` | Exact string equality after trimming (`ignore_case=True` for case-insensitive; renames itself `equals_ci`). |
| `contains` | `Contains` | CODE | — | 1.0 iff a given `substring` appears anywhere in `output` (constructor arg, not sample-derived). |
| `starts_with` | `StartsWith` | CODE | — | 1.0 iff `output` starts with a given `prefix` (constructor arg). |
| `ends_with` | `EndsWith` | CODE | — | 1.0 iff `output` ends with a given `suffix` (constructor arg). |
| `regex` | `Regex` | CODE | — | 1.0 iff a given regex `pattern` (constructor arg) matches anywhere in `output`. |
| `levenshtein` | `Levenshtein` | CODE | — | `1 - (edit distance / max(len(output), len(target)))` — a real Wagner–Fischer DP edit distance, not an approximation. |
| `word_count` | `WordCount` | CODE | — | 1.0 iff `len(output.split())` falls within `[min_words, max_words]` (constructor args; either bound optional). |
| `is_json` | `IsJson` | CODE | — | 1.0 iff `output` parses as JSON; optionally also requires specific top-level keys (`require_keys=`) to be present in a parsed dict. |
| `f1_score` | `F1Score` | CODE | — | Token-level F1 between `output` and `sample.target`, computed via set intersection over whitespace-split tokens (order and duplicates ignored). |

## Generation quality (`src/auditkit/metrics/generation.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `bleu` | `Bleu` | BENCHMARK | `target` | Real BLEU: n-gram precision (up to `max_n=4`) with brevity penalty, geometric mean of precisions, optional smoothing for zero-count n-grams. |
| `rouge_l` | `RogueL` | BENCHMARK | `target` | ROUGE-L: F1 of precision/recall over the longest common subsequence (real DP LCS, not approximated) between `output` and `target`. |
| `chrf` | `ChrF` | BENCHMARK | `target` | Character-level F-beta score over character n-grams (`n=6` by default), a real chrF implementation. |
| `word_error_rate` | `WordErrorRate` | BENCHMARK | `target` | `1 - (word-level edit distance / len(target words))` — reports `1 - WER` (higher = better), registered under the metric name `wer`. |
| `perplexity` | `Perplexity` | BENCHMARK | `target` | Real perplexity of `output` under a real HF causal LM (`gpt2` by default, configurable via `model_name=`) — loads the model on first use; needs the `transformers` extra. Non-deterministic (`is_deterministic = False`) because it depends on whichever checkpoint is loaded. |
| `bert_score` | `BertScore` | BENCHMARK | `target` | Real BERTScore F1 via the `bert-score` package (`microsoft/deberta-xlarge-mnli` by default) — needs `pip install bert-score`. |

## Embedding-based similarity (`src/auditkit/metrics/embedding.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `cosine_similarity` | `CosineSimilarity` | BENCHMARK | `target` | Cosine similarity between real sentence embeddings of `output` and `target` (`sentence-transformers/all-MiniLM-L6-v2` by default) — computed directly via plain `transformers` (mean-pooling + L2-normalization, the same recipe `sentence-transformers` uses internally), not the `sentence-transformers` package itself. Needs the `[transformers]` extra. |
| `token_overlap` | `TokenOverlap` | BENCHMARK | `target` | Jaccard similarity (intersection / union) of whitespace-split token sets between `output` and `target`. No embeddings despite the file name — purely lexical. |
| `bm25_similarity` | `BM25Similarity` | BENCHMARK | `target` | A simplified BM25-style term-overlap score: sum of `min(count_in_output, count_in_target)` per shared token, normalized by the longer token count. Not a full BM25 (no IDF/length normalization against a corpus). |

## Hallucination / factuality (deterministic) (`src/auditkit/metrics/hallucination.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `factual_consistency` | `FactualConsistency` | BENCHMARK | `target` | Real NLI classification via a real HF `text-classification` pipeline (`microsoft/deberta-base-mnli` by default) on `"{target} [SEP] {output}"` — 1.0 if entailment, 0.0 if contradiction, 0.5 for neutral/other. Non-deterministic, needs `transformers`. |

## LLM-as-Judge (`src/auditkit/metrics/judge.py`)

See [LLM-as-Judge Rendering](llm_judge_rendering.md) for the full mechanics
(prompt placeholder resolution, `sample.metadata` extension, parsing). All
four ready-made judges below are `LLMJudge` subclasses with a pre-filled
prompt/choices/scale — none of them do their own generation or parsing.

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `llm_judge` | `LLMJudge` | JUDGE | — (configurable) | The base, fully-custom judge: you supply `prompt=`, `choices=` (classifier mode) or `scale=` (numeric mode), `judge_model=`, `system_prompt=`, `use_cot=`. Non-deterministic — makes a real LLM call. |
| `g_eval` | `GEval` | JUDGE | — | Rubric-driven: builds its prompt from a `rubric=[RubricItem(criterion, weight, description), ...]` list, fixed `scale=(1.0, 5.0)`, `use_cot=True`. |
| `factuality` | `Factuality` | JUDGE | — | Fixed A–E relationship classifier (Braintrust `autoevals`-style) comparing `output` against `expected`; choice→score map `{"A": 0.4, "B": 0.6, "C": 1.0, "D": 0.0, "E": 1.0}` — see the known reliability caveats below. |
| `closed_qa` | `ClosedQA` | JUDGE | — | Judges whether `output` correctly answers `input`, without requiring a gold `target` at all. |
| `relevance` | `Relevance` | JUDGE | — | Judges how relevant `output` is to `input`; `choices={"relevant", "partially_relevant", "irrelevant"}`. |
| `bias_judge` | `BiasJudge` | JUDGE | — | **`MINIMIZE`, 0 = no bias.** Asks the judge to extract the opinions/generalizations the model itself asserts and classify each as biased or fair (gender/race/politics/religion/age); score is the fraction flagged biased, per-opinion breakdown kept in `Score.reason`. Reads *meaning* (catches "women are too emotional to lead", which `representation_skew` scores 0.0). Non-deterministic; pin the `judge_model` and set `temperature=0`. |

> **Known reliability notes** (from real stress testing, documented in
> `logs/`, not fixed by design/instruction): `Factuality`'s A–E map has no
> "off-topic/irrelevant" category, so an irrelevant answer can outscore a
> correct one; its scale also isn't monotonic with "correctness" (it measures
> the *structural relationship* between two answers, not quality). `GEval`'s
> rubric-weighted internal calculation doesn't always reconcile cleanly with
> its inherited fixed 1–5 scale instruction. `ClosedQA`, `Relevance`, and the
> base `LLMJudge` performed reliably across every case tested.

## Encoder judge (`src/auditkit/metrics/encoder_judge.py`)

Same "judge" role as `LLMJudge`, but backed by a real **encoder
classifier** (BERT, RoBERTa, DeBERTa, ELECTRA, ALBERT, or any other
`AutoModelForSequenceClassification`-compatible architecture) instead of a
prompted generative model — it classifies a (candidate, reference) pair in
one forward pass and reads the verdict off the model's own output
distribution, with no free-text parsing. Full detail, verified across 8
real checkpoints spanning 5 architectures, in
[Encoder Judge](encoder_judge.md).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `encoder_judge` | `EncoderJudge` | JUDGE | `target` (default; `None` if `text_pair_template=None`) | Classifies `text_template=`/`text_pair_template=`-rendered spans (default: output vs. target, an NLI-as-judge setup) via a real encoder checkpoint's own pair-sequence tokenizer encoding. `label_map=` maps the checkpoint's labels to a 0–1 score — auto-detected for real label names (entailment/positive/etc.), **raises** rather than guessing for generic `LABEL_0`/`LABEL_1`/... labels. Deterministic — no generation, no sampling. |
| `factuality_encoder_judge` | `FactualityEncoderJudge` | JUDGE | `target` | `EncoderJudge` prebuilt for `microsoft/deberta-base-mnli` — real labels, auto-detected. |
| `sentiment_encoder_judge` | `SentimentEncoderJudge` | JUDGE | none | Prebuilt for `distilbert-base-uncased-finetuned-sst-2-english` — single-sequence 2-way sentiment (not NLI), the one prebuilt that doesn't need `target`. |

## RAG (`src/auditkit/metrics/rag.py`)

All four share one tokenizer (lowercase + strip punctuation), so `"Paris."`, `"paris"` and `"Paris"` tokenize the same. Contexts come from the trace's `retrieved_contexts` when present, else `Sample.retrieval_context`; a sample with no context is skipped (no fake 0).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `lexical_groundedness` | `LexicalGroundedness` | RAG | none | Fraction of `output`'s tokens backed by a retrieved context chunk, via word-prefix matching (`"cat"` matches `"cats"`, never a mid-word substring like `"an"` in `"banana"`). Named for the lexical-overlap heuristic it computes, distinct from RAGAS's LLM-based `Faithfulness`. Skipped when no context is available. |
| `context_coverage` | `ContextCoverage` | RAG | `target` | Fraction of `target`'s tokens that appear somewhere across all retrieved context chunks — measures whether retrieval surfaced what was needed, not whether `output` used it, the same scoping intent as RAGAS's `context_recall` but a raw vocabulary-overlap heuristic. Deliberately independent of `output`; skipped when no context is available. |
| `context_overlap` | `ContextOverlap` | RAG | none | Fraction of retrieved context chunks that share a token with `output` (same word-prefix rule as `lexical_groundedness`) — a crude relevance-of-retrieval check, distinct from RAGAS's LLM-based `ContextPrecision`. Skipped when no context is available. |
| `answer_overlap` | `AnswerOverlap` | RAG | `target` | Token-set intersection over `target`'s token count between `output` and `target` (shared tokenizer, so case and punctuation don't block a match). Doesn't read the retrieved context — distinct from RAGAS's `ResponseRelevancy`, which compares the answer to the *question* via embedding similarity. |

## Retrieval ranking (`src/auditkit/metrics/retrieval.py`)

Reads the ranked list from `context["trace"]["retrieved_contexts"]` (an
`agent:` endpoint's `contexts`, or `Sample.actual_trace`), else
`Sample.retrieval_context`. Full formulas in [Agents & RAG](agents_and_rag.md#rag).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `retrieval` | `RetrievalMetrics` | RAG | `reference_contexts` | Deterministic ranking metrics against gold ids/texts (a list, or `{id: grade}` for graded nDCG): emits `hit_rate`, `precision`, `recall`, `mrr`, `average_precision` and `ndcg`, each with an `@k` suffix when `k=` is set (the metric renames itself `retrieval@k`). `average_precision` divides by **all** relevant items (`min(k, n_relevant)` with `k`), so missed gold chunks lower it; `ndcg` takes its ideal DCG from the gold grades, not the retrieved list; `precision@k` divides by `k` even when fewer came back. Items compare as exact strings after stripping; a duplicate counts once but keeps its slot at zero relevance, so items after it hold their original rank (a repeat never lifts a later item up a rank). Samples without gold are skipped. |

## RAG, LLM-judged (`src/auditkit/metrics/rag_judge.py`)

Same retrieved-context source as `retrieval`. Each takes `judge_model=` (spec
string or model object), `judge_model_args=`, `temperature=` (default `0.0`),
`max_tokens=` and `max_context_chars=`. The judge answers one numbered verdict
per line; a reply that doesn't give exactly verdicts `1..n` raises, so the
Runner records an error and the sample gets no score (never a fake `0.0`).
Non-deterministic. Details in [Agents & RAG](agents_and_rag.md#judge-metrics).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `faithfulness` | `Faithfulness` | RAG | none (skips samples with no contexts or empty output) | Two judge calls: extract the answer's claims, then label each SUPPORTED / UNSUPPORTED / CONTRADICTED against the contexts. Value = supported / claims, so unsupported claims fail too. `metadata` keeps claims, verdicts and `n_contradicted`. A no-claim answer (judge replies `NONE`) is skipped. |
| `context_precision` | `ContextPrecision` | RAG | none (skips samples with no contexts) | One judge call labels each chunk RELEVANT / IRRELEVANT for the question (and `target`, when set). Emits `context_precision` (RAGAS-style rank-weighted AP over the relevant chunks retrieved) and `context_relevance` (plain fraction relevant). |
| `context_recall` | `ContextRecall` | RAG | `target` | `target` is split into sentences locally; one judge call labels each ATTRIBUTED / NOT_ATTRIBUTED to the contexts. Value = attributed / sentences; `reason` lists the missed sentences. |
| `context_relevance` | `ContextRelevance` | RAG | none (skips samples with no contexts) | Mean fraction of retrieved chunks the judge labels relevant to the question alone (no gold answer). The same number `context_precision` emits as its second score, so don't request both. |
| `answer_relevancy` | `AnswerRelevancy` | RAG | none | Two judge calls: extract the answer's statements, then label each ADDRESSES / OFF_TOPIC against the question. Value = addressing statements / statements; `reason` lists the off-topic ones. Needs no context and no gold answer. |
| `response_groundedness` | `ResponseGroundedness` | RAG | none (skips samples with no contexts or empty output) | The answer is split into sentences locally, then one judge call grades each SUPPORTED / UNSUPPORTED against the contexts. Value = supported / sentences. Deterministic sentence split, so the claim set is reproducible; doesn't separate contradictions. |
| `hallucination` | `Hallucination` | RAG | none (skips samples with no contexts or empty output) | **`MINIMIZE`.** `1 - ` the supported fraction from `faithfulness`'s claim extraction and verdicts, so contradicted and unsupported claims both count. `reason` labels the two kinds; `metadata["n_contradicted"]`. |

## Agents / tool use (`src/auditkit/metrics/agent.py`)

Reads the run's tool calls from `context["trace"]` (native `api:` tool calls,
an `agent:` endpoint's transcript, or `Sample.actual_trace`), else parses the
text output as one turn (Hermes `<tool_call>` blocks, JSON, fenced JSON).
References are `Sample.expected_tool_calls` as a list of turns, or
`{"any_of": [...]}` when several routes are correct (all three reference
metrics then score the same chosen route). The first three take `arg_mode=`
(`"exact"`, `"subset"`, `"name"`) and `arg_match=`
(`(tool_name, pred_args, ref_args) -> bool`); a non-default `arg_mode`
suffixes every score name (`tool_call_f1_subset`) and `arg_match` adds
`_custom_<hash>` (the hash identifies the matcher, so two custom matchers
never collide).
Matching is a maximum bipartite matching, order-invariant within a turn.
Details and a worked example in [Agents & RAG](agents_and_rag.md#agent-metrics).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `tool_call_f1` | `ToolCallF1` | AGENT | `expected_tool_calls` | Multiset match of all calls, turns ignored, duplicates counted. Emits `tool_call_precision`, `tool_call_recall`, `tool_call_f1` and `tool_call_exact` (same calls, any order). With `expected_tool_calls=[]`, every score is 1.0 iff no call was made. In `exact` mode, when `Sample.tools` carries schemas, a schema-declared optional argument the reference omits doesn't break the match (BFCL convention); an undeclared extra still does. **All three reference metrics are ineligible (no score) when the trace marks tool coverage unavailable** — an `agent:` reply with no transcript and no explicit `tool_calls`. |
| `trajectory_match` | `TrajectoryMatch` | AGENT | `expected_tool_calls` | `trajectory_strict`: same number of turns, turn *i* holds exactly reference turn *i*. `trajectory_in_order`: reference turns' calls appear in order in the run's calls; extra calls allowed, turn boundaries ignored. |
| `parallel_tool_calls` | `ParallelToolCalls` | AGENT | `expected_tool_calls` | `parallel_recall`: fraction of reference parallel groups (turns with 2+ calls) made together in one predicted turn. `parallel_precision`: fraction of predicted turns with 2+ matched calls whose calls all come from one reference turn. `parallel_detection`: 1.0 when the run parallelized exactly when the reference does. Recall/precision are omitted when they have no denominator. |
| `tool_call_validity` | `ToolCallValidity` | AGENT | `tools` | Fraction of calls valid against the offered schemas: known tool, required args present, no unknown args, JSON type and `enum` respected, arguments valid JSON. Validation recurses into array `items` and a nested object's own `properties`/`required`, not just top-level arguments. No reference needed; samples with no calls are skipped. |
| `redundant_tool_calls` | `RedundantToolCalls` | AGENT | none | **`MINIMIZE`.** Fraction of calls that exactly repeat an earlier call (name and arguments). Argument equality is JSON-value based (shared with `tool_call_f1`), so `f(a=1)` and `f(a=1.0)` are one call. Samples with no calls are skipped. |
| `task_completion` | `TaskCompletion` | AGENT | none (needs `judge_model=`) | `LLMJudge` subclass: the judge sees the task, `target` as the expected outcome, the tool-call transcript and the final answer, and picks `complete` (1.0), `partial` (0.5) or `failed` (0.0). An unparseable verdict **raises** (the Runner records an error and skips the sample), never a silent `0.0` averaged into the headline — same as the RAG verdict judges. Non-deterministic. |
| `tool_selection` | `ToolSelectionJudge` | AGENT | none (needs `judge_model=`) | For each call, the judge sees the task and the calls before it and decides whether that call was justified at that point; the score is the mean over calls. Unparseable verdicts are left out; if all are unparseable, one `unknown` result. Samples with no calls are skipped. |
| `tool_permission` | `ToolPermission` | AGENT | none | Fraction of calls within policy. Allowlist: `Sample.metadata["allowed_tools"]`, else the offered `Sample.tools`; denylist: `denied_tools=` or `Sample.metadata["denied_tools"]` (a denial always wins). `reason` lists the violations. Skipped when there's no policy and no calls. |
| `agent_loop_detection` | `AgentLoopDetection` | AGENT | none | 1.0 = loop-free, 0.0 = a loop was found: an identical call repeated `repetition_threshold` (3) times, two consecutive assistant messages at least `similarity_threshold` (0.85) similar, or a cycle in the call sequence. `reason` names the signals. |

## RAG stress (`src/auditkit/metrics/rag_stress.py`, `rag_stress_runner.py`)

Deterministic, no judge. They read the case and corpus snapshot from
`sample.metadata["rag_case"]` and `sample.metadata["corpus"]`, and the trace from
`context["trace"]` or `Sample.actual_trace`. When the needed gold or trace event is
missing they return an explicit `unknown` / `not_tested` result instead of a score.
Details in [RAG stress testing](rag_stress.md).

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `evidence_set_recall` | `EvidenceSetRecall` | RAG | none | 1.0 iff the retrieved docs cover one **complete** `sufficient_evidence_sets` entry (an alternate complete set also passes). |
| `freshness` | `Freshness` | RAG | none | Fraction of retrieved docs current as of the case `decision_date` (a doc is stale when `superseded_date <= decision_date` or its digest differs from the snapshot); also `stale_hit_rate`. |
| `acl_compliance` | `AclCompliance` | RAG | none | Fraction of retrieved docs the case `tenant`/`identity` may see (prohibited, cross-tenant and not-in-snapshot docs fail); also `unauthorized_hit_rate`. |
| `citation_support` | `CitationSupport` | RAG | none | Whether cited spans exist in the current version and match `gold_citations`: `citation_support`, `citation_precision`, `citation_recall`, `wrong_version_rate`. |
| `abstention` | `Abstention` | RAG | none | 1.0 for abstaining on an unanswerable case or answering an answerable one, 0.0 otherwise. Abstention is read from the trace (`abstained`, else empty `answer_claims`), else a keyword scan of the output. |
| `context_retention` | `ContextRetention` | RAG | none | 1.0 iff a complete sufficient evidence set is still present in the trace's `final_context` (not dropped or truncated after retrieval). |
| `numeric_accuracy` | `NumericAccuracy` | RAG | none | Whether the answer claims carry every number in the case `gold_answer`. `unknown` if the gold answer has no numbers. |

## Security (`src/auditkit/metrics/security.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `keyword_detector` | `KeywordDetector` | SECURITY | — | 1.0 (safe) unless `output` contains any word from a given `blacklist=` (case-insensitive substring match), with an optional `whitelist=` override. |

`DefconGrade` (also in this file) isn't a registered scorer — it's a helper
enum (`DEFCON_1`..`DEFCON_5`) for bucketing any 0–1 score into a severity
label via `DefconGrade.from_score(value)`.

## Toxicity / bias (`src/auditkit/metrics/toxicity.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `toxicity_score` | `ToxicityScore` | SECURITY | — | **By default (`use_model=True`) runs a real classifier** (`unitary/toxic-bert`, via `transformers`) — `1.0` clean, `0.0` toxic; raises `ExtraNotInstalled` if `transformers` isn't present. Pass `use_model=False` for the old dependency-free fallback: `1 - (blacklisted terms found / total blacklist size)` against a small built-in ~15-word list (override via `blacklist=`) — misses phrasing not on the list and can false-positive on safe text that merely mentions a listed word (e.g. "domestic violence prevention"); use the model path unless you specifically need zero dependencies. |
| `representation_skew` | `RepresentationSkew` | SECURITY | — | **`MINIMIZE`, 0 = balanced.** Total-variation distance from balanced representation of demographic *groups* within each axis (gender male/female, race, age young/old; override via `demographic_terms=`), averaged over the axes mentioned. `0.0` = groups mentioned equally, `1.0` = only one group on an axis appears; no demographic terms → `0.0`. **A representation measure, not a bias verdict** — it can't see meaning, so a sentence that mentions men and women equally scores `0.0` even if blatantly sexist, and a single-subject passage reads as skewed. Meaningful mainly *aggregated over a run*. For biased content use `bias_judge`; for whether the model treats groups differently use a counterfactual benchmark (BBQ/CrowS-Pairs via `run_lmeval`). Replaces the old `bias_score` (which took entropy over individual tokens and couldn't tell all-male "he him his" from balanced "he and she"). |
| `guard_judge` | `GuardJudge` | SECURITY | — | **`MINIMIZE`, 0 = safe.** Wraps a purpose-built guard model (default `hf:meta-llama/Llama-Guard-3-8B`; any `AutoModel` spec works, incl. hosted `groq:llama-guard-3-8b`) as a scorer. Sends the prompt+response as role-structured turns, the guard classifies safe/unsafe; value is `1.0` if unsafe. Per-sample `metadata.categories` carries the harm categories (e.g. `["Hate"]`). `assess="prompt"` screens the input instead of the response. Guards differ only in input format + output parser, held in a per-family `profile`: `llama_guard` (chat-template taxonomy guard), `wildguard`/`harmbench` (raw-prompt classifiers), and `shield_gemma`/`granite_guardian` (policy-parameterized — pass `policy=` to pick which harm policy/risk to check, e.g. `"hate"`). Offline safety scoring, not a runtime guardrail. |
| `hate_speech_score` | `HateSpeechScore` | SECURITY | — | `0.6 * toxicity_score + 0.4 * (1 - representation_skew)` — a fixed-weight composite (builds and reuses one `ToxicityScore`/`RepresentationSkew` instance internally, so the classifier pipeline in the default `toxicity_score` path only loads once, not per sample). The toxicity term does the real work; the representation term is a weak signal kept for continuity. Pass `use_model=False` to use the dependency-free `toxicity_score` fallback in the composite too. |

## Pairwise / preference (`src/auditkit/metrics/pairwise.py`)

| Registry name | Class | Kind | Required fields | What it does |
|---|---|---|---|---|
| `win_rate` | `WinRate` | BENCHMARK | `target` | Reads `context["candidates"]`; scores the fraction of them that `output` beats under a comparator (`"exact"` match to `target`, or `"token_overlap"` similarity — constructor arg). |
| `elo_score` | `EloScore` | BENCHMARK | `target` | If `context["pairwise_results"]` (list of `{"winner", "loser"}`) is given, runs a real incremental Elo update (`k=32` default) and returns the mean rating; otherwise falls back to a token-overlap-derived pseudo-rating around `initial_rating=1000`. |
| `preference_accuracy` | `PreferenceAccuracy` | BENCHMARK | `target` | Reads `context["preference_data"]` (list of `{"chosen", "rejected"}` pairs); scores the fraction of pairs where `output` is more token-overlap-similar to `chosen` than to `rejected`. Returns `0.5` if no preference data is given. |

## Not scorers (helpers registered elsewhere in `src/auditkit/metrics/`)

- `perf.py`'s `LatencyStats` and `Throughput` are **not** in the `METRICS`
  registry — they're plain classes for aggregating timing data, not
  per-sample scorers.

## How to look this list up yourself, live

```python
import auditkit  # registers every built-in metric as an import side effect
from auditkit.registry import METRICS

for name in sorted(METRICS._entries):
    cls = METRICS._entries[name]
    print(name, "->", cls.__module__.split(".")[-1], cls.__name__)
```

## Related

- [Annotators](annotators.md) — extract a clean value from raw output
  before it reaches any of these scorers (`extract_with=`); also documents
  the 3 pairwise/preference metrics above (`win_rate`, `elo_score`,
  `preference_accuracy`) that read fixed `context` keys an annotator's
  output can't reach.
- [Known Issues](known_issues.md) — confirmed bugs and honest limitations
  across these scorers, kept up to date independent of this reference.
