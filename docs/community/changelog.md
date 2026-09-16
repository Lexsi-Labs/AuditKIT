# Changelog

## Unreleased

### API changes

- **`CosineSimilarity` no longer depends on the `sentence-transformers`
  package — the `[sentence-transformers]` extra is removed entirely.**
  `sentence-transformers` was used for exactly this one metric in the
  whole library, and it transitively imports `transformers`' audio/video
  processing modules and, in turn, `torchcodec` -- a live-confirmed
  environment failure (`torch`/`torchcodec` version mismatch, missing
  `libavutil`/FFmpeg shared libraries) on some platforms (confirmed:
  Colab), with nothing to do with text embeddings at all. Reimplemented
  directly on plain `transformers.AutoModel`/`AutoTokenizer` (mean-pooling
  + L2-normalization -- the same recipe `sentence-transformers` uses
  internally for MiniLM-family checkpoints, not an approximation),
  already a core dependency for `EncoderJudge`/`HFGenModel` and confirmed
  not to trigger the `torchcodec` import path. `model_name=` default is
  now the full repo path (`sentence-transformers/all-MiniLM-L6-v2`, since
  plain `transformers` doesn't know `sentence-transformers`' short-name
  aliases). Needs `[transformers]` now, not a dedicated extra. Also fixed
  a pre-existing doc bug found in the process: `BM25Similarity` was
  listed everywhere as needing `[sentence-transformers]` despite its code
  never having any embedding-model dependency at all (pure token-frequency
  overlap, stdlib only) — corrected to "no extra needed".
- **New `guard_judge` (`GuardJudge`) — guard-model safety scoring.** Wraps a
  purpose-built safety classifier (default `hf:meta-llama/Llama-Guard-3-8B`;
  any `AutoModel` spec works, including hosted `groq:llama-guard-3-8b`) as an
  ordinary scorer: it classifies a model's output (or the prompt, via
  `assess="prompt"`) as safe/unsafe. `MINIMIZE`, 0 = safe; harm categories land
  in per-sample `metadata.categories`, so per-category rates and safety
  regressions flow into `RunResult`/`compare_models`. Guards differ only in
  input format + output parser, held in a per-family `profile` (`GUARD_PROFILES`;
  v1 ships `llama_guard`, custom profiles accepted). Offline safety scoring, not
  a runtime guardrail. Ships five profiles: `llama_guard` (chat-template
  taxonomy guard); `wildguard`/`harmbench` (raw-prompt classifiers, via a new
  `apply_chat_template=False` request flag the `hf:` backend honors, so a
  classifier's fixed prompt isn't corrupted by the tokenizer's chat template);
  and `shield_gemma`/`granite_guardian` (policy-parameterized — pass `policy=`
  to pick a harm policy/risk, threaded into the guard's chat template via a new
  `chat_template_kwargs` flag). Metric catalog is now 48 (the encoder judges
  below were added the same cycle).
- **Bias metrics reworked.** The old `bias_score` (`BiasScore`) was renamed to
  `representation_skew` (`RepresentationSkew`) *and* rewritten: it now measures
  demographic-representation balance per axis via total-variation distance from
  balanced (`MINIMIZE`, 0 = balanced), instead of Shannon entropy over
  individual tokens. The old version counted tokens and discarded group
  structure, so all-male "he him his" scored the same 0.9 as a balanced "he and
  she"; the rewrite scores those 1.0 vs 0.0. A new **`bias_judge`**
  (`BiasJudge`, LLM-as-judge) reads *meaning* — it flags the fraction of the
  output's own opinions that are biased, catching content ("women are too
  emotional to lead") that any word-count metric misses. `hate_speech_score`'s
  bias term was repolarized accordingly. For the deepest form (does the model
  *treat groups differently*), run BBQ/CrowS-Pairs via `run_lmeval`.
- **`compare_models(adapter=, annotators=, extract_with=)`** — previously
  `compare_models()` had no way to configure the adapter/annotators at all
  (every model silently got `evaluate()`'s bare defaults — `GenerationAdapter`,
  no annotators — regardless of what was passed). Now these three are real
  parameters, shared across every model by default, and `model_opts[name]`
  can override any of them for one model only — the rest of that dict still
  flows through to model construction unchanged. `scorers` remains a single
  shared argument, deliberately not overridable per-model.
- **`auditkit list annotators`** — the CLI's `list` subcommand previously
  covered `metrics`/`datasets`/`adapters`/`models` but had no way to
  enumerate registered `Annotator`s (`regex`, `llm`, `thinking_strip`);
  it's now a resource type alongside the others, and included in the
  default `auditkit list` (no argument) output.
- **`"echo"` removed from the public API surface.** `model=`/`--model` on
  `evaluate()`, `generate()`, `evaluate_many()`, the CLI (`eval`/`redteam`),
  and `RedTeamRunner` no longer default to it — `model=`/`--model` is now a
  required argument everywhere. `EchoModel` still exists internally (used
  by the test suite as a zero-dependency test double) but is no longer
  documented, exported behavior, or resolvable as advertised public API;
  use a plain `list[str] -> list[str]` callable or `"precomputed"` instead.
- **Removed `ASR`, `ClaimPrecision`, and `EntailmentRatio` metrics.** All
  three were thin token-overlap heuristics whose names oversold what they
  measured (see the removed entries in `docs/known_issues.md`'s prior
  "Proxy metrics" section) — not real attack-success judgment, claim
  verification, or entailment. `factual_consistency` (real NLI via
  `transformers`) remains as the one real hallucination-detection metric;
  `keyword_detector`/`DefconGrade` remain for security. Metric catalog is
  39 (was 42).
- **Latency reporting collapsed to one figure (mean), not p50+p95.**
  `RunResult.summary()`, `CompareResult.performance_table()`/`summary()`,
  and `RunComparison.summary()` previously showed `p50`/`p95` side by
  side — almost always identical in practice, since `Runner` times whole
  batched `generate()` calls, not individual requests, so `concurrency=1`
  (the default) produces exactly one latency sample per run and every
  percentile of a single value is trivially that same value.
  `performance_table()`'s returned dicts now have one `latency_mean_ms`
  key instead of `latency_p50_ms`/`latency_p95_ms`/`latency_mean_ms`. The
  full percentile breakdown (`p50`/`p95`/`p99`/`min`/`max`/`std`) is still
  computed and available via `RunResult.perf["latency_ms"]` for anyone
  who wants it — only the default printed/tabular views were simplified.
- **`ak.benchmark()` renamed to `ak.run_lmeval()`, and the `[benchmark]`
  pip extra renamed to `[lmeval]` to match** — makes explicit that this is
  the dedicated, always-lm-eval front door (no native/lmeval switch,
  unlike `evaluate(..., engine="lmeval")`, which reaches the same
  underlying engine) pulling in `lm-eval`/`accelerate` specifically —
  `[benchmark]` was ambiguous with the built-in benchmark *scenarios*
  (MMLU/GSM8K/etc.), which actually only need `[interop]`. Install via
  `pip install auditkit[lmeval]` now; the error message you'd get from
  calling `run_lmeval()` without it installed says so too.
- **New `ak.evaluate_many()`** — runs one model across several datasets in
  a single call (each as its own `evaluate()` call, its own fingerprint/
  cache entry, sequentially), returning one `RunResult` per dataset.
- **Real token throughput** (`output_tokens_per_sec`/`total_tokens_per_sec`)
  now computed automatically in `RunResult.perf`, `RunComparison`, and
  `CompareResult.performance_table()` from each run's own measured latency
  and provider-reported `token_usage` — no separate opt-in needed.
- **`HFGenModel`** no longer emits the `transformers` "Both `max_new_tokens`
  and `max_length` seem to have been set" warning on every call — clears
  the stale default on both the model's and the pipeline's own copy of
  `generation_config` right after loading.
- **Examples restructured**: the old numbered `.py`/`.sh` scripts in
  `examples/` were replaced with Colab-ready `.ipynb` notebooks (real
  models, real datasets). A new `applications/` folder holds end-to-end
  real-world scenarios (pruned-model ship/no-ship decisions, healthcare,
  finance, e-commerce, education, enterprise search, BERT-NLI-as-judge)
  built the same way. Of these, only `applications/01_application_pruned_llama_boolq.ipynb`
  uses the `lm-evaluation-harness` integration (`ak.run_lmeval()`) — as an
  authoritative cross-check next to the native evaluation, needing
  `auditkit[lmeval]`; no other notebook in `examples/` or `applications/`
  depends on it.

### New: Annotators

- **`RegexAnnotator`/`LLMAnnotator`** (`src/auditkit/annotator.py`) — pull a
  clean value out of a model's raw output (e.g. `"FINAL ANSWER: 42"` → `"42"`)
  as a separate, opt-in step from scoring. `RegexAnnotator` matches a
  user-supplied pattern; `LLMAnnotator` asks a second model to extract the
  value via a prompt, for cases with no fixed marker to anchor a regex to.
  Multiple annotators can run in the same call; `extract_with="<name>"`
  selects which one's extraction actually feeds the metrics — the rest still
  run and stay inspectable via `Prediction.context`. See
  [Annotators](../annotators.md).
- Both support `cast=`/`strict=` (convert the extracted string, e.g. to
  `int`/`float`; degrade gracefully or raise on a bad conversion) and are
  fully covered by run fingerprinting — changing a pattern/prompt/cast
  invalidates any cached result.

### New: OpenRouter model backend

- **`openrouter:` / `OpenRouterModel`** (`src/auditkit/model/openrouter_gen.py`)
  — a new T1 model backend for [OpenRouter](https://openrouter.ai), an
  OpenAI-API-compatible proxy in front of many providers. Mirrors
  `GroqModel`'s implementation and lm-eval wiring exactly (fixed base URL
  `https://openrouter.ai/api/v1`, Bearer auth, chat-only, no logprobs, maps
  to lm-eval's `openai-chat-completions` backend with `OPENROUTER_API_KEY`
  mirrored into `OPENAI_API_KEY` only for the run's duration). Models are
  selected via OpenRouter's own `"provider/model"` naming, e.g.
  `"openrouter:openai/gpt-4o-mini"` or
  `"openrouter:anthropic/claude-3.5-sonnet"` — the internal `/` doesn't
  conflict with `AutoModel.resolve()`'s prefix-splitting, which only splits
  on the first `:`. Adds two optional constructor params with no Groq
  equivalent — `site_url=`/`app_name=` — sent as `HTTP-Referer`/`X-Title`
  attribution headers when set. Rides the `requests` extra, same as
  `groq:`/`api:`.

### New: `EncoderJudge` — LLM-as-judge via a real encoder classifier

- **`EncoderJudge`** (`src/auditkit/metrics/encoder_judge.py`) — a judge
  backed by a real `AutoModelForSequenceClassification` encoder (BERT,
  RoBERTa, DeBERTa, ELECTRA, ...) instead of a prompted generative model.
  Classifies a (candidate, reference) pair in one forward pass via the
  tokenizer's real pair-sequence encoding — no free-text generation, no
  regex parsing. `label_map=` auto-detects from common NLI/sentiment-style
  label names, or raises (never guesses) for checkpoints with only generic
  `LABEL_0`/`LABEL_1` labels. `aggregation="probability"` (default, a
  confidence-weighted score) or `"argmax"`. Fully compatible with
  `Annotator`s (`extract_with=`) with no special-casing needed, since
  extraction happens before any metric is invoked. Verified live across 8
  real checkpoints spanning 5 architecture families (BERT, RoBERTa,
  DeBERTa v1/v3, ELECTRA, DistilBERT), including a real independent
  multi-label classifier. See [Encoder Judge](../encoder_judge.md).
- Found and fixed one bug during testing: BERT/RoBERTa/ELECTRA-style
  checkpoints (absolute position embeddings, hard 512-token limit) crashed
  outright on long combined input (`RuntimeError: tensor size mismatch`);
  DeBERTa (relative position embeddings, no hard limit) never hit this,
  which is why it wasn't caught until deliberately testing long inputs
  specifically. Fixed by always calling the classification pipeline with
  `truncation=True`.
- **2 prebuilt `EncoderJudge` subclasses** — `FactualityEncoderJudge`
  (entailment/factual-consistency) and `SentimentEncoderJudge` (2-way
  sentiment) — each with the correct, checkpoint-specific
  `model_name`/`label_map`/template shape baked in as defaults (same
  pattern as `Factuality`/`ClosedQA`/`Relevance` for `LLMJudge`), fully
  overridable.
  **Originally shipped as 6, reduced in two rounds.** First,
  `DebertaNLIJudge`/`DebertaV3NLIJudge`/`RobertaNLIJudge` were collapsed
  into one, `FactualityEncoderJudge` (keeping DeBERTa v1, the strongest of
  the three: no hard input-length limit, and empirically more
  confident/discriminative than DeBERTa-v3 on ambiguous pairs) — all three
  used checkpoints with real, auto-detectable labels, so having three
  separate classes added a name each but no real behavior difference over
  passing `model_name=` to one class directly. Then `BertNLIJudge`/
  `ElectraNLIJudge` were removed entirely: both did the exact same *task*
  as `FactualityEncoderJudge` (entailment/factual-consistency) — their
  only distinguishing feature was an unrecoverable, empirically-verified
  generic label order on their specific checkpoints (confirmed opposite
  conventions from each other), which is real knowledge but not a
  distinct enough task to justify two more classes. That knowledge is
  preserved as a worked example in `docs/encoder_judge.md`
  (`EncoderJudge(model_name=..., label_map=...)`) instead of two more
  classes with no real behavior difference beyond which checkpoint.
  `SentimentEncoderJudge` (renamed from `DistilBertSentimentJudge`) is the
  only prebuilt with a genuinely different task from `FactualityEncoderJudge`.

### Fixes

- **`Runner.score_one()`** no longer silently converts a missing optional
  dependency (`ExtraNotInstalled`, e.g. `cosine_similarity` without
  `[transformers]`) into a fake `0.0` score indistinguishable from a
  genuinely bad model output — it now surfaces via `result.errors`/
  `failed_count` instead.
- **`Perplexity`** no longer returns `nan` on short (single-token) outputs.
- **`WordErrorRate`** clamped to `[0, 1]` (could previously go negative).
- **`Faithfulness`/`ContextPrecision`** (RAG) fixed from plain substring
  matching (a short output word like `"an"` could match inside an unrelated
  word like `"banana"`) to word-boundary matching.
- **`ToxicityScore`** now runs a real classifier (`unitary/toxic-bert`) by
  default instead of a small keyword blacklist, which missed most real
  toxic phrasing and false-flagged safe text merely mentioning a listed
  word. `blacklist=`/`use_model=False` still available as the old
  dependency-free fallback.
- **`RunConfig.concurrency` default changed from `8` to `1`.** `Runner._chunk()`
  always splits requests into exactly `concurrency` chunks regardless of
  request count — with fewer samples than the configured concurrency, this
  produced empty chunks and wasted real model/API calls on them. `1` takes
  a no-chunking code path entirely, sidestepping the bug unconditionally.
  If you need real parallelism, pass a higher `concurrency=` explicitly —
  it's safe as long as your sample count is at least that value.
- **`identity()`/fingerprint correctness**: 14 built-in metrics had real
  constructor config that never affected `RunSpec.fingerprint()` (e.g.
  `Bleu(max_n=...)`, `EloScore(k=...)`, `ToxicityScore(
  use_model=...)`) — two differently-configured instances could silently
  share one cached `RunResult`. Fixed for all 14; `Annotator`/`Adapter`/
  `Metric` now also warn at class-definition time if a *custom* subclass
  looks like it has the same gap.
- **`load_croissant()` always crashed** — it called `mlcroissant.load(...)`,
  a function that doesn't exist in the real `mlcroissant` package (100%
  failure rate for any input, never caught since its only test checked the
  "extra not installed" path and lacked the `skipif` guard that would have
  made it actually run). Fixed to use the real
  `mlcroissant.Dataset(jsonld=...).records(record_set)` API — now takes a
  required `record_set` parameter (a Croissant file can describe several;
  there's no universally correct default) and decodes field values (the
  real API returns `bytes`). Verified against a real 918-row dataset,
  matching `load_hf()` on the same data exactly. `GitPython` added to the
  `[interop]` extra — `mlcroissant` needs it for some datasets' git-based
  downloads but doesn't declare it itself.
- **`bert_score` fixed** — previously documented as blocked by a genuine,
  unfixable `bert_score`/`transformers>=5` incompatibility
  (`OverflowError: int too big to convert`). Root cause: the default
  tokenizer (`microsoft/deberta-xlarge-mnli`) reports `model_max_length` as
  HuggingFace's "no limit configured" sentinel (`~1e30`), which
  `transformers>=5`'s Rust-backed truncation setup can't convert to a
  bounded int. `BertScore.score()` now applies a scoped monkeypatch of
  `bert_score`'s own `get_tokenizer` (reached via
  `sys.modules["bert_score.score"]`, since `bert_score/__init__.py`'s `from
  .score import score` shadows the plain attribute path) that clamps
  `model_max_length` to `512` before use, restoring the original function
  afterward. Verified live through the real `evaluate()` pipeline for both
  correctness (identical text → F1 ≈ 0.99999988) and discriminating power
  (unrelated text → F1 ≈ 0.43, not a degenerate always-high result). See
  `docs/BUGS.md`.
- **`HFGenModel` now honors `stop_sequences`/`seed`** — `stop_sequences` maps
  to `generate()`'s real `stop_strings` kwarg (with the tokenizer
  auto-attached, which that kwarg needs); `seed` calls transformers'
  `set_seed()` (a global RNG reset — the only per-call reproducibility knob
  `generate()`/`pipeline()` actually offers) right before generating.
  Verified live: a stop sequence truncates output exactly at the match, and
  the same seed + prompt + settings reproduces identical output across
  separate `evaluate()` calls. See `docs/BUGS.md`.
- **`VLLMModel.model_info()` implemented** — previously always reported
  identity-only despite `is_local=True`. Now attempts real parameter
  count/size introspection (same approach as `HFGenModel.model_info()`) via
  a few known vLLM-internal engine access paths, falling back to the
  previous identity-only reporting if none resolve on the installed vLLM
  version (its internal structure varies across V0/V1 engines and
  releases). See `docs/VLLM_KNOWN_ISSUES.md`.
- **`vllm:` backend verified live for the first time** (Colab L4 GPU) —
  real generation confirmed working end to end, closing a gap that had
  existed since the backend was first written (mocked unit tests only).
  Surfaced three environment/dependency crashes along the way, now
  fixed permanently inside `VLLMModel` itself (`_apply_environment_defaults()`/
  `_stdout_fix()` in `src/auditkit/model/vllm_gen.py`), so no user needs to
  work around them on any platform: vLLM's V1 engine forking a worker
  process that can't inherit an already-initialized CUDA context
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), Jupyter/Colab's `sys.stdout` having
  no real file descriptor for vLLM's internal log-suppression code
  (`sys.stdout`/`sys.stderr` swap around engine construction), and
  `huggingface_hub`'s newer Xet Storage download path 404ing on some
  legacy repos (`HF_HUB_DISABLE_XET=1`, best-effort). New
  `docs/VLLM_KNOWN_ISSUES.md` consolidates these, the still-open real gaps
  found in the process (no chat-template support, GPU memory never freed
  by `evaluate()` when resolving a model from a string spec), and the one
  unfixable-from-code issue (a `torch`/`vllm` CUDA version
  mismatch on environments with a pre-existing `torch`, e.g. Colab's
  default image). 5 new regression tests in `tests/test_generation_kwargs.py`.
- **`ExtraNotInstalled`'s message was self-contradictory** — every one of its
  ~16 call sites already passes a complete, ready-to-run instruction (e.g.
  `"pip install auditkit[vllm]"`), but the class always prefixed a second,
  separately-generated `"install auditkit[{extra}]"` in front of it,
  producing `"install auditkit[vllm] for pip install auditkit[vllm]"`. Now
  uses the passed hint verbatim, falling back to the generated form only
  when no hint is given at all. New `tests/test_errors.py`.
- **`ChatAdapter`'s stale source comment fixed** — no runtime behavior
  change; the comment claimed no backend reads `request.params["messages"]`,
  which stopped being true once `resolve_messages()` was wired into 5
  backends.
- **New `tests/test_metrics_via_pipeline.py`**: every one of the 39
  registered metrics now has at least one test through the real
  `ak.evaluate()` pipeline (`Runner`/`Adapter`/fingerprinting/`context`
  wiring), not just a direct `Metric.score()` call — previously only
  `exact_match` (plus one `Bleu()` instance) had that coverage, leaving 37
  metrics unverified beyond their own isolated math. A representative
  metric from each family (generation quality, embedding, toxicity, RAG,
  judge, deterministic) is also run through `compare_models()`/`ak.compare()`
  specifically, confirming the comparison layer handles every metric
  *kind* correctly, not just `exact_match`.

## v0.3.0 — 2026-07-01

### Highlights

- **Model comparison.** `compare_models()` runs the same dataset against multiple
  models, producing side-by-side results with bootstrap significance tests.
- **CLI subcommands.** `auditkit init` scaffolds projects, `auditkit list` shows
  available resources, `auditkit compare` compares models.
- **YAML config support.** Define evaluations in YAML files with `prompts:`,
  model config, tags, and split strategies. Load via `--config`.
- **Documentation rewrite.** Full MKDocs site with Material theme, cookbook case
  studies, best practices guides, and Lexsi-branded design system.

### New metrics

- RAG: `Faithfulness`, `ContextRecall`, `ContextPrecision`, `ResponseRelevancy`
- Generation: `Bleu`, `RogueL`, `ChrF`, `BertScore`, `Perplexity`, `WordErrorRate`
- Embedding: `CosineSimilarity`, `TokenOverlap`, `BM25Similarity`
- Hallucination: `FactualConsistency`, `ClaimPrecision`, `EntailmentRatio`
- Toxicity: `ToxicityScore`, `BiasScore`, `HateSpeechScore`
- Pairwise: `WinRate`, `EloScore`, `PreferenceAccuracy`
- Security: `ASR`, `DefconGrade`, `KeywordDetector`, `ThreatCategory`
- Performance: `LatencyStats`, `Throughput`

Conversation (`Coherence`/`TurnTaking`/`ContextAdherence`/`RepetitionPenalty`),
agent (`ToolCorrectness`/`TrajectoryMatch`/`StepEfficiency`/`GoalAccuracy`),
multimodal (`VQAAccuracy`/`ANLS`), and tabular
(`SchemaConformance`/`ExecutionAccuracy`/`DenotationAccuracy`) metrics were
added and then removed within this same version cycle (see
`logs/REMOVE_MULTIMODAL_AGENT_CONVERSATION_TABULAR_2026-07-17.md`) — none
of them exist in the current codebase.

### Fixes

- Bleu `max_n` clamped to `min(len(ref), len(hyp))` — exact match returns 1.0
- CLI `concurrency=None` no longer crashes Runner (defaulted to `8` at the
  time; see Unreleased above — the default later changed to `1`)
- DiskCache invalidation for stale `~/.cache/auditkit/runs/` entries

### Known issues

- CI workflow requires PAT with `workflow` scope

## v0.2.0 — 2026-06-15

- Initial public release
- Core evaluation spine (Sample, Metric, Model, Runner)
- Benchmark metrics (exact match, F1, quasi-exact match)
- CLI for basic evaluations
- Experiment tracking with ExperimentDB
