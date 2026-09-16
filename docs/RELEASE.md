<!--
  AuditKIT — Release Overview (v1.0.0)
  A feature-forward launch document: what AuditKIT offers and how to use it,
  with runnable code for every capability. For the exhaustive internal
  reference (with candid limitations), see docs/GUIDE.md.
-->

# AuditKIT 1.0.0 — Release Overview

> **Evaluate any model, on any dataset, across any technique — from one `ak.evaluate()` call.**
> Benchmark · LLM-as-judge · RAG · performance — one library, one result object, zero required dependencies.

AuditKIT (`auditkit`) is a standalone Python library for evaluating AI models. It unifies the fragmented world of model evaluation — academic benchmarks, LLM-as-judge, RAG scoring, and performance profiling — behind a **single API** and a **single, provenance-stamped result object**, so results from different techniques are directly comparable, cacheable, and reproducible by construction.

- **Version:** 1.0.0 · **Python:** 3.10+ · **License:** Apache-2.0 · **Platforms:** Linux / macOS / Windows
- **Zero required third-party dependencies** — the core runs on the standard library; every heavy backend and metric is an optional extra.

---

## Why AuditKIT

Evaluating models today means stitching together a different tool for every question. Academic accuracy? One harness. Is the answer actually *good*? A judge framework. Does the RAG pipeline hallucinate? Another library. How fast and how expensive? Yet another profiler. None of them share a data model, so you can never line the results up next to each other.

AuditKIT collapses all of that into **one evaluation spine**. Every technique — a benchmark, a judge, a RAG check, a latency profile — flows through the same pipeline and returns the same `RunResult`, carrying a stable fingerprint. That means:

- **One thing to learn.** `ak.evaluate(dataset, model, scorers)` covers every case.
- **Everything is comparable.** A benchmark run and a judge run are the same shape, so `ak.compare(...)` works across them.
- **Everything is reproducible.** Each run is identified by a sha256 over its full configuration; identical inputs hit a disk cache instead of recomputing.
- **Nothing is forced on you.** The core has zero third-party dependencies. Install only the backends and metrics you use.

---

## Release highlights

- 🧩 **One API, two engines.** `ak.evaluate()` runs AuditKIT's own spine; `ak.run_lmeval()` drives the full [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — both return the identical `RunResult`.
- 🤖 **10 model backends** behind one resolver: any Python callable, `openai:`, `anthropic:`, `hf:` (HuggingFace), `vllm:`, `litellm:` (Ollama & 100+ providers), `api:` (any OpenAI-compatible endpoint), `groq:`, `lexsi:`, and `precomputed`.
- 📊 **48 metrics across 10 families** — exact/fuzzy match, BLEU/ROUGE/ChrF, embedding similarity, RAG faithfulness/context, hallucination, toxicity/representation, an LLM-as-judge bias detector, guard-model safety scoring, pairwise/preference — all stdlib-first, heavy variants optional.
- ⚖️ **LLM-as-judge** with a generic classifier/numeric judge plus prebuilt `Factuality`, `ClosedQA`, `Relevance`, and rubric-driven `GEval`.
- 🔬 **Model comparison** — `ak.compare()` (base vs pruned/quantized), `compare_models()` (N-model bake-off, memory-efficient), and `evaluate_many()` (one model across many datasets), with paired-bootstrap significance and quality-vs-size/latency tradeoffs.
- ⚡ **Performance, size & cost measured automatically** on every run — latency, throughput, real model size (params/sparsity/MB), provider token usage, and dollar cost.
- 🔒 **Provenance by default** — every run is fingerprinted; results are cached and comparable by that fingerprint.
- 💻 **CLI + YAML** — `auditkit eval|compare|list|init`, configurable from a single YAML file.

---

## Installation

```bash
pip install auditkit                       # core — zero third-party deps
pip install "auditkit[openai]"             # + OpenAI backend
pip install "auditkit[all]"                # everything
```

Install from source:

```bash
pip install "auditkit @ git+https://github.com/Lexsi-Labs/AuditKIT.git"
```

Pick exactly the extras you need (full matrix [below](#optional-dependencies)):

```bash
pip install "auditkit[transformers]"           # HuggingFace models + several ML metrics, incl. embedding similarity
pip install "auditkit[lmeval]"                 # the lm-eval benchmark engine
pip install "auditkit[anthropic]"              # Anthropic backend
pip install "auditkit[requests]"               # groq: / api: / lexsi: backends
```

Requires Python 3.10+.

---

## 60-second quickstart

```python
import auditkit as ak

samples = [
    ak.Sample(input="What is 2+2?", target="4"),
    ak.Sample(input="Capital of France?", target="Paris"),
]

# A plain callable needs no keys or downloads — great for first contact.
# (Deliberately gets one right, one wrong, so the headline below isn't trivial.)
def toy_model(prompts):
    return ["4" if "2+2" in p else "London" for p in prompts]

result = ak.evaluate(samples, model=toy_model)
print(result.summary())
print(result.headline)        # {'exact_match': 0.5}
```

Swap the callable for a real model — nothing else changes:

```python
ak.evaluate(samples, model="groq:llama-3.3-70b-versatile")   # hosted API
ak.evaluate(samples, model="hf:gpt2", device="cpu")          # local HuggingFace
ak.evaluate(samples, model=lambda prompts: [call_my_app(p) for p in prompts])  # your own app
```

Every call returns a `RunResult`:

| Field | Contents |
|---|---|
| `headline` | per-metric averages, e.g. `{"exact_match": 0.82}` |
| `predictions` | per-sample records (input, output, expected, score) — the answer browser |
| `stats` | full distributions (mean, std, min, max, percentiles) |
| `perf` / `model_size` / `token_usage` | auto-measured latency/throughput, real model size, provider tokens |
| `fingerprint` | stable sha256 for caching & comparison |
| `errors` | anything that went wrong, isolated per-sample (never aborts the run) |

---

## The unified API at a glance

| Function | What it does |
|---|---|
| `ak.evaluate(dataset, model, scorers, ...)` | The front door. Runs a model on a dataset with any scorers → `RunResult`. |
| `ak.run_lmeval(tasks, model, ...)` | Run academic benchmarks through the real lm-eval-harness → the same `RunResult`. |
| `ak.evaluate_many(datasets, model, ...)` | One model across **many datasets** in a single call → `{name: RunResult}`. |
| `ak.generate(dataset, model, ...)` | Generate once, score many times — returns samples with outputs attached. |
| `ak.compare(baseline, candidate)` | Baseline-vs-candidate diff with per-metric percentage differences → `RunComparison`. |
| `ak.compare_models([...], dataset, ...)` | Run several models on one dataset → leaderboard + pairwise significance. |
| `@ak.scorer(direction=...)` | Turn any `(sample, output) -> float` function into a metric. |

---

## Feature tour

### 1. Any model, one interface

`AutoModel` resolves a string spec, a callable, or a `Model` instance — so the *same* evaluation code runs against a local checkpoint, a hosted API, or your own application.

```python
import auditkit as ak

# Hosted APIs (keys read from the usual provider env vars)
ak.evaluate(data, model="openai:gpt-4o-mini")
ak.evaluate(data, model="anthropic:claude-sonnet-4-20250514")
ak.evaluate(data, model="groq:llama-3.3-70b-versatile")           # GROQ_API_KEY
ak.evaluate(data, model="litellm:ollama/llama3")                  # local Ollama & 100+ providers
ak.evaluate(data, model="api:my-model", api_base="https://my-host/v1", api_key="…")

# Local HuggingFace (auto device: CUDA > MPS > CPU; pass device= to pin it)
ak.evaluate(data, model="hf:meta-llama/Llama-3.2-1B", hf_token="hf_…")
ak.evaluate(data, model="vllm:meta-llama/Llama-3.2-1B")

# Your own app — any list[str] -> list[str] callable
ak.evaluate(data, model=lambda prompts: [my_rag_pipeline(p) for p in prompts])
```

| Prefix | Backend | Capabilities | Extra |
|---|---|---|---|
| callable / `precomputed` | your code & pre-generated outputs | generate | core |
| `openai:` | OpenAI | generate | `openai` |
| `anthropic:` | Anthropic | generate | `anthropic` |
| `hf:` | HuggingFace transformers | generate **+ loglikelihood** (MCQ) | `transformers` |
| `vllm:` | vLLM | generate | `vllm` |
| `litellm:` | LiteLLM (Ollama, …) | generate | `litellm` |
| `api:` | any OpenAI-compatible server | generate | `requests` |
| `groq:` | Groq | generate | `requests` |
| `lexsi:` | Lexsi gateway | generate | `requests` |

### 2. Adapters & annotators

**Adapters** turn a `Sample` into a model-ready prompt/request. `adapter=None`
(the default) uses `GenerationAdapter` — renders `sample.input` as-is. Pass
`adapter="auto"` to route by the sample's *shape* instead, via `route_adapter()`:
`choices` present → `MCQAdapter`; `retrieval_context` present → `RAGAdapter`;
otherwise → `GenerationAdapter`. Routing looks only at what the sample carries,
never at the model — chat-vs-base prompt formatting is applied downstream by the
model backend, so the same adapter works correctly on both instruct and base
models without routing having to know or probe which one it is.

```python
ak.evaluate(data, model=m, adapter="auto")            # route by sample shape
ak.evaluate(data, model=m, adapter="mcq")              # explicit adapter by name
ak.evaluate(data, model=m, adapter=ak.ChatAdapter(system_prompt="..."))  # instance
```

`ChatAdapter` is intentionally **never** auto-selected — injecting a system
prompt is an explicit choice, not something routing should guess.

Built-in adapters cover the common shapes (`generation`, `mcq`, `rag`, `chat`,
`instruction`, `fewshot`, `template`), but a custom adapter is a one-liner:
subclass `Adapter`, register it, implement `adapt(sample, config) -> list[Request]`.
Set `method` to something specific to your adapter — it feeds the run
fingerprint, so a config change (e.g. a different `prefix`) is never silently
served from a stale cache:

```python
from auditkit.adapter import Adapter
from auditkit.model import Request
from auditkit.registry import ADAPTERS

@ADAPTERS.register("prefixed")
class PrefixAdapter(Adapter):
    def __init__(self, prefix: str = "Answer concisely: "):
        self.prefix = prefix
        self.method = "prefixed"
    def adapt(self, sample, config):
        return [Request(prompt=self.prefix + sample.input, request_type="generate", params={})]
    def identity(self):
        return {"method": self.method, "prefix": self.prefix}   # -> fingerprint

ak.evaluate(data, model=m, adapter=PrefixAdapter("Reply in one word: "))
```

**Annotators** pull a clean value out of a model's raw output as a separate,
opt-in step from scoring — e.g. extracting `"42"` out of a longer
chain-of-thought reply ending in `"FINAL ANSWER: 42"`. `annotators=None` (the
default) runs none. Multiple annotators can run in the same call;
`extract_with="<name>"` picks which one's extraction actually feeds the
metrics — the rest still run and stay inspectable via `Prediction.context`.

```python
extractor = ak.RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, name="final_answer")
r = ak.evaluate(data, model=m, scorers=["exact_match"],
                 annotators=[extractor], extract_with="final_answer")
r.predictions[0].context["final_answer"]   # {"extracted": "42", "matched": True, "raw": "..."}
```

Two more built-ins ship alongside `RegexAnnotator`: `LLMAnnotator` (asks a
*second* model to extract the value — for cases with no fixed marker a regex
can anchor to) and `ThinkingStripAnnotator` (drops a `<think>...</think>`
reasoning block some models emit). A custom annotator is the same shape as a
custom adapter — subclass `Annotator`, register it, implement
`annotate(sample, results) -> dict`:

```python
from auditkit.annotator import Annotator
from auditkit.registry import ANNOTATORS

@ANNOTATORS.register("word_count_flag")
class WordCountFlagAnnotator(Annotator):
    def __init__(self, max_words: int = 50, name: str = "word_count_flag"):
        self.max_words = max_words
        self.name = name
    def annotate(self, sample, results):
        text = results[0].text if results else ""
        n = len(text.split())
        return {"word_count": n, "over_limit": n > self.max_words}
    def identity(self):
        return {"name": self.name, "max_words": self.max_words}

r = ak.evaluate(data, model=m, scorers=["exact_match"],
                 annotators=[WordCountFlagAnnotator(max_words=5)])
r.predictions[0].context["word_count_flag"]   # {"word_count": 13, "over_limit": True}
```

### 3. Metrics: 39 across 10 families

Pass a metric by **name** (for zero-config metrics) or as an **instance** (to configure it). When you pass `scorers=None`, AuditKIT auto-selects `exact_match` for samples that have a `target`.

```python
ak.evaluate(data, model=m, scorers=["exact_match", "f1_score", "bleu", "rouge_l"])
ak.evaluate(data, model=m, scorers=[ak.Contains("cat"), ak.Regex(r"\d+"), ak.IsJson()])
```

| Family | Metrics |
|---|---|
| **Built-in** | `exact_match`, `quasi_exact_match`, `acc`, `acc_norm` |
| **Code / deterministic** | `equals`, `contains`, `starts_with`, `ends_with`, `regex`, `levenshtein`, `word_count`, `is_json`, `f1_score` |
| **Generation quality** | `bleu`, `rouge_l`, `chrf`, `word_error_rate`, `perplexity`*, `bert_score`* |
| **Embedding similarity** | `cosine_similarity`*, `token_overlap`, `bm25_similarity` |
| **RAG** | `lexical_groundedness`, `context_coverage`, `context_overlap`, `answer_overlap` |
| **Hallucination** | `factual_consistency`* |
| **Toxicity / representation** | `toxicity_score`*, `representation_skew`, `hate_speech_score`* |
| **Pairwise / preference** | `win_rate`, `elo_score`, `preference_accuracy` |
| **Security** | `keyword_detector`, `guard_judge` (guard-model safety scoring) |
| **LLM-as-judge** | `llm_judge`, `g_eval`, `factuality`, `closed_qa`, `relevance`, `bias_judge` |

<sub>`*` loads a model and needs an optional extra; everything else is stdlib-only.</sub>

Every metric declares a **direction** (higher-is-better vs lower-is-better), which the comparison layer uses so a drop in accuracy and a rise in latency are both understood correctly. Two more things every metric carries:

- **`required_fields`** — the sample fields a metric needs (e.g. `lexical_groundedness`
  needs `retrieval_context`, `factual_consistency` needs `target`). A metric is
  silently skipped (not scored, not counted as a failure) for any sample
  missing one of its required fields via `Metric.applicable(sample)` — so
  mixing tasks in one dataset (some with `retrieval_context`, some without)
  never crashes a run; it just scopes each metric to the samples it can
  actually judge.
- **`kind`** (a `ScoreKind`: `BENCHMARK`, `JUDGE`, `CODE`, `SECURITY`, `PERF`,
  `HUMAN`, `RAG`) — labels *how* a score was produced, independent of its
  value or direction, so a report can distinguish "a deterministic string
  match said this passed" from "a model judged this as good."

A `Score` also carries an optional `weight` — `ak.WeightedSum().compute(scores)`
folds several `Score`s into one weighted-average `Score` — and an optional
`threshold`, which turns a raw value into a `passed: bool`. `ak.ScoreGate(metric,
weight=..., threshold=...)` wraps any existing metric to set both without
touching the metric's own code:

```python
gated = ak.ScoreGate(ak.Contains("cat"), threshold=0.5)
score = gated.score(sample, output)[0]
score.passed   # True/False, gated on threshold
```

### 4. Custom scorers & metrics

Your own metric is a one-liner. `direction` is required, so comparisons always know which way is better:

```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)
def mentions_price(sample, output):
    return 1.0 if "$" in output else 0.0

ak.evaluate(data, model=m, scorers=[mentions_price])
```

For anything richer, subclass `Metric` and register it — the spine picks it up by name, no core changes:

```python
from auditkit.metric import Metric
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.registry import METRICS

@METRICS.register("under_limit")
class UnderLimit(Metric):
    direction = Direction.MAXIMIZE
    def __init__(self, max_len=280):
        self.max_len = max_len
        self.name = f"under_limit({max_len})"
    def score(self, sample, output, context=None):
        return Score(name=self.name, value=float(len(output) <= self.max_len),
                     kind=ScoreKind.CODE)
```

### 5. Academic benchmarks with lm-eval

`ak.run_lmeval()` drives the real lm-evaluation-harness — its task templates, filters, and metrics — and maps the results back into the same `RunResult`. Point it at **one task or a list**, and it runs them in a single call with a per-task breakdown.

```python
# one model, several benchmarks, one call
r = ak.run_lmeval(
    ["arc_easy", "hellaswag", "gsm8k"],
    model="hf:meta-llama/Llama-3.2-1B",
    limit=100, device="cpu",
)
print(r.headline)
# {'arc_easy:acc': 0.61, 'hellaswag:acc_norm': 0.58, 'gsm8k:exact_match': 0.34, ...}
```

- **Backends:** `hf:`, `vllm:`, `openai:`, `anthropic:`, `groq:`, `api:` (+ `base_url`).
- MCQ tasks (arc, hellaswag, mmlu) are scored by log-likelihood and need a logprob-capable backend (`hf:`/`vllm:`/`api:`); generative tasks (gsm8k) run on any backend, including chat-only APIs.
- Pass lm-eval knobs straight through: `apply_chat_template=True`, `system_instruction=...`, `gen_kwargs="temperature=0,max_gen_toks=256"`, gated-model `hf_token=...`.

Requires `auditkit[lmeval]`.

### 6. LLM-as-judge

Grade open-ended output with another model — as a classifier, a numeric score, or a weighted rubric. Judges plug in exactly like any other scorer.

```python
# Generic judge: classifier mode
quality = ak.LLMJudge(
    judge_model="groq:llama-3.3-70b-versatile",
    prompt="Question: {input}\nAnswer: {output}\nIs the answer correct and clear?",
    choices={"yes": 1.0, "partial": 0.5, "no": 0.0},
    use_cot=True, name="quality",
)
ak.evaluate(qa, model="hf:gpt2", scorers=[quality], device="cpu")

# Prebuilt judges
ak.evaluate(qa, model=m, scorers=[ak.Factuality(judge_model="groq:llama-3.3-70b-versatile")])
ak.evaluate(qa, model=m, scorers=[ak.Relevance(judge_model="openai:gpt-4o-mini")])

# Rubric-driven G-Eval (chain-of-thought, 1–5 scale)
rubric = [ak.RubricItem("Correctness", weight=2.0, description="factually right"),
          ak.RubricItem("Clarity", weight=1.0)]
ak.evaluate(qa, model=m, scorers=[ak.GEval(rubric, judge_model="groq:llama-3.3-70b-versatile")])
```

The judge parses its verdict from a marker line, ignores chain-of-thought reasoning above it, and emits an explicit **Unknown** (never a silent midpoint) when it can't parse — and the judge model + prompt are folded into the run fingerprint, so changing the "ruler" never silently reuses a cached score.

### 7. RAG evaluation

Give a sample its `retrieval_context` and the RAG metrics score groundedness and retrieval quality. AuditKIT auto-routes RAG-shaped samples to the right adapter.

```python
data = [ak.Sample(
    input="When was the Eiffel Tower completed?",
    target="1889",
    retrieval_context=["The Eiffel Tower was completed in 1889 for the World's Fair."],
)]
ak.evaluate(data, model=m, adapter="auto",
            scorers=["lexical_groundedness", "context_coverage", "context_overlap"])
```

### 8. Model comparison (base vs pruned/quantized)

Compare two runs and get a per-metric, per-task breakdown — reported as **percentage differences**, not an imposed pass/fail verdict, because what counts as "acceptable" depends on your metric. You decide.

```python
base = ak.evaluate(data, model="hf:my-base-model", scorers=["exact_match"])
pruned = ak.evaluate(data, model="hf:my-pruned-model", scorers=["exact_match"])

cmp = ak.compare(base, pruned)
print(cmp.summary())
# exact_match: 0.8200 -> 0.7900  delta=-0.0300 (-3.7% rel)  (n=500->500, std=…)

cmp.retention("exact_match")     # fraction of baseline quality kept
cmp.significance()               # paired bootstrap, aligned by sample_id
cmp.tradeoff(metric="exact_match")   # quality vs measured size/latency
cmp.regressed()                  # the exact samples that got worse
```

Run several models on the same dataset in one call — memory-efficient (each local model is freed before the next loads, so peak GPU stays at one model):

```python
res = ak.compare_models(
    ["hf:base", "hf:pruned", "groq:llama-3.3-70b-versatile"],
    dataset=data, scorers=["exact_match"],
    model_names=["base", "pruned", "prod"],
)
print(res.summary())             # leaderboard + per-task tables + pairwise significance
res.winner("exact_match")
cmp = res.pairwise("base", "pruned")   # zoom into any two → RunComparison
```

Each model in the list can also run under **its own** configuration — useful when
the candidates aren't interchangeable (a quantized checkpoint needs a
different `dtype`, or one candidate only needs a quick sanity-check subset):

```python
res = ak.compare_models(
    ["hf:base", "hf:pruned", "hf:pruned"],
    dataset=data, scorers=["exact_match"],
    model_names=["base", "pruned_fp16", "pruned_int8"],
    configs={"pruned_fp16": ak.RunConfig(limit=200)},        # this one only needs a quick subset
    model_opts={"pruned_int8": {"dtype": "int8"}},           # this one loads quantized
)
print(res.summary())
```

- `configs={name: RunConfig(...)}` overrides the shared `config=` for one model by
  name (falls back to `config` when a name is absent) — e.g. a different `limit`,
  `temperature`, or `concurrency` per candidate.
- `model_opts={name: {...}}` overrides the shared `**opts` for one model by name —
  backend construction kwargs like `dtype=`/`device=`, layered on top of whatever's
  passed to every model.
- **Performance stays honest across mismatched configs.** `RunComparison`/
  `CompareResult` never assume the runs are comparable in *size* — latency,
  throughput, and model size are each measured independently per run as it
  actually executes, not inferred from the shared config. So `pruned_fp16`
  running on 200 samples and `pruned_int8` running on the full set still each
  report their own real, correctly-scoped `performance()`/`tradeoff()` numbers —
  the sample-count difference shows up in each run's own `n`, never silently
  blended into a shared average.

**When candidates need their own adapter or annotators** — e.g. one model
answers directly while a more talkative one needs an `extractor` to pull the
final answer out of its reasoning — pass `adapter=`/`annotators=`/
`extract_with=` to `compare_models()` itself (shared across every model,
same as `evaluate()`'s own defaults), and override any of them for one
model only via that model's `model_opts` entry:

```python
extractor = ak.RegexAnnotator(r"FINAL ANSWER:\s*(.+)", group=1, name="final_answer")

res = ak.compare_models(
    ["hf:base-model", "hf:verbose-finetune"],
    dataset=data, scorers=["exact_match"],
    model_names=["base", "verbose_candidate"],
    model_opts={
        "verbose_candidate": {"annotators": [extractor], "extract_with": "final_answer"},
    },
)
print(res.summary())
res.pairwise("base", "verbose_candidate")   # -> RunComparison, same as always
```

`model_opts[name]` can override `adapter`/`annotators`/`extract_with` for that
model only — anything else in that dict (`dtype`, `device`, `api_key`, ...)
still flows through to model construction unchanged, exactly like before.
`scorers` stays a single shared argument, not overridable per-model — a
comparison assumes one metric set across every model. If you need a
per-model override this mechanism can't express, the
fallback is still available: run `ak.evaluate()` yourself per model and hand
the `RunResult`s to `ak.CompareResult(runs=..., dataset_size=..., scorers=...)`
directly — a plain dataclass, so nothing about the leaderboard/`.pairwise()`/
significance is lost either way.

Opt-in threshold gating is still available if you *want* a pass/fail gate — you pick the thresholds: `ak.compare(base, cand, pass_threshold=0.05).grade()`.

### 9. Evaluate across many datasets

`evaluate_many()` runs **one model across a list of datasets** in a single call — ideal for "evaluate my model on SORRY-Bench *and* OR-Bench *and* GSM8K," where each benchmark needs its own scorer. It runs them sequentially and returns one `RunResult` per dataset (its own headline, fingerprint, and cache entry — never a blended mean).

```python
results = ak.evaluate_many(
    {
        "refusal_bench":  (refusal_samples,  [refusal_judge]),      # refusal is good
        "over_refusal":   (over_ref_samples, [over_refusal_judge]), # over-refusal is bad
        "gsm8k_local":    (gsm_samples,      ["exact_match"]),
    },
    model="groq:llama-3.3-70b-versatile",
    on_error="skip",   # one bad dataset load doesn't lose the rest
)
for name, r in results.items():
    print(name, r.headline)
```

The model is resolved once and reused across every dataset, so a local checkpoint loads a single time.

### 10. Performance, size & cost — measured automatically

Every run measures performance with no extra work — it lands on the `RunResult` and flows into comparisons.

```python
r = ak.evaluate(qa, model="groq:llama-3.3-70b-versatile", scorers=[quality])

r.perf["latency_ms"]        # {count, mean, min, max, p50, p95, p99, std}
r.perf["throughput"]        # {rps, output_tokens_per_sec, total_tokens_per_sec}
r.model_size               # local: real params/nonzero/sparsity/size_mb; hosted: identity
r.token_usage              # provider-reported {prompt_tokens, completion_tokens, total_tokens}
r.cost({"input_per_1m": 0.59, "output_per_1m": 0.79})   # real tokens × your pricing
```

- **Latency & throughput** are timed by the runner as it runs.
- **Model size** is introspected from the loaded checkpoint for local backends (real parameter count, sparsity — so a pruned model reads differently from its base — and on-disk-equivalent MB).
- **Token usage** is the provider's own reported count (never estimated); **cost** multiplies it by pricing you supply (no bundled price table to go stale).

`RunComparison.tradeoff()` and `CompareResult.performance_table()`/`size_table()`/`cost_table()` turn these into quality-vs-cost views for base-vs-compressed comparisons.

### 11. Provenance, caching & reproducibility

Every run is identified by a **fingerprint** — a stable sha256 over the model, config, tasks, scorers, and adapter. Re-running an identical configuration returns a cached `RunResult` from disk instead of recomputing, and any change to any knob changes the fingerprint.

```python
r = ak.evaluate(data, model="hf:gpt2", scorers=["exact_match"], device="cpu")
r.fingerprint                 # 'a1b2c3d4…'
r.save("run.json")            # persist
loaded = ak.Result.load("run.json")

# Track named experiments across runs
r = ak.evaluate(data, model="hf:gpt2", experiment_name="math-v1")
exp = ak.ExperimentDB().load("math-v1")
exp.leaderboard(); exp.significance("exact_match")
```

### 12. Command line & YAML

Everything the Python API does is a command away.

```bash
auditkit eval --model groq:llama-3.3-70b-versatile --csv data.csv \
              --input-col question --target-col answer -o results.json
auditkit compare --models "hf:base,hf:pruned" --dataset gsm8k --baseline hf:base
auditkit list metrics
```

Or drive a run from a single YAML file:

```yaml
model: openai:gpt-4o-mini
temperature: 0.0
max_tokens: 256
prompts:
  - "What is the capital of France?"
adapter: generation
output: results.json
experiment: my-eval
```

```bash
auditkit eval --config auditkit.yaml
```

### 13. Extending AuditKIT

The library's core invariant: **a new modality, technique, or backend is a new plugin file plus one registry line — never a change to the evaluation spine.** Custom scorers, metrics, model backends, and adapters all register the same way and are picked up by name. See the [complete technical guide](GUIDE.md) for the extension points.

---

## Example notebooks

Every capability above has a runnable, Colab-ready notebook with real models
and real datasets — no example here is a toy. `examples/` tours individual
features; `examples/applications/` answers one specific real-world question end to end.

### `examples/`

| Notebook | Covers | File |
|---|---|---|
| 01 | Full pipeline: adapter, 5 metrics, LLM judge, LLM annotator | [`01_full_evaluation_pipeline.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/01_full_evaluation_pipeline.ipynb) |
| 02 | Generation across two real HF model families, compared | [`02_generation_across_hf_families.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/02_generation_across_hf_families.ipynb) |
| 03 | Custom annotators (regex, LLM-backed, fully custom) — §2 above | [`03_custom_annotators.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/03_custom_annotators.ipynb) |
| 04 | Metrics deep dive: built-in, custom, LLM-as-judge, RAG — §3/§4/§6/§7 above | [`04_metrics_deep_dive.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/04_metrics_deep_dive.ipynb) |
| 05 | Every data type/task kind: generative, MCQ, RAG, precomputed, chat — §2 above | [`05_data_types.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/05_data_types.ipynb) |
| 06 | Model comparison deep dive: `compare_models()` + `RunComparison` — §8 above | [`06_model_comparison.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/06_model_comparison.ipynb) |
| 07 | Annotators across 4 real model families, then compared together — §2/§8 above | [`07_annotators_across_models.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/07_annotators_across_models.ipynb) |
| 11 | Performance metrics, model comparison — §10 above | [`11_performance_metrics_demo.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/11_performance_metrics_demo.ipynb) |

### `examples/applications/`

| Notebook | Real-world question answered | File |
|---|---|---|
| 01 | How much does pruning severity (20%/40%/60%) degrade a model? Real BoolQ, real annotator, ship/no-ship verdicts, cross-checked via `ak.run_lmeval()` | [`01_application_pruned_llama_boolq.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/01_application_pruned_llama_boolq.ipynb) |
| 02 | Healthcare: clinical QA correctness vs. grounding (PubMedQA) | [`02_application_healthcare_pubmedqa.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/02_application_healthcare_pubmedqa.ipynb) |
| 03 | Finance: QA grounded in real SEC 10-K filings | [`03_application_finance_10k_qa.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/03_application_finance_10k_qa.ipynb) |
| 04 | E-commerce: review-sentiment triage at scale | [`04_application_ecommerce_review_triage.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/04_application_ecommerce_review_triage.ipynb) |
| 05 | Education: auto-graded tutoring, correctness vs. explanation | [`05_application_education_arc_tutor.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/05_application_education_arc_tutor.ipynb) |
| 06 | Enterprise search: internal knowledge assistant (real retrieval + RAG grounding) | [`06_application_enterprise_search_rag.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/06_application_enterprise_search_rag.ipynb) |
| 07 | LLM-as-judge via a real BERT NLI classifier, not a generative model | [`07_application_bert_nli_judge.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/07_application_bert_nli_judge.ipynb) |

Full, current lists (with fuller per-notebook descriptions) live in
[`examples/README.md`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/README.md) and
[`examples/applications/README.md`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/applications/README.md).

---

## Optional dependencies

| Extra | Installs | Unlocks |
|---|---|---|
| `openai` | `openai` | `openai:` backend |
| `anthropic` | `anthropic` | `anthropic:` backend |
| `transformers` | `torch`, `transformers` | `hf:` backend + `perplexity`, `factual_consistency`, `cosine_similarity`, model-based toxicity |
| `lmeval` | `lm-eval`, `accelerate` | the lm-eval benchmark engine (`ak.run_lmeval`) |
| `requests` | `requests` | `groq:` / `api:` / `lexsi:` backends |
| `vllm` | `vllm` | `vllm:` backend |
| `litellm` | `litellm` | `litellm:` backend (Ollama & 100+ providers) |
| `bert-score` | `bert-score`, `torch` | `bert_score` metric |
| `interop` | `datasets`, `mlcroissant` | HuggingFace / Croissant dataset loaders |
| `mlflow` | `mlflow` | MLflow experiment logging |
| `all` | all of the above | everything |

---

## Design principles

- **Zero required dependencies.** The core is stdlib-only. Heavy backends and metrics are optional extras, imported lazily; a missing one raises a clear, actionable `ExtraNotInstalled` — never an import-time crash.
- **One spine, many plugins.** Every technique flows through `Scenario → Adapter → Model → Metric → RunResult`. Adding a capability means adding a plugin, not editing the spine.
- **Reproducible by construction.** Runs are pure functions of their fingerprint; caching and comparison key off it.
- **Report facts, not verdicts.** Comparisons surface the numbers (deltas, percentages, retention, significance) and leave the accept/reject judgment to you — with opt-in gating when you want it.
- **Measure, don't guess.** Model size, token usage, and cost are introspected or provider-reported, never estimated.

---

## Known limitations

We'd rather you hear these from us than discover them:

- **The native built-in scenario loaders** (`mmlu`/`arc`/…) reference some stale HuggingFace dataset IDs; use `ak.run_lmeval()` for academic benchmarks — that's the maintained path.
- **Latency is call-level** (one timing per batched model call), so percentiles are only meaningful at `concurrency > 1`; a single-call run reports one latency sample.
- **`vllm:` model-size introspection** is not implemented yet (it reports identity only, while `hf:` reports real size).
- **Bootstrap significance** is a conservative heuristic, not a full permutation test — treat it as a guard, not a proof.
- A `Model` object doesn't yet serialize cleanly into a saved `RunResult` (everything else round-trips).

For the full, candid engineering reference — every config, backend, metric, and caveat — see **[docs/GUIDE.md](GUIDE.md)**.

---

## Future works

**Evaluation capabilities on the roadmap — not part of this release:**

- **Multimodal evaluation** — image/audio inputs and vision-language models.
- **Tabular-data evaluation** — structured/tabular model outputs and datasets.
- **Agentic & multi-turn evaluation** — tool-use traces and conversations.
- **Red-teaming & safety** — a first-class adversarial-probing suite (prompt-injection / jailbreak / encoding / over-refusal probes, detectors, and attack-success scoring).

**Engineering refinements:**

- Lineage-based automatic baseline selection for comparisons.
- First-class per-request latency and streaming metrics (TTFT/TPOT).
- Real `vllm:` model-size introspection.
- A rigorous permutation/sign test alongside the bootstrap.
- Cost/energy accounting and a Pareto quality-vs-cost view.

---

## Links & license

- **Repository:** https://github.com/Lexsi-Labs/AuditKIT
- **Full technical guide:** [docs/GUIDE.md](GUIDE.md)
- **License:** Apache-2.0
- **Built by** Lexsi Labs.

*AuditKIT — one evaluation spine for every model, every dataset, every technique.*
