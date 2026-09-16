<!--
  AuditKIT vs. the LLM-evaluation landscape — a comparison study.
  Competitor features summarized from public docs as of mid-2026; frameworks
  evolve quickly, so treat competitor cells as directional and re-verify before
  quoting externally. The AuditKIT column is authoritative (from its own source).
-->

# AuditKIT vs. the Eval Landscape — A Comparison Study

**Purpose.** Where does AuditKIT sit among the major LLM-evaluation frameworks? This document compares AuditKIT feature-by-feature against the tools teams actually reach for today — what they offer, what we match, what we offer that they don't, and where we're honestly behind.

> **On accuracy.** The AuditKIT column comes straight from its own source and is authoritative. Everything about *other* frameworks is summarized from their public documentation around **mid-2026**; these projects move fast, so treat those cells as directional and re-verify a specific claim before you quote it externally. AuditKIT openly took inspiration from several of these — most directly **Braintrust's `autoevals`**, whose `Factuality` / `ClosedQA` / `Relevance` judges are the model for AuditKIT's prebuilt judges.

---

## 1. The landscape at a glance

The "eval" word covers four quite different kinds of tool. Knowing which category a tool is in explains most of the feature differences:

| Category | What it is | Examples |
|---|---|---|
| **Academic benchmark harnesses** | Run standardized leaderboard benchmarks (MMLU, GSM8K, …) | lm-evaluation-harness, HELM, OpenAI Evals, **AuditKIT** |
| **Developer eval libraries** (code-first) | Score your own data/outputs from code | **AuditKIT**, DeepEval, RAGAS, TruLens, Inspect |
| **Eval + observability platforms** (hosted) | Evals *plus* production tracing, dashboards, human review | Braintrust, LangSmith, Arize Phoenix, W&B Weave, MLflow |
| **Red-team / security suites** | Adversarial probing & vulnerability scanning | Promptfoo, DeepTeam, Giskard |

**Where AuditKIT sits:** a **standalone, code-first library** that unusually spans *three* of these — it wraps a real academic-benchmark harness, provides developer-library metrics/judges/RAG, and adds performance/cost/size measurement — behind **one API** and **one result object**, with **zero required dependencies**. It is deliberately **not** a hosted observability platform; that's the boundary line against Braintrust/LangSmith/Phoenix.

---

## 2. Capability matrix

Legend: ✅ first-class · 🟡 partial / limited / via add-on · ❌ not offered. Competitor cells are directional (mid-2026 public docs).

| Capability | AuditKIT | lm-eval-harness | Braintrust | DeepEval | Inspect (AISI) | Promptfoo |
|---|---|---|---|---|---|---|
| **Zero required third-party deps** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Academic benchmarks (MMLU/GSM8K/…) | ✅ *(wraps lm-eval)* | ✅ *(the standard)* | 🟡 | 🟡 | ✅ *(200+ evals)* | 🟡 |
| LLM-as-judge (classifier / numeric / rubric) | ✅ | 🟡 | ✅ *(autoevals)* | ✅ *(50+ incl. G-Eval)* | ✅ | ✅ |
| Prebuilt RAG metrics | ✅ | ❌ | ✅ | ✅ | 🟡 | 🟡 |
| Deterministic / string / code metrics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Custom metric addition** (one function/class, auto-registered, no core changes) | ✅ *(`@ak.scorer` decorator or subclass `Metric`)* | 🟡 *(YAML task + Python function registration)* | ✅ *(autoevals custom scorers)* | ✅ *(subclass `BaseMetric`)* | ✅ *(`@scorer` decorator)* | 🟡 *(JS/Python assertion file + YAML wiring)* |
| **Annotators** — extract a clean value from raw output before scoring, for metrics that need a fixed-format answer | ✅ *(built-in `RegexAnnotator`/`LLMAnnotator`, or subclass `Annotator`; `extract_with=` picks which one feeds scoring, rest stay inspectable via `context`)* | 🟡 *(regex-based answer filters baked into a task's YAML config, not a general reusable component)* | ❌ *(preprocess in your own scorer code)* | ❌ *(preprocess in your own metric code)* | 🟡 *(custom Python preprocessing, no first-class annotator concept)* | 🟡 *(`transform` functions reshape output before assertion)* |
| **Any input format** (CSV, JSON/Python objects, HF datasets) | ✅ *(`load_csv`, `load_hf`, `load_croissant`, or plain `Sample` list)* | 🟡 *(task-registry-centric; less suited to ad hoc CSV/JSON)* | ✅ | ✅ | ✅ *(dataset abstraction)* | ✅ *(YAML + CSV test cases)* |
| Red-teaming / adversarial | 🟡 *(basic probes/detectors shipped; not yet a polished suite)* | ❌ | 🟡 | ✅ *(DeepTeam, 40+ vulns)* | 🟡 *(safety evals)* | ✅ *(OWASP/NIST)* |
| Synthetic data generation | ❌ | ❌ | 🟡 *(Loop)* | ✅ | 🟡 | ✅ *(attacks)* |
| **Model comparison** — base-vs-variant, N-model bake-off, or one model across many datasets | ✅ *(`compare()`, `compare_models()`, `evaluate_many()` + significance)* | 🟡 *(run separately, no built-in comparison object)* | ✅ *(experiments)* | 🟡 *(separate runs, no dedicated leaderboard object)* | 🟡 *(separate logs, compared manually)* | ✅ *(across providers)* |
| Latency / throughput measured | ✅ | 🟡 | ✅ *(traces)* | 🟡 | 🟡 | 🟡 |
| **Model size / sparsity introspection** (pruning/quant) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Token usage & dollar cost | ✅ | 🟡 | ✅ | 🟡 | ✅ | ✅ |
| **Provenance fingerprint + result caching** | ✅ | 🟡 *(caching)* | ✅ *(platform)* | 🟡 | ✅ *(logs)* | 🟡 |
| CLI + config-as-code (YAML) | ✅ | ✅ *(CLI)* | 🟡 | 🟡 | ✅ *(CLI)* | ✅ *(YAML-first)* |
| Multimodal / agent / conversation | ❌ *(future works)* | 🟡 | ✅ | ✅ | ✅ | 🟡 |
| Extensible plugins (metrics/backends) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 3. What AuditKIT offers that most others don't

1. **One API + one result object across techniques — including academic benchmarks.** Benchmark, judge, RAG, deterministic, and performance evals all flow through `ak.evaluate()`/`ak.run_lmeval()` and return the same `RunResult`. *What that buys you:* the score from an MMLU benchmark run and a custom LLM-judge run are the same shape, so `ak.compare()` lines them up directly. Several tools unify judge+RAG (DeepEval) or run benchmarks (lm-eval) — few do **both** behind one interface returning one comparable object.
2. **Model size & sparsity introspection for compression work.** For local `hf:` models, AuditKIT reads real parameter count, **non-zero count / sparsity**, and on-disk-equivalent MB from the loaded checkpoint. *What that buys you:* `ak.compare(base, pruned)` then `.tradeoff()` reports, in one place, "the pruned model kept 98% of accuracy while dropping to 40% of the size and running 1.6× faster" — quality *and* footprint together. This is rare; most eval tools measure quality only, so you'd track size and latency in a separate spreadsheet.
3. **Provenance & caching in the data model.** Every run has a stable sha256 fingerprint over its full configuration. *What that buys you:* re-running an identical eval returns a cached result instantly instead of re-spending GPU/API time, and a change to any knob (seed, prompt, model) produces a new fingerprint — so results are reproducible and comparable by construction, not by convention.
4. **Comparison that reports facts, not verdicts.** `ak.compare()` surfaces per-metric percentage differences, retention, and paired-bootstrap significance and leaves the accept/reject call to you (with opt-in threshold gating) — deliberately avoiding one-size-fits-all pass/fail thresholds that aren't meaningful across different metrics.
5. **Auto-measured performance & cost on every run** — latency, throughput, provider token usage, and dollar cost, with no extra instrumentation. *What that buys you:* the answer to "how much does this eval's model actually cost per 1k calls, and how fast is it?" comes back on the same `RunResult` as the quality scores.

---

## 4. Head-to-head notes

### lm-evaluation-harness (EleutherAI)
The academic-benchmark workhorse — 60+ standardized benchmarks, the backend for the Hugging Face Open LLM Leaderboard, HF/vLLM/API model backends. It answers "how does my model score on MMLU/GSM8K/ARC," and nothing else.

**TL;DR:** AuditKIT doesn't compete with lm-eval — it wraps it, so you get the exact same benchmarks *plus* judge/RAG/custom metrics/perf/comparison in one call and one result object, which lm-eval alone can't give you.
- **AuditKIT relationship:** we don't compete — we **wrap it**. `ak.run_lmeval(tasks, model)` drives the real harness and maps its results into the same `RunResult` as our native judge/RAG/custom evals. So you get lm-eval's benchmarks *and* everything else through one interface and one result object.
- **What AuditKIT adds on top:** LLM-as-judge, RAG, deterministic metrics, model comparison, perf/size/cost — none of which lm-eval does. **What lm-eval has that we don't:** the deepest, most battle-tested academic task library (we expose it rather than reimplement it).

### Braintrust (+ autoevals) — *our acknowledged inspiration*
A commercial **eval + observability platform**: LLM-as-judge / code / human scorers, a browser Playground, the "Loop" AI agent for generating scorers/datasets, a CI GitHub Action for release gating, and production tracing — all hosted, one scorer running across eval and prod. Its open-source `autoevals` library is where AuditKIT's `Factuality`/`ClosedQA`/`Relevance` judges come from.

**TL;DR:** Braintrust's platform polish is real, but AuditKIT gives you the same measurement core — judge, RAG, benchmarks, and a model-compression comparison Braintrust doesn't have — fully offline, with no account, no hosted dependency, and no data ever leaving your box.
- **What AuditKIT does that Braintrust doesn't:** runs fully **offline as a library** (no account, no data leaving the box), **wraps academic benchmarks** (`ak.run_lmeval`), and measures **model size/sparsity + latency + cost** so you can compare a base model against a pruned/quantized one on quality *and* footprint together.
- **What Braintrust has that we don't** — this is the "platform" half, and it's worth being concrete about what that buys you:
  - **A prompt Playground.** A PM or engineer opens a browser, pastes a prompt, runs it against several models side-by-side, tweaks the wording, edits a scorer inline, and sees outputs change live — no code. In AuditKIT you'd edit a Python file and re-run.
  - **Production tracing / observability** (shared with the platforms below) — capturing and replaying real production requests. See the [observability-platforms note](#langsmith-arize-phoenix-wb-weave-mlflow-observability-platforms) for a concrete walk-through.
  - **CI release gating with PR feedback.** Its GitHub Action runs the eval suite on every pull request, **posts the scores as a PR comment**, and **blocks the merge** if a scorer drops below a threshold — turning "did this prompt change regress quality?" into a required check. AuditKIT can run in CI (it's a library), but you'd wire the reporting and the merge-blocking yourself.
  - **"Loop" assistant + hosted collaboration** — describe in plain language what you want to measure and it drafts scorers/datasets for you, in a shared team workspace with saved runs and dashboards.

  In short: AuditKIT is the *measurement engine*; Braintrust wraps a similar engine in a hosted product for teams, prompt-tuning, and production.

### DeepEval (Confident AI)
The most feature-dense open-source eval library: 50+ ready metrics (G-Eval, faithfulness, hallucination, answer relevancy, task completion), "Pytest-for-LLMs" ergonomics, synthetic-data generation, component-level tracing via `@observe`, multimodal + conversational + agent metrics, and a companion **DeepTeam** for red-teaming (40+ vulnerability categories), plus the hosted Confident AI platform.

**TL;DR:** DeepEval's metric catalog and synthetic data are broader, but AuditKIT is the one that also wraps a real academic-benchmark engine and gives you model-compression comparison with a reproducible fingerprint, out of the box.
- **What AuditKIT does that DeepEval doesn't:** an **integrated academic-benchmark engine** (`ak.run_lmeval`, the real lm-eval-harness), **model size/sparsity introspection** for compression comparison, and a **stable run fingerprint + disk cache** baked into the core data model (re-running an identical eval is a cache hit, and comparisons key off the fingerprint).
- **What DeepEval has that we don't** — and what each one actually lets you do:
  - **A larger ready-metric catalog (~50).** Beyond the families AuditKIT ships, DeepEval has metrics like *task completion*, *tool correctness*, and multi-turn *conversation* scores off the shelf. Concretely: to check that a support bot stayed on-policy across an 8-turn chat, DeepEval gives you a `ConversationalTestCase` + conversation metrics; in AuditKIT you'd write that scorer yourself.
  - **Synthetic data generation.** You don't have a test set? Point DeepEval at a few seed examples or your RAG document corpus and it *generates* one — evolving prompts into harder/varied cases and producing Q/A pairs grounded in your chunks. Example: turn 5 seed questions into a 200-case eval set, or auto-build a retrieval test set from your knowledge base. AuditKIT requires you to bring your own dataset.
  - **Shipped red-teaming (DeepTeam, 40+ vulnerability categories).** It *generates* adversarial attacks — prompt injection, jailbreaks, PII leakage, bias/toxicity — runs them against your app, and hands back a report like "leaked its system prompt on 3/40 injection attempts, produced disallowed content on 2/25 jailbreaks." AuditKIT ships basic probes/detectors but can't yet generate attacks at this scale.
  - **Agent evaluation.** Metrics that read a multi-step agent trace and score whether it called the *right tool with the right arguments*, completed the task, and planned efficiently. Example: verify a booking agent picked the correct API call and actually completed the booking. AuditKIT removed agent eval (future works).
  - **Multimodal & conversational evaluation.** Score image+text tasks (does the vision model's caption match the image?) and whole dialogues rather than single turns (role adherence, knowledge retention across turns). AuditKIT is single-turn, text-only today.

### Inspect (UK AI Safety Institute)
The closest architectural cousin: an open-source Python framework with a `dataset → Task → Solver → Scorer` spine (AuditKIT's is `Scenario → Adapter → Model → Metric`), 200+ prebuilt evals, agentic tool use with sandboxed (Docker) execution, one interface over many providers plus local vLLM/Ollama, and a log viewer. Strong for frontier-safety and agentic evals.

**TL;DR:** Inspect wins on sandboxed agentic evaluation, but for benchmark + judge + RAG + base-vs-compressed-model comparison in one library, AuditKIT is the more complete and more practical pick.
- **What AuditKIT does that Inspect doesn't:** a **wrapped lm-eval engine**, first-class **model comparison** (base vs pruned/quantized) with significance + a **size/latency/cost tradeoff**, and **model-size introspection**.
- **What Inspect has that we don't** — concretely:
  - **Agentic evaluation with sandboxed tool execution.** Inspect can run an agent that *actually executes* tools/code inside an isolated Docker container and then scores the whole trajectory. Example: evaluate a coding agent that writes a script, runs it in the sandbox, reads the error, fixes it, and you score whether it eventually produced correct, safe output. AuditKIT sends prompts to a model and scores the text back — it can't run an agent loop or execute tools.
  - **A large curated eval set (200+) and a polished log viewer** (a VS Code / web UI to step through every sample's transcript). If your work is agentic or frontier-safety evaluation, Inspect is more complete there today.

### Promptfoo
Open-source, **config-as-code (YAML)**, 50+ providers, and a strong **red-teaming** suite aligned to OWASP LLM Top 10 / NIST AI RMF, with an attacker-model config that generates adversarial tests. JS/TS-first, CLI-driven, CI-oriented.

**TL;DR:** Promptfoo's red-teaming is ahead today, but AuditKIT is the Python-native library with the richer `RunResult` — real academic benchmarks, real model comparison, and perf/size/cost measurement Promptfoo doesn't offer.
- **What AuditKIT does that Promptfoo doesn't:** a **Python-native library API** returning a rich `RunResult`, **academic benchmarks**, **model comparison with significance**, and **perf/size/cost** measurement.
- **What Promptfoo has that we don't:** production-grade **red-teaming**. Concretely: you point it at your app, pick attack plugins (aligned to OWASP LLM Top 10 / NIST), and it uses an "attacker" model to *auto-generate* hundreds of adversarial prompts, fires them at your app, and produces a vulnerability report by category (e.g., "prompt-injection: 12% success, PII-leak: 4%"). Plus a very polished YAML+CLI workflow for A/B-testing prompts across 50+ providers in CI. AuditKIT's red-teaming today is basic probes/detectors, not this scale.

### RAGAS & TruLens (RAG specialists)
RAGAS defined the standard RAG metrics (faithfulness, context precision/recall, answer relevancy) with a reference-free, claim-decomposition approach; TruLens centers on "feedback functions" and the RAG triad (groundedness / context relevance / answer relevance), now also usable as MLflow scorers.

**TL;DR:** If RAG is one metric family among several you care about, AuditKIT already covers it inside the same unified API you're using for everything else — no second library, no separate result format.
- **AuditKIT** ships the same RAG-metric *family* (`lexical_groundedness`, `context_coverage`, `context_overlap`, `answer_overlap`) as part of the unified API -- named for the lexical-overlap heuristic they actually implement rather than RAGAS's own metric names, since the underlying method differs (see RAGAS/TruLens comparison above) -- convenient if RAG is one of several things you evaluate. If RAG is your *entire* focus, RAGAS/TruLens go deeper (more RAG-specific metrics and tooling).

### LangSmith / Arize Phoenix / W&B Weave / MLflow (observability platforms)
These are **observability-first**. AuditKIT is not in this category and doesn't try to be — but "observability platform" is the phrase that most needs unpacking, because it's the biggest thing these tools do that AuditKIT doesn't. Concretely, an eval-**and**-observability platform gives you:

**TL;DR:** These solve a different problem — production monitoring, not offline measurement. AuditKIT is the zero-dependency library you'd actually run *inside* one of these (or in CI) for the measurement step none of them do on their own.

- **Production tracing (the headline feature).** It sits in your live app's request path and records **every real user request** as a *trace*: the exact prompt, the retrieved RAG chunks, each tool/function call and its result, intermediate agent steps, the final answer, latency, and token cost — all searchable in a dashboard by user, session, or time. *What you can do with it:* a user reports the chatbot gave a wrong answer at 2pm; you open the platform, find that exact conversation, and see that retrieval pulled the wrong document — so you know it's a retrieval bug, not the model. AuditKIT runs an **offline** eval over a dataset you hand it and returns a `RunResult`; it never sees your live traffic, so it can't do this.
- **Online / continuous evaluation.** Run scorers automatically on live production traffic (or a sample) and chart quality over time. *What you can do:* get **alerted** when faithfulness drops the day after a prompt change ships. AuditKIT is a one-shot offline run you trigger.
- **Dashboards & experiment UI.** A browser to browse runs, compare experiments side-by-side, filter to just the failing samples, and watch a metric trend across model versions. AuditKIT gives you a `RunResult` object + a text summary; any charting is on you.
- **Human review / annotation queues.** Route model outputs to human raters, collect thumbs-up/down or labels, and feed those back as ground truth (or to calibrate an LLM judge). AuditKIT has no human-in-the-loop UI.

LangSmith is closed-source and LangChain-centric; Phoenix is OSS and OpenTelemetry-native (a strong LangSmith alternative); Weave is W&B's; MLflow adds `mlflow.evaluate` + GenAI scorers to the MLflow ecosystem. **AuditKIT is the offline measurement library you'd run in CI or a notebook; a platform is what you'd add on top for production monitoring and team collaboration.** (In Lexsi's own architecture, that platform layer is a separate concern from this library.)

### OpenAI Evals & HELM (other academic tools)
OpenAI Evals is an OSS registry-based, model-graded framework centered on OpenAI models (narrower coverage). HELM (Stanford) is a *holistic* suite scoring accuracy **and** bias/toxicity/efficiency in one run, with domain variants (MedHELM, VHELM). Both are benchmark-oriented; AuditKIT reaches the same academic-benchmark territory through lm-eval while adding judge/RAG/custom/perf on top.

**TL;DR:** AuditKIT reaches the same benchmark territory as both, through the real lm-eval-harness, while adding judge/RAG/custom metrics/perf/comparison that neither OpenAI Evals nor HELM offers.

---

## 5. Where AuditKIT is behind (honest gaps)

- **No observability / tracing / dashboards / hosted UI.** If you need production monitoring, span-level tracing, or team collaboration in a browser, a platform (Braintrust, LangSmith, Phoenix, Weave) is the right tool; AuditKIT is a library.
- **Red-teaming is not a shipped, polished capability** (it's on the roadmap). Promptfoo and DeepTeam are far more complete for adversarial testing today.
- **No synthetic data generation** (DeepEval has this).
- **No human-annotation workflows.**
- **Smaller ready-metric catalog** than DeepEval's 50+ (AuditKIT ships ~40).
- **Python-only** (Braintrust and Promptfoo also serve JS/TS).
- **Multimodal / agent / conversation / tabular evaluation** are future works (removed to keep the current release focused); DeepEval and Inspect ship agent/multimodal today.
- **The native built-in benchmark scenario loaders are partly stale** — use the `ak.run_lmeval()` path for academic benchmarks.

---

## 6. When to choose which

- Choose **AuditKIT** when you want a standalone Python library to run benchmark + judge + RAG + custom evals through one reproducible API — especially to compare a base model against a pruned/quantized variant with quality, size, latency, and cost together, offline and in CI, with no platform to stand up.
- Choose lm-eval-harness / HELM for pure academic leaderboard benchmarking — though **AuditKIT** gives you lm-eval anyway, wrapped, plus judge/RAG/custom/comparison on top.
- Choose DeepEval for the largest ready-metric set, synthetic data, and shipped agent/multimodal/red-team coverage in a Pytest-style workflow — **AuditKIT** covers less breadth here but adds the wrapped academic-benchmark engine and model-compression comparison DeepEval doesn't have.
- Choose Inspect for agentic and frontier-safety evaluations with sandboxed tool execution — **AuditKIT** doesn't run agent loops or execute tools at all, so it isn't a substitute for this specific need.
- Choose Promptfoo / DeepTeam / Giskard when red-teaming and security are the priority — **AuditKIT** ships basic probes/detectors today, not yet at this scale, so it isn't a substitute here either.
- Choose RAGAS / TruLens when RAG evaluation is your entire focus — **AuditKIT**'s own RAG metric family covers this well if RAG is one of several things you evaluate, just with less RAG-specific depth.
- Choose Braintrust / LangSmith / Arize Phoenix / Weave when you need a hosted observability + evaluation platform with tracing, dashboards, and human review — **AuditKIT** is not a platform and isn't a substitute for this; it's the offline measurement step you'd run alongside one.

In one line: **AuditKIT** is the standalone, provenance-first library for measuring model quality *and* the cost of getting it — with a unique strength in comparing compressed models — and is complementary to (not a replacement for) the hosted observability platforms.

---

## Sources & further reading

Competitor feature summaries above are drawn from these public docs (as of mid-2026):

- **Braintrust** — https://www.braintrust.dev/ · autoevals: https://github.com/braintrustdata/autoevals
- **DeepEval** — https://deepeval.com/ · DeepTeam: https://www.trydeepteam.com/
- **Promptfoo** — https://www.promptfoo.dev/
- **Inspect (UK AISI)** — https://inspect.aisi.org.uk/
- **lm-evaluation-harness** — https://github.com/EleutherAI/lm-evaluation-harness
- **HELM (Stanford CRFM)** — https://crfm.stanford.edu/helm/
- **OpenAI Evals** — https://github.com/openai/evals
- **RAGAS** — https://docs.ragas.io/
- **TruLens** — https://www.trulens.org/
- **LangSmith** — https://docs.smith.langchain.com/ · **Arize Phoenix** — https://phoenix.arize.com/ · **MLflow eval** — https://mlflow.org/docs/latest/genai/

*Feature sets change; re-verify any competitor claim against its current docs before using this comparison externally.*
