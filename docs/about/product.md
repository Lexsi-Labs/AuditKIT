# Product Overview

AuditKIT is a unified evaluation library for any model, any dataset, and any task, with zero required dependencies and a provenance-first data model.

---

## Vision

**A single evaluation spine for AI models that works across every technique, modality, and team role.**

Today, evaluating AI models means stitching together a fragmented toolchain: academic benchmarks (MMLU, GSM8K) use one framework, LLM-as-judge evaluations require another, and red-teaming or security assessments each come with their own bespoke harness. Results across these tools cannot be compared, combined, or traced back to a shared provenance model.

AuditKIT replaces this fragmentation with one library that provides:

| Capability | Today's approach | AuditKIT |
|---|---|---|
| Academic benchmarks | `lm-eval-harness`, HELM | Built-in scenarios (MMLU, GSM8K, ARC, HellaSwag, TruthfulQA) via registry |
| LLM-as-judge | GPT eval, custom scripts | `GEval` + `RubricItem` with any judge model |
| Safety / security metrics | Garak, PyRIT, custom | `KeywordDetector`, `DefconGrade`, toxicity/bias/hate-speech scorers (red-team probe suite on the roadmap) |
| Performance profiling | Custom benchmarking | `LatencyStats`, `Throughput`, token timing |
| Post-operation eval | Manual, disconnected | Single `evaluate()` call after any operation |

## Personas

| Persona | Goal | Pain point | How AuditKIT helps |
|---|---|---|---|
| **Model optimizer** (fine-tuning, quantization, pruning) | Measure impact of each optimization step on model quality | No fast, reproducible eval loop; manual comparison across steps | `evaluate()` after any operation; `RunDiff` for pairwise comparison; `fingerprint` for provenance |
| **ML engineer** (integration, deployment) | Validate model quality in CI/CD, ensure no regressions | Eval scripts drift from benchmark configs; no cache across runs | CLI + YAML config; stable fingerprint caching; experiment tracking with `ExperimentDB` |
| **Project lead / reviewer** | Compare model versions, approve releases, audit vendor models | No unified dashboard; results in different formats per tool | `compare_models()` with bootstrap significance; `Report` with per-metric breakdowns |
| **OSS practitioner** | Evaluate models from HuggingFace, replicate leaderboards, contribute benchmarks | High barrier to entry; heavy dependency chains; closed-source tooling | `pip install auditkit` with zero deps; `hf:` model backend; scenario registry; library + CLI |

## Core Jobs-to-be-Done

| Job | Trigger | Success metric |
|---|---|---|
| **Evaluate after any operation** | After fine-tuning, pruning, quantization, or prompt change | Single `evaluate()` call returns `RunResult` with per-metric scores |
| **Compare against baselines** | Before deploying a new model version | `RunDiff` highlights regressions and improvements per metric |
| **Detect regressions** | CI/CD gate, pre-merge check | Automated pass/fail per metric threshold; fingerprint-driven caching |
| **Reproduce published results** | Replicating a paper or leaderboard | Built-in scenarios + model backends produce identical fingerprints |
| **Audit a third-party model** | Vendor evaluation, safety review | Standardized security metrics (DEFCON grade) + toxicity/bias scorers |
| **Track quality over time** | Ongoing model development | `ExperimentDB` stores every run; `compare()` produces leaderboard |

## User Journeys

### Journey 1: Post-operation evaluation

A model optimizer applies quantization to a model and wants to measure quality degradation.

```python
import auditkit as ak

# Before quantization
baseline = ak.evaluate(dataset, model="hf:Qwen/Qwen2.5-7B")

# After quantization
quantized = ak.evaluate(dataset, model=quantized_model)

# Compare
diff = ak.compare(baseline, quantized)
print(diff.summary())  # side-by-side with deltas
```

### Journey 2: Standalone benchmark

An ML engineer benchmarks a model on MMLU and GSM8K from the CLI.

```bash
pip install auditkit
auditkit eval --dataset mmlu --model openai:gpt-4o --output results.json
auditkit eval --dataset gsm8k --model openai:gpt-4o --output results.json
```

### Journey 3: OSS practitioner workflow

A community member clones a model from HuggingFace and evaluates it with a custom scenario.

```python
import auditkit as ak

samples = [
    ak.Sample(input="What is the derivative of x²?"),
    ak.Sample(input="Explain gradient descent."),
]
result = ak.evaluate(
    samples,
    model="openai:gpt-4o-mini",
    scorers=[ak.GEval(
        rubric=[ak.RubricItem(criterion="correctness", weight=1.0)],
        judge_model="openai:gpt-4o-mini",
    )],
)
```

### Journey 4: CI/CD regression gate (planned — T5, not yet implemented)

A project lead wants automated eval gates in a deployment pipeline. The
threshold-based pass/fail YAML config and a `--gate` CLI flag don't exist
yet (see the [roadmap](../community/roadmap.md)'s T5 tier). Today, the
same regression-gating job is done in code via `RunComparison`/`RunDiff`:

```python
import auditkit as ak

baseline = ak.evaluate(dataset, model="hf:Qwen/Qwen2.5-7B", scorers=["exact_match", "f1_score"])
candidate = ak.evaluate(dataset, model=candidate_model, scorers=["exact_match", "f1_score"])

cmp = ak.compare(baseline, candidate)
if cmp.grade() != ak.DeltaGrade.PASS:
    raise SystemExit(f"regression gate failed: {cmp.summary()}")
```

## Scope: Feature Tiers

| Tier | Focus | Features | Status |
|---|---|---|---|
| **T0** | Core evaluation spine | `evaluate()`, `Sample`, `RunResult`, `Metric` ABC, `CallableModel` | ✅ Done |
| **T1** | Production-hardening | Dataset splitting, concurrent execution, error handling + retry, model backends (OpenAI, Anthropic, HF, Lexsi, vLLM, LiteLLM, Groq), built-in benchmarks/scenarios, experiment tracking, generation metrics (BLEU, ROUGE, ChrF, WER, perplexity, BERTScore) | ✅ Done |
| **T2** | Rich evaluation types | LLM-as-judge (`GEval`), RAG evaluation, embedding similarity | ✅ Done |
| **T3** | Safety + security | Toxicity/bias metrics, security metrics (DEFCON grade), hallucination detection | ✅ Done |
| **T3+** | Adversarial red-teaming | Probe/detector suite (prompt-injection / jailbreak / encoding / over-refusal) | 🔮 Planned |
| **T4** | Performance | Latency/throughput profiling, cost tracking | 🛠 Partial |
| **T5** | Governance + compliance | Lineage graphs, policy-as-code, audit trails, data minimization attestation, bias monitoring | 🔮 Planned |
| **T6** | Ecosystem + scale | Plugin system, distributed execution, cloud-native experiment DB, leaderboard hosting | 🔮 Planned |

## Feature Coverage vs Other Tools

| Capability | AuditKIT | `lm-eval-harness` | HELM | LangSmith | DeepEval | Garak |
|---|---|---|---|---|---|---|
| Academic benchmarks (MMLU, GSM8K, etc.) | ✅ Built-in | ✅ Rich | ✅ Rich | ❌ | ✅ Some | ❌ |
| LLM-as-judge / rubric eval | ✅ GEval | ❌ | ❌ | ✅ | ✅ | ❌ |
| Red-teaming / adversarial probes | 🔮 Planned | ❌ | ❌ | ❌ | ❌ | ✅ Probes |
| Security metrics (DEFCON grade) | ✅ Built-in | ❌ | ❌ | ❌ | ❌ | ✅ ASR |
| Performance profiling | ✅ Latency + throughput | ❌ | ✅ | ✅ Traces | ❌ | ❌ |
| **Post-operation eval + lineage** | ✅ Evaluate after any op + fingerprint | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Governance / audit trails** | 🔮 T5 planned | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Zero required dependencies** | ✅ Core on stdlib | ❌ Heavy | ❌ Heavy | ❌ SDK | ❌ | ❌ |
| **Standalone OSS library** | ✅ `pip install auditkit` | ✅ | ❌ SaaS | ❌ SaaS | ✅ | ✅ |
| **Consistent API across all techniques** | ✅ One `evaluate()` | ❌ Per-task | ❌ Per-task | ❌ Per-product | ✅ Partial | ❌ Per-probe |
| **Provenance / caching** | ✅ sha256 fingerprint | ❌ | ❌ | ❌ | ❌ | ❌ |

## Design Principles

### Zero required dependencies

The core evaluation spine runs entirely on the Python standard library. Heavy model backends (OpenAI, Anthropic, vLLM, LiteLLM, HuggingFace Transformers) and computationally intensive metrics (BERTScore, embedding similarity, perplexity) are optional extras installed via `pip install auditkit[extra]`. This means:

- `pip install auditkit` takes seconds, not minutes
- No risk of dependency conflicts with existing projects
- CI/CD pipelines stay lean
- Contributors can run the test suite (750+ tests) without any third-party packages

### Provenance-first

Every `evaluate()` call produces a `RunResult` containing a stable sha256 fingerprint derived from:

- Model identity and configuration
- Dataset samples (via hash)
- Metric list and parameters
- Random seed

This fingerprint enables:

| Feature | Mechanism |
|---|---|
| **Caching** | Same fingerprint skips re-execution |
| **Comparison** | `RunDiff` operates on fingerprints, not timestamps |
| **Reproducibility** | Fingerprint + `RunConfig` fully captures an eval run |
| **Lineage** | Chain of fingerprints tracks model → operation → eval |
| **CI/CD gating** | Gate thresholds by fingerprint, not by run ID |

### Consistent API across all techniques

Every evaluation technique (benchmark, judge, RAG, security, performance) uses the same `evaluate()` entry point and returns the same `RunResult` structure. This means:

```python
# Benchmark eval
result = ak.evaluate(mmlu_samples, model="openai:gpt-4o")

# Judge eval — same API
result = ak.evaluate(samples, model="openai:gpt-4o", scorers=[ak.GEval(rubric=[...])])

# All produce RunResult — compare across techniques
```

### Library + CLI + YAML

AuditKIT works as a Python library, a CLI tool, or with declarative YAML config files, whichever fits the workflow.

```
Library:   ak.evaluate(dataset, model="hf:gpt2")
CLI:       auditkit eval --model hf:gpt2
YAML:      auditkit eval --config eval.yaml
```

### Extensible by design

- **Metrics:** Subclass `Metric` and implement `score()` — registered automatically
- **Scenarios:** Decorate with `@SCENARIOS.register("name")` — available via `--dataset`
- **Model backends:** Subclass `Model` with `generate()` — resolved via `AutoModel.resolve()` with prefix syntax (`openai:`, `hf:`, `vllm:`, etc.)
- **Adapters:** Subclass `Adapter` — turns a `Sample` into model `Request`s (few-shot, chat, RAG, template)

---

*See the [roadmap](../community/roadmap.md) for planned milestones and the [changelog](../community/changelog.md) for release history.*
