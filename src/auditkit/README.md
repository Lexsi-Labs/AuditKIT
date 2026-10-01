# AuditKIT — Source Layout

```
auditkit/
├── __init__.py          # Public API exports, version, __all__
├── __main__.py          # python -m auditkit entry point
├── adapter.py           # Adapter ABC + 8 implementations (generation, chat, fewshot, tools, etc.)
├── annotator.py         # Annotator ABC for post-generation annotation
├── api.py               # evaluate(), evaluate_many(), generate(), compare(), run_lmeval()
├── cache.py             # DiskCache — fingerprint-keyed result cache
├── cli.py               # argparse CLI: eval, init, list, redteam, compare, agent
├── compat.py            # check_compat(), CompatReport: sglang/vllm/torch/transformers pin and platform check (stdlib only)
├── comparison.py        # RunComparison, MetricDelta, TaskDelta, grade_delta
├── diff.py              # RunDiff, DeltaGrade — comparison between runs
├── errors.py            # AuditKitError, CapabilityError, ExtraNotInstalled, etc.
├── evaluator.py         # Evaluator ABC for technique plugins
├── experiment.py        # Experiment, ExperimentDB — run management
├── lmeval_engine.py     # run_benchmark() — lm-evaluation-harness engine (extra)
├── loaders.py           # load_csv, load_jsonl, load_agenttune, load_hf, load_croissant
├── logs.py              # configure_logging
├── metric.py            # Metric ABC + ExactMatch, QuasiExactMatch, Acc, AccNorm
├── model_compare.py     # CompareResult, compare_models() — multi-model comparison
├── registry.py          # ObjectSpec, Registry — SCENARIOS, ADAPTERS, MODELS, etc.
├── report.py            # Prediction, RunResult — evaluation output
├── report_format.py     # Report — markdown formatting
├── router.py            # route_adapter() — auto-picks an Adapter from sample shape
├── runner.py            # Runner — 5-stage evaluation pipeline
├── runspec.py           # RunConfig, RunSpec, fingerprint
├── sample.py            # Sample
├── scenario.py          # Scenario ABC, ListScenario, CallableScenario
├── score.py             # Score, Stat, pass_at_k, pass_hat_k
├── scorers.py           # FunctionScorer, ScorerMetric, @scorer decorator
├── scoring.py           # ScoreGate, WeightedSum
├── trace.py             # ToolCall, to_turns, parse_tool_calls: tool-call turns, normalizing, bipartite matching
├── types.py             # Enums: TaskKind, ScoreKind, DataType, etc.
│
├── agent_eval/          # End-to-end agent evaluation (stdlib only)
│   ├── types.py         # AgentCase, AgentEvent, AgentEpisode, coverage markers, redact()
│   ├── importers.py     # AgentTune files, EventLog, OpenAI messages, Sample traces -> episodes
│   ├── outcome.py       # FinalState/Artifact/Answer assertions, CustomPredicate, judge fallback
│   ├── runner.py        # AgentEvalSpec, AgentEvalRunner (recorded/deployed), rescore
│   ├── report.py        # AgentCaseResult, AgentEvalResult: summary, drilldown, JSON
│   └── lexsi.py         # Lexsi evidence importer: SourceEvidence readers, join_evidence, lineage_sidecar
│
├── metrics/             # metric families
│   ├── code.py          # Contains, Equals, F1Score, IsJson, Levenshtein, etc.
│   ├── embedding.py     # CosineSimilarity, TokenOverlap, BM25Similarity
│   ├── generation.py    # Bleu, RogueL, ChrF, BertScore, Perplexity, WordErrorRate
│   ├── hallucination.py # FactualConsistency
│   ├── agent.py         # ToolCallF1, TrajectoryMatch, ParallelToolCalls, ToolCallValidity, RedundantToolCalls, TaskCompletion
│   ├── judge.py         # JudgeMetric, LLMJudge, GEval, RubricItem, Factuality, ClosedQA, Relevance
│   ├── pairwise.py      # WinRate, EloScore, PreferenceAccuracy
│   ├── perf.py          # LatencyStats, Throughput
│   ├── rag.py           # LexicalGroundedness, ContextCoverage, ContextOverlap, AnswerOverlap
│   ├── rag_judge.py     # Faithfulness, ContextPrecision, ContextRecall (LLM-judged)
│   ├── retrieval.py     # RetrievalMetrics: hit rate, precision, recall, MRR, AP, nDCG
│   ├── security.py      # DefconGrade, KeywordDetector, ThreatCategory
│   └── toxicity.py      # ToxicityScore, RepresentationSkew, HateSpeechScore
│
├── model/               # Model backends
│   ├── __init__.py      # Request, Generated, Result_, Model ABC, AutoModel, CallableModel, PrecomputedModel
│   ├── agent_endpoint.py # AgentEndpointModel: `agent:` deployed-agent backend (stdlib)
│   ├── anthropic.py     # Anthropic backend (extra)
│   ├── api_gen.py       # Generic OpenAI-compatible API backend, native tool calls (extra)
│   ├── groq_gen.py      # Groq backend (extra)
│   ├── hf_gen.py        # HuggingFace Transformers backend (extra)
│   ├── lexsi.py         # Lexsi gateway backend (extra)
│   ├── litellm_gen.py   # LiteLLM backend (extra)
│   ├── openai.py        # OpenAI backend (extra)
│   └── vllm_gen.py      # vLLM backend (extra)
│
├── redteam/             # Red teaming
│   ├── __init__.py      # Probe, Detector, RedTeamRunner, RedTeamResult exports
│   ├── probe.py         # Probe ABC + ProbeResult
│   ├── detector.py      # Detector ABC + DetectorResult
│   ├── runner.py        # RedTeamRunner, RedTeamResult — orchestration
│   ├── probes/
│   │   └── builtin.py   # PromptInjectionProbe, JailbreakProbe, EncodingProbe, RefusalProbe
│   └── detectors/
│       └── builtin.py   # KeywordDetector, RefusalDetector, InjectionSuccessDetector, etc.
│
└── scenarios/           # Built-in benchmark datasets
    ├── arc.py           # AI2 Reasoning Challenge
    ├── gsm8k.py         # Grade School Math 8K
    ├── hellaswag.py     # HellaSwag
    ├── humaneval.py     # HumanEval
    ├── mmlu.py          # Massive Multitask Language Understanding
    └── truthfulqa.py    # TruthfulQA
```

## Architecture (the spine)

Every evaluation flows through the same 5-stage pipeline:

```
Dataset → Adapter → Model → Metrics → RunResult
```

1. Build requests — `Adapter.adapt(sample, config) → list[Request]`
2. Execute — `Model.generate(requests) → list[Result_]`
3. Annotate — `Annotator.annotate(sample, results) → dict`
4. Score — `Metric.score(sample, output, context) → Score`
5. Aggregate — `Stat.add(value)` → per-metric stats

## Extending

**Add a metric:** Create a class inheriting `Metric`, implement `score()`, export via `__init__.py`.

**Add a model backend:** Create a file in `model/`, inherit `Model`, implement `generate()`, add prefix to `AutoModel._T1_BACKENDS`.

**Add a probe:** Create a class inheriting `Probe`, implement `prompts()`, register in the probe registry.

**Add a detector:** Create a class inheriting `Detector`, implement `detect()`, register in the detector registry.
