# Roadmap

## T0 — Core spine (current)

- [x] Sample, Score, Metric base classes
- [x] Model interface with AutoModel resolution
- [x] Runner (5-stage: build → execute → annotate → score → aggregate)
- [x] RunSpec fingerprinting for reproducibility
- [x] Zero required dependencies
- [x] CLI (eval, init, list, compare)
- [x] YAML config file support
- [x] Experiment tracking (ExperimentDB)

## T1 — Benchmark technique

- [x] MMLU, GSM8K, ARC benchmarks
- [x] Generation metrics (Bleu, RogueL, ChrF, etc.)
- [x] Code/deterministic metrics (Contains, Equals, Regex, etc.)
- [x] llm-evaluation-harness integration
- [x] Per-sample Predictions with answer browser
- [x] Rich HTML reports

## T2 — Judge + RAG + user datasets

- [x] LLM-as-judge (GEval, RubricItem)
- [x] RAG quartet (LexicalGroundedness, ContextOverlap, ContextCoverage, AnswerOverlap)
- [x] Hallucination detection (FactualConsistency)
- [x] Weighted scoring and custom rubrics (`WeightedSum`, `RubricItem(weight=...)`)
- [ ] User dataset versioning
- [ ] Prompt playground

## T3 — Comparison and regression

- [x] Model comparison (compare_models)
- [x] Bootstrap significance tests
- [x] Baseline comparison with per-task deltas (`RunComparison`)
- [ ] Cross-run comparison dashboard
- [ ] CI gating

## T6 — Security + performance

- [x] Security metrics (KeywordDetector, DefconGrade)
- [x] Performance metrics (LatencyStats, Throughput)
- [ ] Red teaming — adversarial-probing suite (probes, detectors, runner)
- [ ] Performance profiling (TTFT, ITL, TPOT distributions)
- [ ] Load testing profiles
