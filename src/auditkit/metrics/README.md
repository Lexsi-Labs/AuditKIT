# Metrics

Built-in metric families for evaluating model outputs:

- **code** — Contains, Equals, F1Score, IsJson, Levenshtein, Regex, StartsWith, EndsWith, WordCount
- **embedding** — CosineSimilarity, TokenOverlap, BM25Similarity
- **generation** — Bleu, RogueL, ChrF, BertScore, Perplexity, WordErrorRate
- **hallucination** — FactualConsistency
- **judge** — JudgeMetric, LLMJudge, GEval, RubricItem, Factuality, ClosedQA, Relevance, BiasJudge
- **pairwise** — WinRate, EloScore, PreferenceAccuracy
- **perf** — LatencyStats, Throughput
- **rag** — LexicalGroundedness, ContextCoverage, ContextOverlap, AnswerOverlap
- **retrieval**: RetrievalMetrics (hit rate, precision, recall, MRR, average precision, nDCG, with optional `@k`)
- **rag_judge**: Faithfulness, ContextPrecision, ContextRecall, ContextRelevance, AnswerRelevancy, ResponseGroundedness, Hallucination (LLM-judged, numbered verdicts)
- **rag_stress**, **rag_stress_runner**, **rag_stress_ops** — deterministic RAG stress checks (EvidenceSetRecall, Freshness, AclCompliance, CitationSupport, Abstention, ContextRetention, NumericAccuracy), the paired stress runner, judge calibration and human review
- **agent**: ToolCallF1, TrajectoryMatch, ParallelToolCalls, ToolCallValidity, RedundantToolCalls, TaskCompletion, ToolSelectionJudge, ToolPermission, AgentLoopDetection
- **security** — DefconGrade, KeywordDetector, ThreatCategory
- **toxicity** — ToxicityScore, RepresentationSkew, HateSpeechScore
- **guard** — GuardJudge (safety scoring via a guard model, e.g. Llama Guard)
- **encoder_judge** — EncoderJudge, FactualityEncoderJudge, SentimentEncoderJudge (encoder classifiers as judges)
