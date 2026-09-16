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
- **security** — DefconGrade, KeywordDetector, ThreatCategory
- **toxicity** — ToxicityScore, RepresentationSkew, HateSpeechScore
- **guard** — GuardJudge (safety scoring via a guard model, e.g. Llama Guard)
