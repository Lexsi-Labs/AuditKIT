# Glossary

| Term | Definition |
|------|------------|
| **Sample** | One evaluation item, consisting of an input (the prompt or query) and an optional expected output for reference. |
| **Golden** | A sample with a known target answer, used as ground truth for scoring. |
| **Scenario** | A task definition plus its associated data; when instantiated, produces a list of Samples for evaluation. |
| **Model** | Abstract interface for batched text generation. Concrete backends (OpenAI, Anthropic, vLLM, HuggingFace, Echo, etc.) implement `generate()` and are resolved via `AutoModel` with prefixed identifiers like `openai:`, `hf:`, `vllm:`. |
| **Adapter** | Transforms a Sample into one or more model Requests (e.g., formatting a few-shot prompt or a chat turn list). |
| **Metric** | Scores a model's output against a sample, producing a Score. All metrics inherit from `Metric` and implement `score()`. |
| **Score** | A single evaluation result containing a numeric value, a kind label, and optional metadata (e.g., tokens used, latency, explanation). |
| **Stat** | Aggregated statistics for a metric across all samples in a run: count, mean, standard deviation, standard error, min, and max. |
| **RunResult** | The top-level output of `evaluate()`. Contains per-sample Predictions, per-metric Stats, the run's `RunConfig`, and a stable sha256 Fingerprint. |
| **RunSpec** | Everything needed to reproduce one evaluation run: dataset (scenario), model, adapter, metric list, annotators, and `RunConfig`. |
| **RunConfig** | The runtime knobs that shape how a run executes: concurrency, retries/timeout, sampling params (temperature, top_p, max_tokens, ...), split config, judge model. |
| **Fingerprint** | A deterministic sha256 hash that uniquely identifies a run. Derived from the model identity, dataset contents, metric parameters, and seed. Enables caching, comparison, and provenance tracking. |
| **Technique** | A family of evaluation approaches: benchmark, LLM-as-judge, RAG, security, performance. |
| **TaskKind** | The type of task being evaluated: `mcq`, `generative`, `rag`, `security`, `performance`, `language_modeling`. |
| **CompareResult** | Side-by-side comparison of multiple models on the same dataset and metrics, with bootstrap significance testing and per-metric deltas. |
| **RunComparison** | A baseline-vs-candidate diff of two runs: per-metric and per-task percentage deltas, retention, paired-bootstrap significance, and a per-sample browser of what regressed. |
