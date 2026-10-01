# Known issues

Open problems and limitations in the current code. The
[Changelog](community/changelog.md) records what changed and when. For the
`vllm:` backend's own gaps and environment issues, see
[vLLM Known Issues](VLLM_KNOWN_ISSUES.md).

## Open issues

| Issue | Where | Detail |
|---|---|---|
| No annotator integration for 3 metrics | `metrics/pairwise.py` | `win_rate`, `elo_score`, `preference_accuracy` all read fixed top-level `context` keys (`candidates`, `pairwise_results`, `preference_data`) that an `Annotator`'s output never populates. `extract_with=` has no effect on any of these — see [Annotators](annotators.md). |
| `RunResult.model_spec` doesn't serialize cleanly | `src/auditkit/report.py` | `RunResult.to_dict()` stores the live `Model` instance, not the original spec string — `json.dump(..., default=str)` produces an object-repr string (e.g. `"<auditkit.model.groq_gen.GroqModel object at 0x...>"`). Everything else on `RunResult` round-trips through save/load. |
| `num_completions`/`best_of` don't affect the score; `RunConfig.extra` is unused | `src/auditkit/runner.py` | Both reach backends that support them, but scoring only reads `completions[0]`, so extra completions change cost, not the score. `RunConfig.extra` is hashed into the fingerprint but nothing reads it. |
| Two percentile formulas | `score.py`, `metrics/perf.py` | `Stat.percentile()` interpolates linearly; `LatencyStats` (used for `RunResult.perf["latency_ms"]`) uses index truncation, so their p95/p99 differ on the same data (e.g. `[10, 20, 30, 40, 100]`: 88 vs 40). At the default `concurrency=1` a run has one timed call, so it only shows at `concurrency>1`. |
| `Throughput` overcounts elapsed time under concurrency | `metrics/perf.py` | Requests and tokens per second divide by the **sum** of per-call durations, not wall-clock time, so with `concurrency>1` they understate real throughput. |

## Limitations

- **`lexsi:` doesn't apply a chat template to flat prompts.** `hf:` and `vllm:` render the model's own chat template (including Cohere's named templates), and the hosted chat APIs (`openai:`/`anthropic:`/`groq:`) take chat messages; `lexsi:` still sends the raw flat prompt, so an instruct model served that way is under-formatted. See [vLLM Known Issues](VLLM_KNOWN_ISSUES.md) for the `vllm:`-specific gaps that remain (GPU memory not freed by `evaluate()` alone, `model_info()` fallback status) and the dependency/environment issues.

- **Native tool calling works only on `api:`, `agent:`, `hf:` and `vllm:`.** They are the only backends that declare `Capability.TOOLS` (`hf:` and `vllm:` render the schemas through the chat template, so the template must support tools; none of the Cohere models' templates do, so use `mode="prompt"` with Tiny Aya, Aya Expanse, Aya Vision and North). With `ToolCallAdapter()` (native mode) on `openai:`, `anthropic:`, `groq:`, `openrouter:` or `litellm:`, the Runner raises `CapabilityError` instead of silently dropping the tools. Use `api:` with the provider's OpenAI-compatible base URL, or `ToolCallAdapter(mode="prompt")`, which describes the tools in the system prompt and parses `<tool_call>` blocks from the text. See [Agents & RAG](agents_and_rag.md).
- **A single-response model shows one turn.** `api:` (and any text backend) makes one request per sample, so it only produces the model's next turn; AuditKit does not run the tool loop. With a multi-turn `expected_tool_calls`, the dependent later turns count as missed. Keep only the first reference turn for single-step evals, or evaluate a full agent through `agent:` or a recorded `actual_trace`.
- **Judge claim extraction is non-deterministic.** `Faithfulness` asks the judge to split the answer into claims, and two runs (or two judges) can split the same answer differently, so the score can move without the answer changing. `temperature` defaults to `0.0`, which reduces but doesn't remove this. All retrieved contexts also go into one judge prompt; set `max_context_chars` for a small judge.
- **`HFGenModel.loglikelihood()`** does one forward pass per request, not batched across a request list.
- **`representation_skew`** measures demographic-*representation* balance, not bias. By design it can't see meaning — a sentence mentioning men and women equally scores `0.0` even if blatantly sexist — and it's a per-sample signal best read in aggregate over a run. For biased *content*, use `bias_judge` (LLM-as-judge); for whether the model *treats groups differently*, run a counterfactual benchmark (BBQ / CrowS-Pairs via `run_lmeval`).
- **`win_rate`/`elo_score`/`preference_accuracy`** silently fall back to a crude token-overlap proxy score whenever the caller doesn't manually populate the relevant `context` key (`candidates`/`pairwise_results`/`preference_data`) — easy to use without realizing you're getting the degraded fallback.
- **Bootstrap significance is a conservative heuristic.** `paired_bootstrap` (used by `RunComparison.significance`, `CompareResult.significance`, `Experiment.significance`) resamples the per-sample diffs and compares each resample's mean magnitude to the observed mean — so a large but *low-variance* difference (e.g. every sample flips the same way) reports `significant=False`. Treat it as a rough guard, not a rigorous test.
- **Model comparison scores each run independently, then diffs.** Pairwise-preference judging (`metrics/pairwise.py`) isn't wired into `RunComparison`. `RunComparison.tradeoff()` adds latency and size ratios from each run's measured `RunResult.perf` and `RunResult.model_size` when both sides have them (`RunConfig.track_performance`, on by default).
