# Known issues

A current source of truth for problems and limitations found via
live testing (not inferred from reading code), as of 2026-07-16. Distinct
from the [Changelog](community/changelog.md), which records what changed and
when — this page tracks what's still open right now.

For a flat, single-purpose bug list with copy-pasteable reproductions (no
narrative, no doc-debt/never-verified sections), see [Bugs](BUGS.md) —
re-verified live as of 2026-07-30, including two items not in this page's
table (`Stat`/`LatencyStats` percentile disagreement, `Throughput` overcounting
under concurrency).

For everything specific to the `vllm:` backend (implementation gaps, plus
real dependency/environment issues found running it live for the first
time on a real GPU), see [vLLM Known Issues](VLLM_KNOWN_ISSUES.md).

## Confirmed bugs, unfixed

| Issue | Where | Detail |
|---|---|---|
| No annotator integration for 3 metrics | `metrics/pairwise.py` | `win_rate`, `elo_score`, `preference_accuracy` all read fixed top-level `context` keys (`candidates`, `pairwise_results`, `preference_data`) that an `Annotator`'s output never populates. `extract_with=` has no effect on any of these regardless of configuration — see [Annotators](annotators.md). |
| The 6 built-in benchmark scenarios fail to load | `src/auditkit/scenarios/` | `mmlu`/`gsm8k`/`arc`/`hellaswag`/`truthfulqa`/`humaneval` all point at stale/unqualified HuggingFace dataset references that fail with the `datasets`/`huggingface_hub` versions in a typical current environment (`HfUriError`/`DatasetNotFoundError`). The same class of issue was independently hit again testing RAG with bare `squad` — the canonical namespaced ID (`rajpurkar/squad`) works; the built-in scenarios need the equivalent fix. |
| `RunResult.model_spec` doesn't serialize cleanly | `src/auditkit/report.py` | `RunResult.to_dict()` stores the live `Model` instance, not the original spec string/dict — `json.dump(..., default=str)` produces an unusable object-repr string (e.g. `"<auditkit.model.groq_gen.GroqModel object at 0x...>"`). Everything else on `RunResult` (`config`, `headline`, `predictions`) round-trips correctly through save/load. |
| `num_completions`/`best_of` reach the API but don't affect the score; `RunConfig.extra` unused | `src/auditkit/runner.py`, `src/auditkit/report.py` | Adapters now forward all generation params consistently (see "Fixed recently"), so `num_completions`/`best_of` do reach a backend that supports them — but the scoring path only ever reads `completions[0]`, so extra completions change cost/output without changing the score. `RunConfig.extra: dict` is declared and hashed into the fingerprint but never read anywhere in the codebase. |

## Real limitations — by design or upstream, not bugs to fix

- **`vllm:`/`lexsi:` backends don't apply a chat template to flat prompts.** The `hf:` backend (via the model's own `tokenizer.chat_template`) and the hosted chat APIs (`openai:`/`anthropic:`/`groq:`) format any adapter's prompt correctly for an instruct model; `vllm:` and `lexsi:` still send the raw flat prompt, so an instruct model served that way is under-formatted. Extending the same wrap-as-user-turn rendering to those two backends is the follow-up. See [vLLM Known Issues](VLLM_KNOWN_ISSUES.md) for this and every other `vllm:`-specific gap (GPU memory leak via `evaluate()`, `model_info()` fallback status, and the dependency/environment issues hit getting it running at all).

- **`HFGenModel.loglikelihood()`** does one forward pass per request, not batched across a request list — correctness was prioritized for the first implementation; batching ragged prompt/continuation lengths correctly is additional surface area for a numerical bug.
- **`representation_skew`** (which replaced the mislabeled `bias_score`) measures demographic-*representation* balance, not bias. By design it can't see meaning — a sentence mentioning men and women equally scores `0.0` even if blatantly sexist — and it's a per-sample signal best read in aggregate over a run. This is now explicit in the name/docstring rather than a hidden flaw. For biased *content*, use `bias_judge` (LLM-as-judge); for whether the model *treats groups differently*, run a counterfactual benchmark (BBQ / CrowS-Pairs via `run_lmeval`).
- **`win_rate`/`elo_score`/`preference_accuracy`** silently fall back to a crude token-overlap proxy score whenever the caller doesn't manually populate the relevant `context` key (`candidates`/`pairwise_results`/`preference_data`) — easy to use without realizing you're getting the degraded fallback.
- **Bootstrap significance is a conservative heuristic.** `paired_bootstrap` (used by `RunComparison.significance`, `CompareResult.significance`, `Experiment.significance`) resamples the per-sample diffs and compares each resample's mean magnitude to the observed mean — so a large but *low-variance* difference (e.g. every sample flips the same way) reports `significant=False`. Treat it as a rough guard, not a rigorous test; a proper permutation/sign test is a follow-up.
- **Model comparison is quality-only unless you supply sizes.** `RunComparison.tradeoff()` computes size/latency ratios only from numbers you pass in — the library never measures model size or latency itself. First-class size/latency capture on `RunResult` (and lineage-based *auto* baseline selection) are deferred to the platform. Comparison is also independent-scoring-then-diff; pairwise-preference judging (`metrics/pairwise.py`) is not wired into it.

## Documentation debt

A full pass over every user-facing doc (this repo's root/examples/
applications READMEs, the entire mkdocs `docs/` site, and the `src/auditkit/*/README.md`
files) found and fixed a large batch of stale/fabricated content: the
`ak.benchmark()` → `ak.run_lmeval()` rename left two stale strings in
actual source code (not just docs); `docs/cookbook/llm_judge.md` and
`docs/integrations.md` used a fabricated API (`@ak.model`,
`ak.run_experiment()`, wrong `GEval`/`WordCount` kwargs, a nonexistent
`ollama:`/`sentence-transformers:` model prefix — the latter also present
in the real `cli.py` help text, now fixed); `docs/scenarios.md` had a
nonexistent `ARCScenario(challenge=...)` param and a broken top-level
import; `docs/scorers_reference.md`/`docs/annotators.md` referenced 6
metrics (`tool_correctness`, `trajectory_match`, `step_efficiency`,
`coherence`, `turn_taking`, `context_adherence`) deleted from the
codebase entirely; `docs/community/changelog.md` still advertised those
same deleted features. See recent commits for the full list — this
section intentionally stays short rather than re-deriving as a static
snapshot that will itself go stale.

## Never verified this session (not necessarily broken — just unconfirmed)

- **`Experiment`/`ExperimentDB`** (leaderboards, significance testing), **MLflow logging** — not touched at all in the most recent testing pass.
- **`LatencyStats`/`Throughput`** (`src/auditkit/metrics/perf.py`) — not read or tested at all.

## Fixed recently (for context — see the [Changelog](community/changelog.md) for full detail)

`Runner.score_one()` silently masking `ExtraNotInstalled` as a fake `0.0`
score; `Perplexity` returning `nan` on short outputs; `WordErrorRate` going
negative; `Faithfulness`/`ContextPrecision` substring-matching false
positives; `ToxicityScore`'s keyword-only detection; `RunConfig.concurrency`'s
unsafe default; 14 built-in metrics with constructor config invisible to
run fingerprinting; `Runner.execute()` now enforces `Model.threadsafe` (and
never builds empty chunks); adapters now forward one consistent generation-param
set (`presence_penalty`/`frequency_penalty`/`top_k`/… no longer silently dropped
by ChatAdapter/FewShot/Instruction/RAG/Template while GenerationAdapter kept them).
Extended the `ExtraNotInstalled` fix to *any* metric exception: `score_one()`
used to fake a `0.0` `Score` when a metric crashed (indistinguishable in every
downstream mean from "the model's output genuinely scored zero") — it now
skips that metric for that sample instead (still recorded in `errors`), so a
computation failure no longer masquerades as a real, bad score. `RunComparison`
(`comparison.py`) also now surfaces per-metric sample counts/std alongside
every delta (`MetricDelta.baseline_count`/`.candidate_count`/`.baseline_std`/
`.candidate_std`, and `coverage_warnings()`), so a metric silently computed
over fewer samples than another — whether from this fix, differing
`RunConfig.limit`s, or any other cause — is now visible instead of invisible.
`bert_score` was previously documented here as blocked by an unfixable
`bert_score`/`transformers>=5` incompatibility — root-caused and fixed
instead (see [Bugs](BUGS.md#bert_score-was-thought-unfixable-from-this-repo--it-wasnt)):
a scoped monkeypatch in `BertScore.score()` clamps the affected tokenizer's
`model_max_length` before use. `HFGenModel` now honors `stop_sequences`/
`seed`; `VLLMModel.model_info()` now attempts real introspection with
graceful fallback (unverified against a real vLLM install, still
CUDA/Linux-only); `ChatAdapter`'s stale comment corrected. See
[Bugs](BUGS.md) for all three.

## How to check current status yourself

Most of the "confirmed bugs" above were found via direct reproduction, not
just reading code — the pattern is worth reusing if something on this list
looks fixed by the time you're reading it, or if you find a new one:

```python
import auditkit as ak
from auditkit.runspec import RunSpec
from auditkit.scenario import ListScenario
from auditkit.sample import Sample

# Runner._chunk() itself still splits into exactly n_chunks regardless of
# item count if called directly -- but Runner.execute() (the only real
# caller) always clamps n_chunks = min(concurrency, len(flat)) first, so
# this can no longer produce empty batches through an actual run:
from auditkit.runner import Runner
print(Runner()._chunk([1], 8))  # [[1], [], [], [], [], [], [], []] -- raw method, not what execute() does
```
