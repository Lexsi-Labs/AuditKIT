# Changelog

## Unreleased

### Added

- `RunConfig.chat_template_kwargs`: passed to the chat template of every request on
  `hf:` and `vllm:`, and sent as `chat_template_kwargs` to `api:` servers (vLLM /
  SGLang), e.g. `{"enable_thinking": False}` for Qwen3. A request's own values win.
  Part of the run fingerprint.
- `api:` says when a server may have ignored `chat_template_kwargs` or
  `parallel_tool_calls`: `RunResult.metadata["unverified_request_fields"]`, a
  `summary()` line, one warning, and `cap_unverified` on the parallel scores. The
  server is probed once (`GET /models` `owned_by`) and recorded in
  `metadata["api_server"]`; only `APIModel(server="sglang"|"vllm")` clears the note.
  New `Model.run_notes()` hook for backend facts about a run.
- `ak.load_bfcl()` and `ak.bfcl_arg_match`: BFCL v3's 11 single-turn
  categories plus the two irrelevance ones, read from the Hub files (JSON Lines). The
  accepted-value lists stay on the reference, and `bfcl_arg_match` compares against
  them as BFCL's AST checker does. It handles a drifted answer id, flags unsatisfiable
  answers, passes system prompts through, converts BFCL types to JSON Schema, and has an
  optional `dots_to_underscores`. Multi-turn is refused: its answers are call strings.
- `RunResult.route_distribution()` and a `summary()` line: for samples whose
  reference names several routes (`any_of`), which route each took and which rule
  decided (`f1`, `recall`, `turn_structure`, `call_order`, `listed_order`), with the
  ties broken by listed order counted. Information only: it never reaches the stats.
  Multi-route scores carry `path_decided_by` in their metadata.
- `check_compat()` rule (i): detects SGLang's `UnifiedRadixCache.dec_lock_ref()` bug
  from the installed source. Every multimodal model on SGLang's Transformers backend
  (Aya Vision, measured on 0.5.20) loads and then crashes on its first request; the bug
  is also on sglang `main`. `python compat.py --patch-sglang`, run in the SGLang env,
  fixes it idempotently (`auditkit.compat.patch_sglang()`). Patched, Aya Vision 8B scores
  5/5 on text and passes an image check on SGLang.
- `check_compat()` / `--patch-sglang` rule (l): SGLang's `Cohere2ForCausalLM` (Tiny Aya,
  Command R7B) never reports its sliding window to the runner. With Triton attention (the
  fallback where FlashInfer can't build), every sliding-window layer then crashes with
  `kv_indptr=None`. The patch adds `get_attention_sliding_window_size()`, returning the
  layers' own `config.sliding_window`. Rule (m) does the same for SGLang's generic
  Transformers backend (Aya Vision: Triton decode crashed with `kv_indptr=None`),
  returning a window only for models with sliding-window layers.
- The `vllm:` column covers all nine Cohere models: the full matrix ran on an RTX PRO 6000
  (CUDA 12.8) with the FlashInfer sampler fix, every check passing. Aya Expanse 32B and Aya
  Vision 32B score 5/5; the other seven match their A100 scores exactly.
- `vllm:` on SM 12.x GPUs (RTX PRO 6000, RTX 50-series) with a CUDA toolkit older than 12.9:
  vLLM's default FlashInfer sampler can't compile there, and every model failed to load with
  "FlashInfer requires GPUs with sm75 or higher" (Colab, CUDA 12.8). The backend now sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` before the engine starts (never overriding a value you set) and
  reports it in `RunResult.metadata["vllm_launch_env"]`. `compat.vllm_launch_advice()` /
  `compat.py --vllm-launch` give the same for `vllm serve`; `check_compat(check_cuda=True)` rule
  (o) warns. One check, `compat.flashinfer_jit_blocked()`, now drives both this and SGLang's
  Triton advice.
- `--patch-sglang` rule (n): North Micro Vision's `rope_parameters` is keyed by layer
  type, with `None` for its full-attention layers (no RoPE). SGLang's Transformers
  backend calls `.get` on every entry while checking torch.compile, so North failed with
  `'NoneType' object has no attribute 'get'` (measured: RTX PRO 6000, transformers 5.16.1).
  The patch skips entries without RoPE; a dynamic-RoPE entry still disables compile.
  Patched, North works on SGLang (text, image and layout checks pass), as does Aya Vision
  32B: all nine Cohere models now run on SGLang 0.5.20. `colab_sglang_32b_north.ipynb`
  measures the two.
- `docs/model_backends.md`: the `vllm:` column is measured for the seven Cohere models that fit
  a 40 GB A100 (vLLM 0.30.0): Tiny Aya 4/5 each, North 5/5, Aya Expanse 8B and Aya Vision
  8B 5/5, image checks passing. vLLM falls back to its Transformers backend for Aya Vision on
  its own.
  "Aya Vision loads on neither vLLM nor SGLang" is corrected: it loads on both, non-natively.
  The images section and README now say `api:` sends images and the offline
  `vllm:` backend still drops them; the install trap names the cause (a stale auditkit 1.0.0).
- `check_compat()` rule (j): with transformers >= 5.15 in the SGLang env (needed for North
  Micro Vision), SGLang's Transformers backend rejects the `embedding_rowwise` TP style
  that transformers adds for tied embeddings, so North still fails to load. The entry is
  inert there (only `nn.Linear` reads the plan); `--patch-sglang` maps it to `replicate`.
- transformers' "does not recognize this architecture" is rewritten by `hf:` and `vllm:` to
  name the model type, the installed transformers and the version it needs:
  "'…North-Micro-Vision-Instruct' is a 'cohere_compass' model, and the installed
  transformers 4.57.6 does not know that architecture: it needs transformers>=5.15. The
  checkpoint is fine." The original read like a broken checkpoint. The real cause was a
  stale auditkit (1.0.0, `transformers<5`) installed from another repository.
  `colab_cohere_models.ipynb` now installs from this repository and stops unless the
  installed auditkit is 1.2+; the `docs/SGLANG.md` pin table shows the current extras.
- Runs on any GPU, not just recent ones:
  - `compat.sglang_launch_advice()` and `python compat.py --sglang-launch MODEL` print the
    env and flags an SGLang server needs on the machine it runs on, with the machine's own
    CUDA toolkit: `SGLANG_ENABLE_JIT_DEEPGEMM=0` on Hopper/Blackwell (DeepGEMM serves FP8
    only, and its startup JIT needs nvcc >= 12.9), and `--dtype float16` without bf16
    (T4, V100), and `--attention-backend triton --sampling-backend pytorch` on SM 12.x GPUs
    with an nvcc older than 12.9. FlashInfer can't compile for them there; on Colab's RTX PRO
    6000 it failed with "FlashInfer requires GPUs with sm75 or higher". When FlashInfer's
    kernels don't build with the machine's toolkit for any other reason, the
    fallback is `--attention-backend triton --sampling-backend pytorch` (no nvcc). The
    SGLang matrix notebook switches to it automatically and labels the results.
  - `check_compat(check_cuda=True)` rule (k) warns about the DeepGEMM case. On Colab's RTX
    PRO 6000 every SGLang model failed to load with "NVCC version must be at least 12.9".
  - `hf:` loads a bf16 checkpoint as fp16 on a CUDA GPU without native bf16; an explicit
    `dtype=` still wins.
- `vllm:`: an architecture vLLM refuses (Aya Vision, "supported until v0.24.0") now
  says to try `model_impl="transformers"`, vLLM's Transformers backend.
- `docs/SGLANG.md` and `docs/model_backends.md`: the SGLang column is measured for 7 of
  the 9 Cohere models (the two 32B models need 80 GB). Aya Vision works on SGLang through
  its Transformers backend, patched. Tiny Aya scores the same on SGLang as on `hf:` on the
  same GPU. New Colab notebooks
  `colab_sglang_cohere_matrix.ipynb` and `colab_vllm_cohere_matrix.ipynb` measure the
  SGLang and vLLM columns.
- `check_compat()`: warns for sglang on Python 3.13, and for accelerate < 1.0 with
  transformers 5; with `check_cuda=True`, errors on a torchaudio that won't load
  against torch's CUDA.

### Fixed

- The six built-in benchmark scenarios load again. They used bare Hub ids (`mmlu`, `arc` and
  `truthfulqa` no longer exist; the others failed with `HfUriError`) and now use `cais/mmlu`,
  `openai/gsm8k`, `allenai/ai2_arc`, `Rowan/hellaswag`, `truthfulqa/truthful_qa` and
  `openai/openai_humaneval`. HumanEval also referenced a `TaskKind.CODE` that doesn't exist; it loads
  as a generative task (no code-execution metric, so use `ak.run_lmeval("humaneval", ...)` for pass@k).
- Docs: no METEOR (never implemented) and no "Rich HTML reports" tick (not implemented); current
  Claude model ids instead of retired Claude 3 ones; "source-available", not "open-source"; no
  internal wording, links to a private repository or private issue numbers in public pages;
  GUIDE's table of contents, catalog heading and version line match its sections and 1.1.2.

### Changed

- **`f1_score` now normalises SQuAD-style** (lowercase, punctuation and articles
  removed) before comparing tokens, so "Red" matches "red" and "Triangle." matches
  "triangle". **Scores change.** `F1Score(normalize=False)` keeps the old raw
  whitespace split; the setting is part of the run fingerprint.
- `[lmeval]` extra: `accelerate>=1.0`.
- **`abstention`: `short_reply_markers` now takes regular expressions**. The
  defaults require refusal-shaped wording ("insufficient information to/for/in/about…",
  "no relevant information/documents/context…"), so a short real answer such as
  "Yes, there is no relevant fee for domestic wires." is no longer a refusal, and a
  first sentence opening with "yes" never is. Bare phrases passed to
  `Abstention(short_reply_markers=...)` still work (a plain phrase is a valid regex);
  a phrase containing regex metacharacters must be escaped (`re.escape`).
- **An unnamed recorded tool call is unscored, not a measured 0.0**. An EventLog
  or AgentTune trajectory action that carries call arguments (`arguments`,
  `parameters`, `args`, `input`, `query`) but no tool name leaves
  `tool_call_names` unavailable, so `tool_call_f1` / `trajectory_match` /
  `parallel_tool_calls` are skipped, and `rescore` reports `trace.tool_call_names`.
- **A trajectory whose actions are all empty is a measured zero calls**: it
  scores (0.0 against expected calls, 1.0 on a no-call case) instead of being dropped.
- Docs: SGLang needs Python 3.10-3.12 (a uv 3.12 venv on 3.13 hosts); example 13
  builds its SGLang venv that way. `docs/VLLM_KNOWN_ISSUES.md` covers the Colab
  torchaudio mismatch after `[vllm]`.

## 1.1.2 — 2026-10-02

Lexsi stack interop (CuratorKIT, AlignTune, SafeTune, CircuitKIT and AgentTune
outputs evaluate with no glue code), plus fixes for the Cohere models.

### Added

- **Dataset folders.** `ak.load_dataset(path, config=None, split=None)` reads a
  `.jsonl`/`.csv` file, a CuratorKIT export folder (README `configs:` layout,
  what `datasets.load_dataset(dir, config)` loads) or a Hub id. It picks
  test > validation > train when no split is given and reads the `sft_alpaca`,
  `sft_sharegpt`, `dpo`, `grpo` and `ppo` columns. `ak.evaluate(<folder>,
  model, dataset_config="sft_alpaca", dataset_split=...)` does the same.
- **Provenance (`lexsi_provenance.json`).** Read from `hf:<dir>` model folders
  and dataset folders into `RunResult.metadata["inputs"]`, listed by
  `summary()` and `Report`, written next to saved results (`RunResult.save`)
  with the inputs embedded, and uploaded by `push_to_hub`. The Lexsi evidence
  importers take `provenance=` (and read CuratorKIT's `manifest["provenance"]`).
  New module `auditkit.provenance`; `ak.read_provenance`.
- **Native tool calling on `hf:`.** Tool schemas go to the chat template
  (`apply_chat_template(tools=...)`), so `ToolCallAdapter()` works on local
  models whose template renders tools; a template without tool support raises
  `CapabilityError`. None of the Cohere models' templates (Tiny Aya,
  Aya Expanse, Aya Vision, North) render tools: use
  `ToolCallAdapter(mode="prompt")` with them.
- **Cohere tool-call format.** The tool-call parser reads Command R7B
  `<|START_ACTION|>[{"tool_name", "parameters"}]<|END_ACTION|>`, the same text
  with the markers stripped, and Command-R / Aya Expanse
  `Action: ```json [...]```; recorded `{"tool_name"}` calls count as calls.
- `episodes_from_agenttune(<run folder>)` reads every `*.jsonl` in an AgentTune
  run folder and attaches its provenance to each episode.
- `Sample.images`: images for vision models (Aya Vision, North Micro Vision),
  sent through the model's processor by `hf:`.

### Changed

- `[transformers]` is `transformers>=5.15,<6` (North Micro Vision needs 5.15)
  and adds `peft`; `[vllm]` is `vllm>=0.30`; `[vision]` adds `torchvision`.
  `check_compat()` follows the new ranges.
- `__version__` comes from the installed package metadata.
- Docs are public at https://auditkit.lexsi.ai/.

### Fixed

- Aya prompts got two `<BOS_TOKEN>`s on `hf:` and `vllm:` (template plus
  tokenizer); now one.
- `hf:<PEFT adapter dir>` without `peft` failed as "generate failed after 3
  retries"; it now says peft is needed. Missing extras and capability errors
  are no longer retried and wrapped.
- Images sent to a text-only model raise instead of being dropped.

## 1.1.0

### New

- **Reference-free metrics (no gold standard required).** For RAG:
  `answer_relevancy` (answer addresses the question), `response_groundedness`
  (answer supported by the retrieved context), `hallucination` (claims
  unsupported or contradicted by context, lower is better), and
  `context_relevance` (retrieved contexts relevant to the question). For agents:
  `agent_loop_detection` (repeated calls, reasoning stagnation, call-graph
  cycles), `tool_permission` (least-privilege allow/deny check) and
  `tool_selection` (each call justified at that point). All need only the
  question, context, answer or trace, never a gold answer or labeled chunks.
- **`AssertionOracle`.** A first-class agent-eval outcome from a human-written
  natural-language criterion and a held-out judge, returning a real
  success/failure/unknown verdict with no gold trajectory. A verified state
  assertion still overrides it when both are present.
- **Metadata is metric-addressable.** A metric or the assertion oracle can read
  a user's own reference fields from `Sample.metadata` / `AgentCase.metadata`,
  so a custom column can gate a metric or feed the oracle.

- **Agent and tool-use evals, including parallel tool calls.** `Sample` has new
  fields: `tools`, `expected_tool_calls` (a list of turns; a turn with two or
  more calls is a parallel group), `reference_contexts` and `actual_trace`.
  New metrics:
  - `tool_call_f1`: precision, recall, F1 and exact match.
  - `trajectory_match`: strict and in-order.
  - `parallel_tool_calls`: `parallel_recall` catches independent calls that
    were serialized, `parallel_precision` catches dependent calls that were
    batched, and `parallel_detection` scores the should-I-parallelize decision.
  - `tool_call_validity`, `redundant_tool_calls` and the `task_completion`
    judge.

  Calls are matched with a maximum bipartite matching, not greedy pairing.
- **`agent:` backend.** Evaluate an externally deployed agent over HTTP (stdlib
  only). It reads the agent's answer, its tool-call transcript and its
  retrieved contexts.
- **Native tool calling on `api:`.** `tools`, `tool_choice` and
  `parallel_tool_calls` are forwarded, and `tool_calls` are captured. Use it
  with OpenAI-compatible servers, including vLLM and SGLang with a tool-call
  parser. `ToolCallAdapter(mode="prompt")` covers text-only backends.
- **RAG evals.**
  - `retrieval(k)`: hit rate, precision, recall, MRR, average precision and nDCG.
  - LLM-judged `faithfulness`, `context_precision` / `context_relevance` and
    `context_recall`. They work with any text judge, and a parse failure is
    recorded as an error, never scored 0.
- **Loaders.** `load_jsonl()` and `load_agenttune()` (AgentTune eval rows,
  trajectories and reports).
- **SGLang.** New `[sglang]` extra, mutually exclusive with `[vllm]` and
  `[transformers]`, plus `check_compat()` for environment and pin checks.
  See `docs/SGLANG.md`.
- Notebooks `13_sglang_compatibility` and `14_agent_and_rag_evals`.

### New: End-to-end agent evaluation (`agent_eval`)

- **`auditkit.agent_eval` package.** Evaluates what an agent did and achieved
  across a complete task episode. Stdlib only; AgentTune, torch and transformers
  are never imported. See
  [Agent evaluation](../agent_eval.md).
  - **Episode contract.** `AgentCase` (stable `digest()`, validated id/task),
    `AgentEvent` (order, role, type, `timestamp`-or-`None`, `call_id`,
    `turn_id`, raw payload, source, tier), and `AgentEpisode` (schema-versioned,
    mode `recorded`/`deployed`/`harness`, ordered events, final
    answer/state/artifacts, counters, a loss-aware `coverage` map, strict-JSON
    round-trip). `source_reward` / `source_verdict` / `source_scores` are kept
    as provenance only, never promoted to the outcome.
  - **AgentTune bridge.** `episodes_from_agenttune`,
    `episode_from_openai_messages` and `episode_from_sample` turn recorded runs
    into episodes with honest coverage: a flattened report is marked
    tool-name-only so argument-sensitive metrics stay ineligible rather than
    reading empty arguments as wrong ones. `inspect_agenttune` (and
    `supported_scorers_from_coverage`) report which metrics a trace can feed.
  - **Outcome oracles.** `FinalStateAssertion`, `ArtifactAssertion`,
    `AnswerAssertion` and `CustomPredicate`, with verdicts `success`,
    `failure`, `unknown` and `error`. A state or artifact oracle is
    authoritative; missing evidence is `unknown`, never a fabricated pass or
    fail. The `TaskCompletion` judge is a labeled diagnostic that never
    overrides a verified state assertion (a parse failure is `unknown`).
  - **Runner and report.** `AgentEvalRunner` / `AgentEvalSpec` run recorded
    episodes (no agent or tool call) or a deployed `agent:` endpoint (captures
    output, trace, contexts, errors, latency and usage). `AgentEvalResult`
    reports the verified outcome as the headline plus separate diagnostic
    columns, a per-case status (`completed`, `budget_exhausted`, `method_error`,
    `target_error`, `judge_error`, `ineligible`, `not_applicable`), and a
    summary with both a decided-only and an all-cases success denominator.
    `rescore()` replays saved episodes offline. `trials > 1` and
    `mode="harness"` raise `NotImplementedError` (deferred to A3/A4).
  - **CLI.** `auditkit agent eval` (`--dry-run` / `--output`),
    `auditkit agent import-agenttune` (`--inspect`) and `auditkit agent rescore`.
    Inspect and dry-run load no model and call no endpoint.
  - **Deployed state.** When the `agent:` reply carries a `final_state` or
    `artifacts` object (paths set by `agent_opts` `state_path` /
    `artifacts_path`), state and artifact oracles use it. Without it, a state
    oracle resolves `unknown`.
  - **AgentTune `EventLog`.** `episode_from_eventlog` reads an `EventLog`
    object or its JSON (duck-typed): turn boundaries become parallel groups,
    the episode reward becomes `source_reward`, and token spans and logprobs
    stay in the payload. The heal audit JSONL from `Project.heal` is refused
    with a clear error instead of being read as empty trajectories.
  - **Report provenance.** A `run_eval` report's `model`, `use_case`,
    `pass_rate`, counts and timestamp are kept on every episode
    (`metadata["source_report"]`); `source_verdict` is set when a row carries
    `passed`.
  - **Opt-in export redaction.** `agent eval` / `agent import-agenttune`
    accept `--redact-key KEY` and `--redact-env VAR`, and `redact()` does
    the same in Python. Nothing is redacted by default.
- **Lexsi evidence importer (`auditkit.agent_eval.lexsi`).** Read-only readers
  for artifacts that AgentTune, AlignTune, SafeTune, CuratorKIT and CircuitKIT
  already wrote. Each reader wraps the artifact in a `SourceEvidence` envelope
  with provenance, granularity, and the join keys present and missing.
  `join_evidence` attaches evidence to an episode only on ids the caller names:
  at least one identity key is required, aggregate reports are never pinned to
  one case or call, and unknown keys are refused. `lineage_sidecar` summarizes
  sample lineage. No producer library is imported. The gate tests load 27 real
  files that each library's own serializer wrote on placeholder input
  (`tests/fixtures/lexsi_real/`).

### Changed

- The inline-dataset fingerprint now also covers `metadata` and the new
  fields, but only when they are set. A dataset that carries `metadata` gets a
  new fingerprint once (one cache miss).

### Fixes: agent/RAG edge-case hardening

- **Linear tool-call and thinking parsing.** `parse_tool_calls` and
  `strip_thinking` ran in cubic time on repeated unclosed `<tool_call>` /
  `<think>` tags (a nested `(\s*)` pattern a runaway model could blow up);
  rewritten to run in linear time.
- **Bounded parallel matching.** `ParallelToolCalls`' parallel-recall search now
  runs under a fixed work budget (20,000 units) and returns a valid lower bound
  when the budget is hit, flagged by `recall_exact`, so a pathological trace
  cannot make it blow up. Trajectory in-order matching is single-pass, and
  JSON-equality and bipartite matching are iterative.
- **Per-request isolation in `api_gen`.** The request body (including the tool
  keys `tools` / `tool_choice` / `parallel_tool_calls`) is built per request
  from `Request.params`, and the prompt is string-coerced, so one request's
  fields never leak into another's.
- **Wall-clock-bounded runner timeout.** `Runner` now bounds each chunk's wait
  with `future.result(timeout)` instead of a `with` block that waited for
  `generate()` to return (so the configured `timeout` never actually cut a slow
  call); the wait is now bounded by wall-clock time, not by the call itself.
- **Robust loaders.** `load_jsonl` / `load_agenttune` name the offending line on
  bad JSON, reject non-object rows and non-dict `metadata`, treat a scalar id
  field as one id (a lone string is not char-split), keep a bare numeric or
  keyword answer as its raw string (never reformatted so it can no longer match
  a retrieved id), read `null` gold as no gold, preserve a `trajectory_id` as
  the sample id, and surface a structural mismatch as a named bad-line error.
- **Real AgentTune interop tests.** `tests/test_agenttune_integration.py` runs
  AgentTune's actual output shapes (trajectory JSONL, run-eval rows, RAG-GRPO
  rows, `trace.jsonl`, `Report.save` JSON, `TrajectoryStore` export) end to end
  through `load_agenttune` and the matching agent/RAG/answer metrics, each
  fixture cited by source file and line against the pinned mirror.

### Fixes: agent_eval, backends and RAG metrics (stress and audit pass)

- **Recorded runs match cases by id only.** A case is matched to its episode by
  `case_id`, then `source_id`, never by position. A case with no episode is
  `not_applicable`, and duplicate ids warn.
- **Deployed calls are bounded and isolated.** Every `agent:` call runs under a
  wall-clock deadline. An error from one call fails only that case as
  `target_error`, and the run fingerprint records the resolved endpoint
  identity, never raw `agent_opts`.
- **`agent:` / `api:` never raise per request.** HTTP errors, broken replies,
  deeply nested JSON and bad redirects fail only that request, so a batch is
  never replayed and side-effecting calls are not re-sent. 429 and 5xx
  responses and connection resets are retried per request. Non-list
  `tool_calls` are rejected rather than read as no calls.
- **Runner scoring.** A sample whose metric crashed counts in `failed_count`
  and is marked incorrect. `timeout` bounds one request on per-request
  backends. String or non-finite token usage is coerced.
- **Importers.** An observed transcript with zero tool calls counts as
  evidence that no tool was called, so a skipped tool is scored. JSONL is split
  on `\n` only, and AgentTune store exports group calls by `step_number` and
  never claim a per-call result pairing from a copied step observation.
- **RAG judges.** The judge cache key includes judge arguments and a stable
  identity for plain callables. Verdict parsing is linear-time and rejects
  conflicting duplicates. `max_context_chars` truncates each chunk to an equal
  share, so every chunk stays visible. Non-finite relevance grades raise.

## 1.0.0

### API changes

- **`CosineSimilarity` no longer depends on the `sentence-transformers`
  package — the `[sentence-transformers]` extra is removed entirely.**
  `sentence-transformers` was used for exactly this one metric in the
  whole library, and it transitively imports `transformers`' audio/video
  processing modules and, in turn, `torchcodec` -- a live-confirmed
  environment failure (`torch`/`torchcodec` version mismatch, missing
  `libavutil`/FFmpeg shared libraries) on some platforms (confirmed:
  Colab), with nothing to do with text embeddings at all. Reimplemented
  directly on plain `transformers.AutoModel`/`AutoTokenizer` (mean-pooling
  + L2-normalization -- the same recipe `sentence-transformers` uses
  internally for MiniLM-family checkpoints, not an approximation),
  already a core dependency for `EncoderJudge`/`HFGenModel` and confirmed
  not to trigger the `torchcodec` import path. `model_name=` default is
  now the full repo path (`sentence-transformers/all-MiniLM-L6-v2`, since
  plain `transformers` doesn't know `sentence-transformers`' short-name
  aliases). Needs `[transformers]` now, not a dedicated extra. Also fixed
  a pre-existing doc bug found in the process: `BM25Similarity` was
  listed everywhere as needing `[sentence-transformers]` despite its code
  never having any embedding-model dependency at all (pure token-frequency
  overlap, stdlib only) — corrected to "no extra needed".
- **New `guard_judge` (`GuardJudge`) — guard-model safety scoring.** Wraps a
  purpose-built safety classifier (default `hf:meta-llama/Llama-Guard-3-8B`;
  any `AutoModel` spec works, including hosted `groq:llama-guard-3-8b`) as an
  ordinary scorer: it classifies a model's output (or the prompt, via
  `assess="prompt"`) as safe/unsafe. `MINIMIZE`, 0 = safe; harm categories land
  in per-sample `metadata.categories`, so per-category rates and safety
  regressions flow into `RunResult`/`compare_models`. Guards differ only in
  input format + output parser, held in a per-family `profile` (`GUARD_PROFILES`;
  v1 ships `llama_guard`, custom profiles accepted). Offline safety scoring, not
  a runtime guardrail. Ships five profiles: `llama_guard` (chat-template
  taxonomy guard); `wildguard`/`harmbench` (raw-prompt classifiers, via a new
  `apply_chat_template=False` request flag the `hf:` backend honors, so a
  classifier's fixed prompt isn't corrupted by the tokenizer's chat template);
  and `shield_gemma`/`granite_guardian` (policy-parameterized — pass `policy=`
  to pick a harm policy/risk, threaded into the guard's chat template via a new
  `chat_template_kwargs` flag). Metric catalog is now 48 (the encoder judges
  below were added the same cycle).
- **Bias metrics reworked.** The old `bias_score` (`BiasScore`) was renamed to
  `representation_skew` (`RepresentationSkew`) *and* rewritten: it now measures
  demographic-representation balance per axis via total-variation distance from
  balanced (`MINIMIZE`, 0 = balanced), instead of Shannon entropy over
  individual tokens. The old version counted tokens and discarded group
  structure, so all-male "he him his" scored the same 0.9 as a balanced "he and
  she"; the rewrite scores those 1.0 vs 0.0. A new **`bias_judge`**
  (`BiasJudge`, LLM-as-judge) reads *meaning* — it flags the fraction of the
  output's own opinions that are biased, catching content ("women are too
  emotional to lead") that any word-count metric misses. `hate_speech_score`'s
  bias term was repolarized accordingly. For the deepest form (does the model
  *treat groups differently*), run BBQ/CrowS-Pairs via `run_lmeval`.
- **`compare_models(adapter=, annotators=, extract_with=)`** — previously
  `compare_models()` had no way to configure the adapter/annotators at all
  (every model silently got `evaluate()`'s bare defaults — `GenerationAdapter`,
  no annotators — regardless of what was passed). Now these three are real
  parameters, shared across every model by default, and `model_opts[name]`
  can override any of them for one model only — the rest of that dict still
  flows through to model construction unchanged. `scorers` remains a single
  shared argument, deliberately not overridable per-model.
- **`auditkit list annotators`** — the CLI's `list` subcommand previously
  covered `metrics`/`datasets`/`adapters`/`models` but had no way to
  enumerate registered `Annotator`s (`regex`, `llm`, `thinking_strip`);
  it's now a resource type alongside the others, and included in the
  default `auditkit list` (no argument) output.
- **`"echo"` removed from the public API surface.** `model=`/`--model` on
  `evaluate()`, `generate()`, `evaluate_many()`, the CLI (`eval`/`redteam`),
  and `RedTeamRunner` no longer default to it — `model=`/`--model` is now a
  required argument everywhere. `EchoModel` still exists internally (used
  by the test suite as a zero-dependency test double) but is no longer
  documented, exported behavior, or resolvable as advertised public API;
  use a plain `list[str] -> list[str]` callable or `"precomputed"` instead.
- **Removed `ASR`, `ClaimPrecision`, and `EntailmentRatio` metrics.** All
  three were thin token-overlap heuristics whose names oversold what they
  measured (see the removed entries in `docs/known_issues.md`'s prior
  "Proxy metrics" section) — not real attack-success judgment, claim
  verification, or entailment. `factual_consistency` (real NLI via
  `transformers`) remains as the one real hallucination-detection metric;
  `keyword_detector`/`DefconGrade` remain for security. Metric catalog is
  39 (was 42).
- **Latency reporting collapsed to one figure (mean), not p50+p95.**
  `RunResult.summary()`, `CompareResult.performance_table()`/`summary()`,
  and `RunComparison.summary()` previously showed `p50`/`p95` side by
  side — almost always identical in practice, since `Runner` times whole
  batched `generate()` calls, not individual requests, so `concurrency=1`
  (the default) produces exactly one latency sample per run and every
  percentile of a single value is trivially that same value.
  `performance_table()`'s returned dicts now have one `latency_mean_ms`
  key instead of `latency_p50_ms`/`latency_p95_ms`/`latency_mean_ms`. The
  full percentile breakdown (`p50`/`p95`/`p99`/`min`/`max`/`std`) is still
  computed and available via `RunResult.perf["latency_ms"]` for anyone
  who wants it — only the default printed/tabular views were simplified.
- **`ak.benchmark()` renamed to `ak.run_lmeval()`, and the `[benchmark]`
  pip extra renamed to `[lmeval]` to match** — makes explicit that this is
  the dedicated, always-lm-eval front door (no native/lmeval switch,
  unlike `evaluate(..., engine="lmeval")`, which reaches the same
  underlying engine) pulling in `lm-eval`/`accelerate` specifically —
  `[benchmark]` was ambiguous with the built-in benchmark *scenarios*
  (MMLU/GSM8K/etc.), which actually only need `[interop]`. Install via
  `pip install auditkit[lmeval]` now; the error message you'd get from
  calling `run_lmeval()` without it installed says so too.
- **New `ak.evaluate_many()`** — runs one model across several datasets in
  a single call (each as its own `evaluate()` call, its own fingerprint/
  cache entry, sequentially), returning one `RunResult` per dataset.
- **Real token throughput** (`output_tokens_per_sec`/`total_tokens_per_sec`)
  now computed automatically in `RunResult.perf`, `RunComparison`, and
  `CompareResult.performance_table()` from each run's own measured latency
  and provider-reported `token_usage` — no separate opt-in needed.
- **`HFGenModel`** no longer emits the `transformers` "Both `max_new_tokens`
  and `max_length` seem to have been set" warning on every call — clears
  the stale default on both the model's and the pipeline's own copy of
  `generation_config` right after loading.
- **Examples restructured**: the old numbered `.py`/`.sh` scripts in
  `examples/` were replaced with Colab-ready `.ipynb` notebooks (real
  models, real datasets). A new `applications/` folder holds end-to-end
  real-world scenarios (pruned-model ship/no-ship decisions, healthcare,
  finance, e-commerce, education, enterprise search, BERT-NLI-as-judge)
  built the same way. Of these, only `applications/01_application_pruned_llama_boolq.ipynb`
  uses the `lm-evaluation-harness` integration (`ak.run_lmeval()`) — as an
  authoritative cross-check next to the native evaluation, needing
  `auditkit[lmeval]`; no other notebook in `examples/` or `applications/`
  depends on it.

### New: Annotators

- **`RegexAnnotator`/`LLMAnnotator`** (`src/auditkit/annotator.py`) — pull a
  clean value out of a model's raw output (e.g. `"FINAL ANSWER: 42"` → `"42"`)
  as a separate, opt-in step from scoring. `RegexAnnotator` matches a
  user-supplied pattern; `LLMAnnotator` asks a second model to extract the
  value via a prompt, for cases with no fixed marker to anchor a regex to.
  Multiple annotators can run in the same call; `extract_with="<name>"`
  selects which one's extraction actually feeds the metrics — the rest still
  run and stay inspectable via `Prediction.context`. See
  [Annotators](../annotators.md).
- Both support `cast=`/`strict=` (convert the extracted string, e.g. to
  `int`/`float`; degrade gracefully or raise on a bad conversion) and are
  fully covered by run fingerprinting — changing a pattern/prompt/cast
  invalidates any cached result.

### New: OpenRouter model backend

- **`openrouter:` / `OpenRouterModel`** (`src/auditkit/model/openrouter_gen.py`)
  — a new T1 model backend for [OpenRouter](https://openrouter.ai), an
  OpenAI-API-compatible proxy in front of many providers. Mirrors
  `GroqModel`'s implementation and lm-eval wiring exactly (fixed base URL
  `https://openrouter.ai/api/v1`, Bearer auth, chat-only, no logprobs, maps
  to lm-eval's `openai-chat-completions` backend with `OPENROUTER_API_KEY`
  mirrored into `OPENAI_API_KEY` only for the run's duration). Models are
  selected via OpenRouter's own `"provider/model"` naming, e.g.
  `"openrouter:openai/gpt-4o-mini"` or
  `"openrouter:anthropic/claude-3.5-sonnet"` — the internal `/` doesn't
  conflict with `AutoModel.resolve()`'s prefix-splitting, which only splits
  on the first `:`. Adds two optional constructor params with no Groq
  equivalent — `site_url=`/`app_name=` — sent as `HTTP-Referer`/`X-Title`
  attribution headers when set. Rides the `requests` extra, same as
  `groq:`/`api:`.

### New: `EncoderJudge` — LLM-as-judge via a real encoder classifier

- **`EncoderJudge`** (`src/auditkit/metrics/encoder_judge.py`) — a judge
  backed by a real `AutoModelForSequenceClassification` encoder (BERT,
  RoBERTa, DeBERTa, ELECTRA, ...) instead of a prompted generative model.
  Classifies a (candidate, reference) pair in one forward pass via the
  tokenizer's real pair-sequence encoding — no free-text generation, no
  regex parsing. `label_map=` auto-detects from common NLI/sentiment-style
  label names, or raises (never guesses) for checkpoints with only generic
  `LABEL_0`/`LABEL_1` labels. `aggregation="probability"` (default, a
  confidence-weighted score) or `"argmax"`. Fully compatible with
  `Annotator`s (`extract_with=`) with no special-casing needed, since
  extraction happens before any metric is invoked. Verified live across 8
  real checkpoints spanning 5 architecture families (BERT, RoBERTa,
  DeBERTa v1/v3, ELECTRA, DistilBERT), including a real independent
  multi-label classifier. See [Encoder Judge](../encoder_judge.md).
- Found and fixed one bug during testing: BERT/RoBERTa/ELECTRA-style
  checkpoints (absolute position embeddings, hard 512-token limit) crashed
  outright on long combined input (`RuntimeError: tensor size mismatch`);
  DeBERTa (relative position embeddings, no hard limit) never hit this,
  which is why it wasn't caught until deliberately testing long inputs
  specifically. Fixed by always calling the classification pipeline with
  `truncation=True`.
- **2 prebuilt `EncoderJudge` subclasses** — `FactualityEncoderJudge`
  (entailment/factual-consistency) and `SentimentEncoderJudge` (2-way
  sentiment) — each with the correct, checkpoint-specific
  `model_name`/`label_map`/template shape baked in as defaults (same
  pattern as `Factuality`/`ClosedQA`/`Relevance` for `LLMJudge`), fully
  overridable.
  **Originally shipped as 6, reduced in two rounds.** First,
  `DebertaNLIJudge`/`DebertaV3NLIJudge`/`RobertaNLIJudge` were collapsed
  into one, `FactualityEncoderJudge` (keeping DeBERTa v1, the strongest of
  the three: no hard input-length limit, and empirically more
  confident/discriminative than DeBERTa-v3 on ambiguous pairs) — all three
  used checkpoints with real, auto-detectable labels, so having three
  separate classes added a name each but no real behavior difference over
  passing `model_name=` to one class directly. Then `BertNLIJudge`/
  `ElectraNLIJudge` were removed entirely: both did the exact same *task*
  as `FactualityEncoderJudge` (entailment/factual-consistency) — their
  only distinguishing feature was an unrecoverable, empirically-verified
  generic label order on their specific checkpoints (confirmed opposite
  conventions from each other), which is real knowledge but not a
  distinct enough task to justify two more classes. That knowledge is
  preserved as a worked example in `docs/encoder_judge.md`
  (`EncoderJudge(model_name=..., label_map=...)`) instead of two more
  classes with no real behavior difference beyond which checkpoint.
  `SentimentEncoderJudge` (renamed from `DistilBertSentimentJudge`) is the
  only prebuilt with a genuinely different task from `FactualityEncoderJudge`.

### Fixes

- **`Runner.score_one()`** no longer silently converts a missing optional
  dependency (`ExtraNotInstalled`, e.g. `cosine_similarity` without
  `[transformers]`) into a fake `0.0` score indistinguishable from a
  genuinely bad model output — it now surfaces via `result.errors`/
  `failed_count` instead.
- **`Perplexity`** no longer returns `nan` on short (single-token) outputs.
- **`WordErrorRate`** clamped to `[0, 1]` (could previously go negative).
- **`Faithfulness`/`ContextPrecision`** (RAG) fixed from plain substring
  matching (a short output word like `"an"` could match inside an unrelated
  word like `"banana"`) to word-boundary matching.
- **`ToxicityScore`** now runs a real classifier (`unitary/toxic-bert`) by
  default instead of a small keyword blacklist, which missed most real
  toxic phrasing and false-flagged safe text merely mentioning a listed
  word. `blacklist=`/`use_model=False` still available as the old
  dependency-free fallback.
- **`RunConfig.concurrency` default changed from `8` to `1`.** `Runner._chunk()`
  always splits requests into exactly `concurrency` chunks regardless of
  request count — with fewer samples than the configured concurrency, this
  produced empty chunks and wasted real model/API calls on them. `1` takes
  a no-chunking code path entirely, sidestepping the bug unconditionally.
  If you need real parallelism, pass a higher `concurrency=` explicitly —
  it's safe as long as your sample count is at least that value.
- **`identity()`/fingerprint correctness**: 14 built-in metrics had real
  constructor config that never affected `RunSpec.fingerprint()` (e.g.
  `Bleu(max_n=...)`, `EloScore(k=...)`, `ToxicityScore(
  use_model=...)`) — two differently-configured instances could silently
  share one cached `RunResult`. Fixed for all 14; `Annotator`/`Adapter`/
  `Metric` now also warn at class-definition time if a *custom* subclass
  looks like it has the same gap.
- **`load_croissant()` always crashed** — it called `mlcroissant.load(...)`,
  a function that doesn't exist in the real `mlcroissant` package (100%
  failure rate for any input, never caught since its only test checked the
  "extra not installed" path and lacked the `skipif` guard that would have
  made it actually run). Fixed to use the real
  `mlcroissant.Dataset(jsonld=...).records(record_set)` API — now takes a
  required `record_set` parameter (a Croissant file can describe several;
  there's no universally correct default) and decodes field values (the
  real API returns `bytes`). Verified against a real 918-row dataset,
  matching `load_hf()` on the same data exactly. `GitPython` added to the
  `[interop]` extra — `mlcroissant` needs it for some datasets' git-based
  downloads but doesn't declare it itself.
- **`bert_score` fixed** — previously documented as blocked by a genuine,
  unfixable `bert_score`/`transformers>=5` incompatibility
  (`OverflowError: int too big to convert`). Root cause: the default
  tokenizer (`microsoft/deberta-xlarge-mnli`) reports `model_max_length` as
  HuggingFace's "no limit configured" sentinel (`~1e30`), which
  `transformers>=5`'s Rust-backed truncation setup can't convert to a
  bounded int. `BertScore.score()` now applies a scoped monkeypatch of
  `bert_score`'s own `get_tokenizer` (reached via
  `sys.modules["bert_score.score"]`, since `bert_score/__init__.py`'s `from
  .score import score` shadows the plain attribute path) that clamps
  `model_max_length` to `512` before use, restoring the original function
  afterward. Verified live through the real `evaluate()` pipeline for both
  correctness (identical text → F1 ≈ 0.99999988) and discriminating power
  (unrelated text → F1 ≈ 0.43, not a degenerate always-high result). See
  [Known issues](../known_issues.md).
- **`HFGenModel` now honors `stop_sequences`/`seed`** — `stop_sequences` maps
  to `generate()`'s real `stop_strings` kwarg (with the tokenizer
  auto-attached, which that kwarg needs); `seed` calls transformers'
  `set_seed()` (a global RNG reset — the only per-call reproducibility knob
  `generate()`/`pipeline()` actually offers) right before generating.
  Verified live: a stop sequence truncates output exactly at the match, and
  the same seed + prompt + settings reproduces identical output across
  separate `evaluate()` calls. See [Known issues](../known_issues.md).
- **`VLLMModel.model_info()` implemented** — previously always reported
  identity-only despite `is_local=True`. Now attempts real parameter
  count/size introspection (same approach as `HFGenModel.model_info()`) via
  a few known vLLM-internal engine access paths, falling back to the
  previous identity-only reporting if none resolve on the installed vLLM
  version (its internal structure varies across V0/V1 engines and
  releases). See `docs/VLLM_KNOWN_ISSUES.md`.
- **`vllm:` backend verified live for the first time** (Colab L4 GPU) —
  real generation confirmed working end to end, closing a gap that had
  existed since the backend was first written (mocked unit tests only).
  Surfaced three environment/dependency crashes along the way, now
  fixed permanently inside `VLLMModel` itself (`_apply_environment_defaults()`/
  `_stdout_fix()` in `src/auditkit/model/vllm_gen.py`), so no user needs to
  work around them on any platform: vLLM's V1 engine forking a worker
  process that can't inherit an already-initialized CUDA context
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), Jupyter/Colab's `sys.stdout` having
  no real file descriptor for vLLM's internal log-suppression code
  (`sys.stdout`/`sys.stderr` swap around engine construction), and
  `huggingface_hub`'s newer Xet Storage download path 404ing on some
  legacy repos (`HF_HUB_DISABLE_XET=1`, best-effort). New
  `docs/VLLM_KNOWN_ISSUES.md` consolidates these, the still-open real gaps
  found in the process (no chat-template support, GPU memory never freed
  by `evaluate()` when resolving a model from a string spec), and the one
  unfixable-from-code issue (a `torch`/`vllm` CUDA version
  mismatch on environments with a pre-existing `torch`, e.g. Colab's
  default image). 5 new regression tests in `tests/test_generation_kwargs.py`.
- **`ExtraNotInstalled`'s message was self-contradictory** — every one of its
  ~16 call sites already passes a complete, ready-to-run instruction (e.g.
  `"pip install auditkit[vllm]"`), but the class always prefixed a second,
  separately-generated `"install auditkit[{extra}]"` in front of it,
  producing `"install auditkit[vllm] for pip install auditkit[vllm]"`. Now
  uses the passed hint verbatim, falling back to the generated form only
  when no hint is given at all. New `tests/test_errors.py`.
- **`ChatAdapter`'s stale source comment fixed** — no runtime behavior
  change; the comment claimed no backend reads `request.params["messages"]`,
  which stopped being true once `resolve_messages()` was wired into 5
  backends.
- **New `tests/test_metrics_via_pipeline.py`**: every one of the 39
  registered metrics now has at least one test through the real
  `ak.evaluate()` pipeline (`Runner`/`Adapter`/fingerprinting/`context`
  wiring), not just a direct `Metric.score()` call — previously only
  `exact_match` (plus one `Bleu()` instance) had that coverage, leaving 37
  metrics unverified beyond their own isolated math. A representative
  metric from each family (generation quality, embedding, toxicity, RAG,
  judge, deterministic) is also run through `compare_models()`/`ak.compare()`
  specifically, confirming the comparison layer handles every metric
  *kind* correctly, not just `exact_match`.

## v0.3.0 — 2026-07-01

### Highlights

- **Model comparison.** `compare_models()` runs the same dataset against multiple
  models, producing side-by-side results with bootstrap significance tests.
- **CLI subcommands.** `auditkit init` scaffolds projects, `auditkit list` shows
  available resources, `auditkit compare` compares models.
- **YAML config support.** Define evaluations in YAML files with `prompts:`,
  model config, tags, and split strategies. Load via `--config`.
- **Documentation rewrite.** Full MKDocs site with Material theme, cookbook case
  studies, best practices guides, and Lexsi-branded design system.

### New metrics

- RAG: `Faithfulness`, `ContextRecall`, `ContextPrecision`, `ResponseRelevancy`
- Generation: `Bleu`, `RogueL`, `ChrF`, `BertScore`, `Perplexity`, `WordErrorRate`
- Embedding: `CosineSimilarity`, `TokenOverlap`, `BM25Similarity`
- Hallucination: `FactualConsistency`, `ClaimPrecision`, `EntailmentRatio`
- Toxicity: `ToxicityScore`, `BiasScore`, `HateSpeechScore`
- Pairwise: `WinRate`, `EloScore`, `PreferenceAccuracy`
- Security: `ASR`, `DefconGrade`, `KeywordDetector`, `ThreatCategory`
- Performance: `LatencyStats`, `Throughput`

Conversation (`Coherence`/`TurnTaking`/`ContextAdherence`/`RepetitionPenalty`),
agent (`ToolCorrectness`/`TrajectoryMatch`/`StepEfficiency`/`GoalAccuracy`),
multimodal (`VQAAccuracy`/`ANLS`), and tabular
(`SchemaConformance`/`ExecutionAccuracy`/`DenotationAccuracy`) metrics were
added and then removed within this same version cycle (see
`logs/REMOVE_MULTIMODAL_AGENT_CONVERSATION_TABULAR_2026-07-17.md`) — none
of them exist in the current codebase.

### Fixes

- Bleu `max_n` clamped to `min(len(ref), len(hyp))` — exact match returns 1.0
- CLI `concurrency=None` no longer crashes Runner (defaulted to `8` at the
  time; see Unreleased above — the default later changed to `1`)
- DiskCache invalidation for stale `~/.cache/auditkit/runs/` entries

### Known issues

- CI workflow requires PAT with `workflow` scope

## v0.2.0 — 2026-06-15

- Initial public release
- Core evaluation spine (Sample, Metric, Model, Runner)
- Benchmark metrics (exact match, F1, quasi-exact match)
- CLI for basic evaluations
- Experiment tracking with ExperimentDB
