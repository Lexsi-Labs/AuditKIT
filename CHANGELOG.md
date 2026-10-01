# Changelog

All notable changes to AuditKIT. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The full history, with every earlier release, is in [docs/community/changelog.md](docs/community/changelog.md).

## [Unreleased]

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

- `auditkit eval --config FILE` accepts `model:` from the config file. `--model` was
  `required=True` on the parser, so argparse rejected the command with "the following
  arguments are required: --model" before the YAML was read, even though `docs/cli.md`
  documents exactly that invocation. It is now optional on the parser and checked after the
  config merge; a genuinely missing model still exits 2, naming both ways to supply it.
- The six built-in benchmark scenarios load again. They used bare Hub ids (`mmlu`, `arc` and
  `truthfulqa` no longer exist; the others failed with `HfUriError`) and now use `cais/mmlu`,
  `openai/gsm8k`, `allenai/ai2_arc`, `Rowan/hellaswag`, `truthfulqa/truthful_qa` and
  `openai/openai_humaneval`. HumanEval also referenced a `TaskKind.CODE` that doesn't exist; it loads
  as a generative task (no code-execution metric, so use `ak.run_lmeval("humaneval", ...)` for pass@k).
- Docs: the scorer reference and metrics overview cover all 68 registered metrics, and the
  data model documents `Sample.images`, `RunConfig.seed`/`track_performance`/
  `chat_template_kwargs` and `RunResult.unscored`/`metadata`; every `eval` flag is in
  `docs/cli.md`; the extras table matches `pyproject.toml`; the claimed-but-unimplemented
  features are gone, as are the internal pages that were still being published to the site.
- The `dev` extra declares `pyyaml`, which the `--config` tests need. PyYAML remains an
  optional runtime dependency; the core install is still dependency-free.
- Docs: no METEOR (never implemented) and no "Rich HTML reports" tick (not implemented); current
  Claude model ids instead of retired Claude 3 ones; "source-available", not "open-source"; no
  internal wording, links to a private repository or private issue numbers in public pages;
  GUIDE's table of contents, catalog heading and version line match its sections and 1.2.0.

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

## [1.2.0] - 2026-09-28

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

- A JSON-string reference (a CSV cell, a JSONL field) now fingerprints like its
  decoded form, in the dataset fingerprint and in `AgentCase.digest()`, so the
  same dataset read from a file reuses the cached run. Dict and list references keep
  their digests. `auditkit agent --dry-run` shows `refs=yes` for an explicit `[]`
  reference ("expect no call").
- A bare prefix is now decided per backend. `api:`, `hf:` and `vllm:` raise
  "empty model spec" (`api:` used to send `{"model": null}`). The hosted prefixes
  (`openai:`, `anthropic:`, `groq:`, `openrouter:`, `lexsi:`, `litellm:`) fall back
  to their backend's default model, and `agent:` still takes its URL from `url=`.
- **`vllm:` `unload()` left the engine's GPU memory allocated on vLLM 0.30.** The
  teardown dropped the runner's `model`/`kv_caches` before calling vLLM's own
  `shutdown()`, whose runner step starts with `kv_caches.clear()`. So it raised,
  silently, and `Worker.shutdown()` never reached `CuMemAllocator.release_pools()`.
  On a Colab L4, "16.89 GiB still in use" after unload made the next `vllm serve` fail
  its free-memory check. vLLM's own teardown now runs first; the sleep-and-drop
  fallback runs only when the runner still holds its model (vLLM 0.19). A failing
  step is logged, and `unload()` logs the memory it returned.
- **`vllm:` sent raw prompts to every Cohere model.** Cohere ships named chat
  templates (Tiny Aya: `default`; Command R7B: `default`/`tool_use`/`rag`), which
  transformers loads as a dict, and `vllm:` took only a str as "has a template". So
  the models continued the prompt instead of answering, and native tools were
  refused as "no chat template". Found by `examples/colab/06_cohere_all_backends.ipynb`.
- Aya prompts got two `<BOS_TOKEN>`s on `hf:` and `vllm:` (template plus
  tokenizer); now one.
- `hf:<PEFT adapter dir>` without `peft` failed as "generate failed after 3
  retries"; it now says peft is needed. Missing extras and capability errors
  are no longer retried and wrapped.
- Images sent to a text-only model raise instead of being dropped.

## [1.1.0]

### Added

- Agent and tool-use evals (`tools`, `expected_tool_calls` as turns,
  `tool_call_f1`, `trajectory_match`, `parallel_tool_calls`,
  `tool_call_validity`, `redundant_tool_calls`, `task_completion`), the `agent:`
  backend, native tool calling on `api:`, and `ToolCallAdapter`.
- RAG evals: `retrieval(k)`, judged `faithfulness`, `context_precision`,
  `context_recall`, and reference-free RAG and agent metrics.
- `auditkit.agent_eval`: episode contract, AgentTune importers
  (`episodes_from_agenttune`, `episode_from_eventlog`), outcome oracles,
  recorded/deployed/harness runs, reliability, and the read-only Lexsi evidence
  importers (`safetune_bench_row`, `curatorkit_manifest`,
  `circuitkit_faithfulness_report`, `aligntune_*`, ...).
- `load_jsonl()`, `load_agenttune()`, the `[sglang]` extra and `check_compat()`,
  RAG stress fixtures.

Details: [docs/community/changelog.md](docs/community/changelog.md#110).

## [1.0.0]

See [docs/community/changelog.md](docs/community/changelog.md#100).
