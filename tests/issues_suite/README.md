# Issues suite: #43, #44, #45, and the `vllm:` Cohere template fix

These tests use real data, and they exercise the fixes the way users hit the bugs.

| File | Covers | Data |
|---|---|---|
| `test_issue43_specs.py` | Bare `api:` / `hf:` / `vllm:` fail before any request, with a spec to copy. A real local model folder still resolves. `api:my-model:v2` stays whole. Bare hosted prefixes use the service default. `agent:` takes `url=`. | offline |
| `test_issue44_user_flows.py` | A tool-call dataset exported to CSV (pretty-printed cells) and reloaded hits the cached run with no new model call. Two JSONL exports, one with the reference as a JSON string, share an identity. Edited references don't. `"[]"` is not a missing reference. A broken cell fails its own sample, not the hash. Agent config digests match. `auditkit agent --dry-run` reports `[]` as a reference. | real BFCL answers |
| `test_issue45_bfcl.py` | The count behind the `one_of` decision, and the loader on every supported category. It checks: a ground-truth replay scores 1.0; the id-drift pairing; the unsatisfiable answers; system prompts; the sql shape; JSON Schema conversion; Java typing; `dots_to_underscores`; the refusals; array re-saves. It also runs the matcher on real entries: optional arguments, case and space, `10.0 == 10`, bool is not int, unexpected and missing parameters, nested objects, list order, parallel order, and irrelevance. | BFCL v3 @ `61fc060` |
| `test_issue45_routes.py` | Real BFCL parallel answers as `any_of`, batched or sequential. The tests check the histogram, what decided the route, the listed-order ties, the partial-run credit, the multi-route-only denominator and CSV cells, and that nothing reaches the stats. | BFCL v3 |
| `test_vllm_cohere_templates.py` | `vllm:` with the real Tiny Aya and Command R7B tokenizers, whose named templates load as a dict. It covers the chat template and one BOS, the Colab failure, Tiny Aya's refusal reason, Command R7B native tools with its action list scored, and prompt mode. | cached tokenizers (gated) |
| `test_live.py` | **Opt-in.** A real Qwen3-1.7B behind four servers (`serve_model.py`: SGLang-like, vLLM-like, a gateway reporting `owned_by: openai`, and a bare server with no `/models`). It covers the probe-vs-declare notes, a gateway that drops `chat_template_kwargs` (Qwen3 thinks and runs out of tokens), the one-call cap unverified on SGLang with `tool_choice="auto"` and applied on declared vLLM, a BFCL slice through `api:`, and a CSV reload that doesn't re-bill the server. | BFCL v3 plus the model |

```bash
# BFCL: point at a downloaded copy, or let the Hub cache fetch the pinned revision
AK_BFCL_ROOT=/path/to/bfcl python -m pytest -o addopts= tests/issues_suite -q -rs
AK_BFCL_DOWNLOAD=1 python -m pytest -o addopts= tests/issues_suite -q -rs
AK_LIVE_HF=1 AK_BFCL_ROOT=/path/to/bfcl python -m pytest -o addopts= tests/issues_suite/test_live.py -s
```

Local results (M3 Pro, MPS, 2026-09-30):
- offline: **70 passed**;
- live: **8 passed**.

The BFCL counts, with their revision, are in `docs/notes/bfcl-possible-answer-counts.md`.
