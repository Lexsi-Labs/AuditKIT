# Cohere notebooks

These notebooks evaluate Cohere models (Tiny Aya, Aya Expanse, Aya Vision, North Micro Vision) with AuditKIT. Each is self-contained: on Colab, its first code cell installs AuditKIT from `feat/rag-agent-evals` (#1, with #9, #10 and the #15 / #27 fixes merged), and updates an existing clone so a reused runtime never runs stale code.

| Notebook | What it does | Models | Colab GPU |
|---|---|---|---|
| `compare_models.ipynb` | Two `compare_models` runs. **Run 1, text:** Tiny Aya (`cohere2`) vs Aya Expanse (`cohere`) on multilingual QA and translation. **Run 2, vision:** Aya Vision (`aya_vision`) vs North Micro Vision (`cohere_compass`) on generated images. Together they cover all 4 Cohere architectures. Reports per-metric scores, the winner, significance, performance, and answers side by side. | all 4 | L4 / A100 (T4 with `LOAD_IN_4BIT=True`) |
| `run_lmeval.ipynb` | A **transformers 5 + vLLM** compatibility check on Tiny Aya: `check_compat()`, the native `vllm:` backend, `run_benchmark` on `arc_easy` + `gsm8k` through vLLM, parity with `hf:`, and a PASS/FAIL checklist. | Tiny Aya | L4 / A100 |
| `metrics_generation.ipynb` | AuditKIT's generation metrics on Tiny Aya: short answers (EM, QEM, F1, Levenshtein), translation (BLEU, chrF, ROUGE-L, token overlap, BERTScore optional), format checks (JSON, regex, word count, prefix), a GSM8K slice with a custom last-number metric, and latency / throughput. | Tiny Aya | T4 is enough |
| `sglang_auditkit.ipynb` | Serves Tiny Aya with **SGLang** in its own venv (the version is picked from the NVIDIA driver). Then: `check_compat()` in both environments, a generation eval through `api:`, a prompt-mode tool-call eval through `api:`, and shutdown. | Tiny Aya | L4 / A100 (T4 with `DTYPE="half"`) |

**Colab secrets:**
- `GITHUB_TOKEN`, to clone the private repo;
- `HF_TOKEN`, from an account that has accepted the licences for Tiny Aya, Aya Expanse and Aya Vision (North Micro Vision is open).

**Quick smoke test with other models:** the settings cells read `NB_*` environment variables (`NB_MODEL`, `NB_RUN1_A`, …), so the same notebook can run with an open model.

## Validation status (local, M3 Pro / MPS, transformers 5.17)

| Notebook | Checked locally |
|---|---|
| `compare_models` | Ran end to end, 0 error cells. **Vision run with the real North Micro Vision (`cohere_compass`): 5/5 correct** (Red, 3, 42, C, Triangle). The text run used Qwen3-1.7B / 4B standing in for the gated Tiny Aya / Aya Expanse. |
| `metrics_generation` | Ran end to end, 0 error cells, with Qwen3-1.7B standing in for Tiny Aya. |
| `run_lmeval` | vLLM needs Linux + CUDA. Checked locally: the `run_benchmark` lm-eval path on transformers 5 (hf backend) and the `vllm:` model mapping. Every code cell parses. **Run it on Colab.** |
| `sglang_auditkit` | **Run end to end on a Colab A100** (driver 580), saved outputs included: sglang 0.5.20 in a Python 3.12 uv venv, the server healthy on the first launch, `check_compat()` OK in both environments, generation and prompt-mode tool evals through `api:`, clean shutdown. That run's `f1_score` (0.333) predates the G14 normalisation, because the runtime reused an older clone; the install cell now updates an existing clone. |

The gated Cohere models (Tiny Aya, Aya Expanse, Aya Vision) have not been run yet: this machine has no `HF_TOKEN`.
