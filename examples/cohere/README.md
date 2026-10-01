# Cohere notebooks

These notebooks evaluate Cohere models (Tiny Aya, Aya Expanse, Aya Vision, North Micro Vision) with AuditKIT. Each is self-contained: on Colab, its first code cell installs AuditKIT from `feat/rag-agent-evals` (#1, with #9, #10 and the #15 / #27 fixes merged), and updates an existing clone so a reused runtime never runs stale code.

| Notebook | What it does | Models | Colab GPU |
|---|---|---|---|
| `compare_models.ipynb` | Two `compare_models` runs. **Run 1, text:** Tiny Aya (`cohere2`) vs Aya Expanse (`cohere`) on multilingual QA and translation. **Run 2, vision:** Aya Vision (`aya_vision`) vs North Micro Vision (`cohere_compass`) on generated images. Together they cover all 4 Cohere architectures. Reports per-metric scores, the winner, significance, performance, and answers side by side. | all 4 | L4 / A100 (T4 with `LOAD_IN_4BIT=True`) |
| `run_lmeval.ipynb` | A **transformers 5 + vLLM** compatibility check on Tiny Aya: `check_compat()`, the native `vllm:` backend, `run_benchmark` on `arc_easy` + `gsm8k` through vLLM, parity with `hf:`, and a PASS/FAIL checklist. | Tiny Aya | L4 / A100 |
| `metrics_generation.ipynb` | AuditKIT's generation metrics on Tiny Aya: short answers (EM, QEM, F1, Levenshtein), translation (BLEU, chrF, ROUGE-L, token overlap, BERTScore optional), format checks (JSON, regex, word count, prefix), a GSM8K slice with a custom last-number metric, and latency / throughput. | Tiny Aya | T4 is enough |
| `sglang_auditkit.ipynb` | Serves Tiny Aya with **SGLang** in its own venv (the version is picked from the NVIDIA driver). Then: `check_compat()` in both environments, a generation eval through `api:`, a prompt-mode tool-call eval through `api:`, and shutdown. | Tiny Aya | L4 / A100 (T4 with `DTYPE="half"`) |
| `tgi_endpoint.ipynb` | Evaluates a model served by Hugging Face **TGI** through `api:` (AuditKIT never imports TGI). Needs a running TGI server; ships without outputs. | any TGI-served model | the TGI server's |

**Colab secrets:**
- `GITHUB_TOKEN`, to clone the private repo;
- `HF_TOKEN`, from an account that has accepted the licences for Tiny Aya, Aya Expanse and Aya Vision (North Micro Vision is open).

**Quick smoke test with other models:** the settings cells read `NB_*` environment variables (`NB_MODEL`, `NB_RUN1_A`, …), so the same notebook can run with an open model.
