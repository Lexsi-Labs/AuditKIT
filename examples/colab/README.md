# Colab notebooks

Three self-contained notebooks that install AuditKIT from `feat/rag-agent-evals` (#1's branch, with #10 and the #15 fixes merged) and test it against live models.

| Notebook | What it tests | Runtime | Secrets |
|---|---|---|---|
| [`01_cohere_and_hf_api_models.ipynb`](01_cohere_and_hf_api_models.ipynb) | Hosted models from the **Cohere API** and **Hugging Face Inference Providers** through `api:`: multilingual QA, JSON output and native tool calling, compared side by side | CPU | `GITHUB_TOKEN`, `CO_API_KEY` and/or `HF_TOKEN` |
| [`02_agentic_rag_real_agent.ipynb`](02_agentic_rag_real_agent.ipynb) | A **real agentic RAG system** (a bank support agent with BM25 retrieval and account tools, served over HTTP), scored with the agentic, retrieval, LLM-judge and deterministic RAG stress metrics, plus the `agent_eval` harness verifying final account state | CPU for `cohere` / `hf`; GPU for `local` | `GITHUB_TOKEN`, plus `CO_API_KEY` or `HF_TOKEN` for the backend |
| [`04_cohere_hf_fixes_live.ipynb`](04_cohere_hf_fixes_live.ipynb) | The **Cohere and `hf:` integration and its fixes**, each shown working with a PASS/FAIL checklist: native tools and `chat_template_kwargs` (Qwen3), `f1_score` normalisation, Tiny Aya prompt-mode tools, Command R7B's template and action format, the Cohere API, the harness on a text model, AgentTune and `agent:` coverage, setup errors | T4 GPU (24 GB+ for Command R7B) | `GITHUB_TOKEN`; `HF_TOKEN` (gated Cohere models); `CO_API_KEY` |
| [`05_real_world_agentic_rag.ipynb`](05_real_world_agentic_rag.ipynb) | A **real-world agentic RAG application**: a LangGraph ReAct agent over the public `rag-mini-wikipedia` corpus with dense retrieval, served over HTTP, then evaluated (`agent:`), error-analysed, A/B tested (retrieval `top_k`, paired significance) and its logged transcripts rescored offline | CPU for `cohere` / `hf`; L4 for `vllm` | `GITHUB_TOKEN`, plus the backend's key |
| [`03_vllm_integration.ipynb`](03_vllm_integration.ipynb) | **vLLM** end to end: `check_compat`, the offline `vllm:` engine (generation, `chat_template_kwargs`, prompt-mode tools), lm-eval through vLLM, and a `vllm serve` server through `api:` (native and parallel tool calls, the harness) | L4 / A100 GPU | `GITHUB_TOKEN` |

Add the secrets in Colab's 🔑 panel and give the notebook access. Each notebook skips what its missing secrets need, with a note.
