# Agentic + RAG metrics test suite

An independent test suite for the agent and RAG evaluation added on `feat/rag-agent-evals`. It was
written while reviewing PR #1: every issue raised in that review was found by one of these tests. It
covers:
- the agentic metrics: `tool_call_f1`, `trajectory_match`, `parallel_tool_calls`, `tool_call_validity`,
  `redundant_tool_calls`, `task_completion`;
- the RAG metrics: `retrieval(k)`, `faithfulness`, `context_precision`/`context_relevance`,
  `context_recall`, and the four lexical metrics;
- the run fingerprint and the `api:` / `agent:` backends.

It asserts the **current** behaviour of `feat/rag-agent-evals`, including the design choices made in the
review fixes (see `FINDINGS.md`).

| File | Needs | What it checks |
|---|---|---|
| `test_metrics_offline.py` | nothing | Each agentic metric against hand-computed scores. **Calls:** missing, extra, duplicate. **Argument equality:** int vs float, bool vs int, case, whitespace, key order, nested values, unicode. **Matching:** `arg_mode`s, custom matchers, bipartite vs greedy. **Structure:** turn order, parallel groups. **Also:** schema violations, repeats, judge verdicts. |
| `test_formats_offline.py` | nothing | How calls reach the metrics: Hermes blocks, `<think>`, unclosed tags, bare/concatenated/fenced JSON, OpenAI/BFCL/alias shapes, transcripts, precedence of trace over text, reference shapes. One behaviour in four wire formats scores identically. |
| `test_pipeline_offline.py` | local mock servers | `ak.evaluate()` end to end: `precomputed`, prompt mode, the real `api:` backend against a mock OpenAI server (per-sample failures, retries), the real `agent:` backend against a mock agent, headline aggregation, nested validation. |
| `test_agentic_pipeline_all_metrics.py` | nothing | One readable file: all six agentic metrics on nine behaviours of one task. |
| `test_rag_offline.py` | nothing | Retrieval ranking (graded nDCG, @k, duplicates); judge metrics with a scripted judge; the lexical metrics; one full RAG pipeline. |
| `test_fingerprint_offline.py` | nothing | The run fingerprint (cache key): fields that must change it, pairs that must differ or match, the version, stability across processes, and cache correctness end to end. |
| `test_resolved_issues.py` | local mock servers | Regressions for the review items: answer-only agent replies, schema-aware `exact`, the judge timeout, `OPENAI_API_KEY` scoping. |
| `test_live.py` | `AK_LIVE=1` + a model server | Every model × {native tools, prompt mode} on 16 single-turn cases, a real agent loop (model + stub tools) on 4 multi-turn cases, and `task_completion` with a real judge. Writes `live_reports/`. |
| `test_live_rag.py` | `AK_LIVE=1` + a model server | Each model as the RAG judge, on cases with a known right ordering, plus its parse-error rate. |
| `make_metric_reports.py` | — | Writes `metric_reports/<metric>.md`: expected vs actual for every case, the test status, notes, and live results. |
| `colab_offline_suite.ipynb` | Colab CPU | Runs the offline layers and regenerates the reports. |
| `colab_model_matrix.ipynb` | Colab GPU | Serves about 19 models from 10 families (0.5B–8B) with vLLM and each family's tool-call parser, runs both live files per model, and builds comparison tables. |

```bash
python -m pytest -o addopts= tests/agentic_suite -q                 # offline layers, seconds
AK_LIVE=1 python -m pytest -o addopts= tests/agentic_suite/test_live.py tests/agentic_suite/test_live_rag.py -q -rA
AK_LIVE=1 AK_LIVE_BASE=http://localhost:8000 AK_LIVE_MODELS=Qwen/Qwen2.5-7B-Instruct AK_LIVE_NATIVE=all \
  python -m pytest -o addopts= tests/agentic_suite/test_live.py -q    # vLLM / SGLang
python -m pytest -o addopts= tests/agentic_suite -q --junitxml=/tmp/j.xml && \
  PYTHONPATH=src python -m tests.agentic_suite.make_metric_reports /tmp/j.xml
```

**Live settings (environment variables):**
- `AK_LIVE_BASE`: the model server URL (Ollama by default).
- `AK_LIVE_MODELS`: which models to run.
- `AK_LIVE_NATIVE`: which of those models get native tool calls.
- `AK_LIVE_JUDGE`: the judge model.
- `AK_LIVE_TAG`: a label for the report files.
- `AK_LIVE_MIN_F1`: the agentic quality floor (default 0.5).
- `AK_LIVE_QUALITY`: whether the RAG ordering checks must pass (default 1).
- `AK_LIVE_JUDGE_TIMEOUT`: the judge timeout.

Each test isolates `XDG_CACHE_HOME`, so no result is ever served from `~/.cache/auditkit`.
