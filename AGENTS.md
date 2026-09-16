# Agent instructions — AuditKit (release v1.0.0)

AuditKit is a standalone Python library that evaluates any model on any dataset
across any technique. The core runs on the standard library with **zero required
third-party dependencies**; model backends and heavy metrics are optional extras.

## Architecture — the 5-stage spine

`Dataset → Adapter → Model → Metrics → RunResult`, orchestrated by `ak.evaluate()`.

- **Dataset** — prompts and expected outputs (a YAML file, a Python list, or a loader).
- **Adapter** — the elicitation technique: benchmark, LLM-as-judge (`GEval`), code
  checks, RAG, hallucination, embedding similarity, toxicity/bias, pairwise/preference,
  red-teaming, security, performance.
- **Model** — 9 backends resolved by `AutoModel.resolve()`: `openai:`, `anthropic:`,
  `hf:`, `lexsi:`, `vllm:`, `litellm:` (also reaches Ollama), `api:`, `groq:`,
  `openrouter:`. Any `list[str] → list[str]` callable also works; `echo` is the
  dependency-free stub for tests.
- **Metrics** — every metric subclasses `auditkit.metric.Metric` (10 families).
- **RunResult** — aggregate scores, per-sample rows, and `perf` (latency/throughput).

## Install

```bash
pip install auditkit                 # core, zero deps
pip install "auditkit[openai]"       # one backend
pip install "auditkit[all]"          # all backends + heavy metrics
```

## Quickstart

```python
import auditkit as ak

result = ak.evaluate(
    model="echo",                    # or "openai:gpt-4o-mini", "hf:Qwen/Qwen2.5-0.5B", ...
    prompts=["2+2=", "capital of France?"],
    metrics=[ak.Contains("4"), ak.Contains("Paris")],
)
print(result.summary())
```

## CLI

```bash
auditkit eval config.yaml            # run a YAML-defined evaluation
auditkit init                        # scaffold a config
auditkit list                        # list available metrics/backends
auditkit compare ...                 # compare models
auditkit redteam ...                 # red-team probes
```

## Test

```bash
python -m pytest tests/ -q --tb=short   # pyproject sets pythonpath=src; use python3 on Unix
```

~53 test modules under `tests/`. The CLI subprocess tests need an editable install
(`pip install -e .`). Some `test_extra_not_installed` checks fail on machines that
have optional extras (e.g. `transformers`) installed globally — that is expected.

## Conventions

- `from __future__ import annotations` at the top of every module.
- **Zero required third-party deps.** Never import an optional extra at module top
  level — import it lazily inside the function that needs it, and raise a clear error
  if the extra is missing.
- Metrics inherit `auditkit.metric.Metric`; models inherit `auditkit.model.Model`.
  Register new ones through `registry.py` / `router.py`.
- One test file per module or feature area, in `tests/`.
- **Version lives in two places, keep them in sync:** `src/auditkit/__init__.py`
  (`__version__`) and `pyproject.toml` (`version`).

## Module map (`src/auditkit/`)

| Module | Role |
|--------|------|
| `api.py` | Public `ak.evaluate()` surface |
| `runner.py`, `runspec.py` | Execution engine and run configuration |
| `adapter.py`, `scenario.py` | Elicitation techniques and scenarios |
| `model/`, `router.py`, `registry.py` | Model backends and resolution |
| `metric.py`, `metrics/`, `scorers.py`, `scoring.py` | Metrics and scoring |
| `comparison.py`, `model_compare.py`, `diff.py` | Model comparison |
| `redteam/` | Red-team probes |
| `report.py`, `report_format.py` | Result rendering |
| `cli.py`, `__main__.py` | Command-line interface |

## License

Released under the **Lexsi Labs Source Available License (LSAL) v1.2** — free for
academic research and teaching (MIT-like); organizations must acknowledge their use
or obtain permission (Section 1A); commercial use requires a separate license. See
[LICENSE.md](LICENSE.md).
