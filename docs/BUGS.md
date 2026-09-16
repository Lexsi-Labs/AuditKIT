# Bugs — consolidated, currently live

One place for every real, currently-present bug in this repo, pulled together
from `docs/known_issues.md`'s table, caveats scattered through `docs/GUIDE.md`,
and issues found during later sessions that hadn't been consolidated anywhere.
Every entry below was **re-verified live** (not just copied from an older doc)
as of 2026-07-30 — see the reproduction snippet in each entry.

This does not replace [`docs/known_issues.md`](known_issues.md) (which also
covers documentation debt and "never verified" caveats in more narrative
form) — it's the flat, single-purpose bug list.

---

## Confirmed bugs

### 1. Two different, incompatible percentile formulas

**Where:** `src/auditkit/score.py` (`Stat.percentile()`) vs.
`src/auditkit/metrics/perf.py` (`LatencyStats.percentile()`, private, used by
`LatencyStats.stats()`).

`Stat.percentile()` uses linear interpolation between bracketing values.
`LatencyStats`'s internal percentile function uses nearest-rank via integer
index truncation. On the same data they disagree:

```python
from auditkit.score import Stat
from auditkit.metrics.perf import LatencyStats

data = [10, 20, 30, 40, 100]

st = Stat(name="x")
for v in data:
    st.add(v)
print(st.percentile(95))   # 87.99999999999999  (linear interpolation)

ls = LatencyStats()
for v in data:
    ls.record(v)
print(ls.stats()["p95"])   # 40    (index truncation)
```

**Why it rarely shows up:** `Runner` times whole batched `generate()` calls,
not individual requests. At the default `concurrency=1`, a run makes exactly
one call, so latency has `n=1` and every percentile of one value is trivially
that value (`mean == p50 == p95`) — the divergence only appears at
`concurrency>1`. `RunResult.summary()`/`compare_models()`'s tables were
changed to print only `mean` for this reason, but the underlying dual-formula
inconsistency in `LatencyStats.stats()` itself was **not** unified — anyone
reading `result.perf["latency_ms"]["p95"]` directly and comparing it to a
`Stat`-based percentile elsewhere will see different numbers.

### 2. `Throughput.rps` overcounts elapsed time under concurrency

**Where:** `src/auditkit/metrics/perf.py`, `Throughput.stats()`.

`_total_time_ms` is the **sum** of every call's own duration, not true
wall-clock elapsed time for the whole run:

```python
class Throughput:
    def record(self, total_ms: float, num_requests: int = 1) -> None:
        self._total_time_ms += total_ms          # <- summed, not wall-clock
        self._total_requests += num_requests

    def stats(self) -> dict[str, float]:
        rps = self._total_requests / (self._total_time_ms / 1000.0)
        ...
```

Under `concurrency=1` (the default) this is correct — calls are sequential,
so summed duration *is* wall-clock time. Under `concurrency>1`, calls
overlap in real time but their durations still get added up as if
sequential, understating true `rps`/tokens-per-sec by roughly the
concurrency factor (~3–4× has been observed). A wall-clock-based fix exists
in design but isn't merged.

### 3. No annotator integration for 3 pairwise metrics

**Where:** `src/auditkit/metrics/pairwise.py` — `win_rate`, `elo_score`,
`preference_accuracy`.

All three read fixed top-level `context` keys (`candidates`,
`pairwise_results`, `preference_data`) that no `Annotator`'s output ever
populates. `extract_with=` has no effect on any of them regardless of
configuration — they always fall back to a crude token-overlap proxy unless
you manually populate that exact `context` key yourself:

```python
import auditkit as ak
s = ak.Sample(input="q", target="a")
score = ak.WinRate().score(s, "some output")   # no context supplied
print(score.value)   # 0.0 -- silent proxy fallback, not an error
```

### 4. All 6 built-in benchmark scenarios fail to load

**Where:** `src/auditkit/scenarios/` — `mmlu`, `gsm8k`, `arc`, `hellaswag`,
`truthfulqa`, `humaneval`.

All point at stale/unqualified HuggingFace dataset references:

```python
from auditkit.registry import SCENARIOS
for name in SCENARIOS.names():
    list(SCENARIOS.get(name)().samples())
# arc/mmlu/truthfulqa -> DatasetNotFoundError
# gsm8k/hellaswag/humaneval -> HfUriError
```

`ak.run_lmeval()` (the real lm-evaluation-harness integration) is the
maintained path for academic benchmarks — it does not have this problem.

### 5. `RunResult.model_spec` doesn't serialize cleanly

**Where:** `src/auditkit/report.py`, `RunResult.to_dict()`.

Stores the live `Model` instance, not the original spec string:

```python
r = ak.evaluate([ak.Sample(input="q", target="a")], model=lambda p: ["a"], scorers=["exact_match"])
r.to_dict()["model_spec"]   # '<auditkit.model.CallableModel object at 0x...>' -- unusable after json.dump
```

Everything else on `RunResult` (`config`, `headline`, `predictions`)
round-trips correctly through save/load.

### 6. `num_completions`/`best_of` reach the API but never affect the score

**Where:** `src/auditkit/runner.py`, `src/auditkit/report.py`.

Adapters forward these into the request correctly, and a supporting backend
does generate the extra completions — but the scoring path only ever reads
`completions[0]`, so paying for `num_completions=5` changes cost/latency
without changing what gets scored. `RunConfig.extra: dict` is declared and
hashed into the fingerprint but never read anywhere in the codebase.

---

## Fixed since this document was written

### `HFGenModel` ignored `stop_sequences`/`seed`

**Where:** `src/auditkit/model/hf_gen.py`.

Every adapter forwarded both into `request.params` regardless of backend,
but `HFGenModel`'s key-mapping table listed neither.

**Fixed**: `stop_sequences` now maps to `generate()`'s real `stop_strings`
kwarg (which additionally needs the tokenizer passed alongside it — added
automatically whenever `stop_strings` is present). `seed` has no
per-call `generate()`/`pipeline()` kwarg at all; transformers' own
`set_seed()` (a global RNG reset) is applied right before the batched
`pipeline()` call when a request carries a seed. Verified live:

```python
import auditkit as ak
from auditkit.runspec import RunConfig
from auditkit.sample import Sample

samples = [Sample(input="Once upon a time", target="x")]
cfg = RunConfig(stop_sequences=["Late"], max_tokens=50, temperature=0.9, seed=42)
r = ak.evaluate(samples, model="hf:sshleifer/tiny-gpt2", scorers=["exact_match"], config=cfg)
r.predictions[0].raw_output  # '653 membership mutual factorsSexual Late' -- truncated exactly at the stop string

r1 = ak.evaluate(samples, model="hf:sshleifer/tiny-gpt2", scorers=["exact_match"],
                  config=RunConfig(seed=42, max_tokens=15, temperature=0.9))
r2 = ak.evaluate(samples, model="hf:sshleifer/tiny-gpt2", scorers=["exact_match"],
                  config=RunConfig(seed=42, max_tokens=15, temperature=0.9))
r1.predictions[0].raw_output == r2.predictions[0].raw_output  # True -- reproducible
```

Mocked regression tests added to `tests/test_generation_kwargs.py`
(`TestHFGenModelHonorsRequestParams`).

### `VLLMModel.model_info()` was unimplemented

**Where:** `src/auditkit/model/vllm_gen.py`.

Was marked `is_local=True` but didn't override `model_info()`, so it always
reported identity-only (no real parameter count/size) even though it
genuinely runs locally.

**Fixed** with graceful degradation, since vLLM's internal engine structure
has changed across its V0/V1 architectures and varies by release, with no
single stable public attribute for "the loaded `nn.Module`":
`model_info()` now tries a few known internal access paths in order (same
real `numel()`/`element_size()` approach as `HFGenModel.model_info()`) and
falls back to the previous identity-only reporting if none of them resolve
on the installed vLLM version, rather than crashing. Mocked tests added to
`tests/test_generation_kwargs.py` cover both the fallback path and a
resolved-path case using a real `torch.nn.Linear` module standing in for
vLLM's internal model object. **Update:** `vllm:` has since been run for
real (Colab L4 GPU) — see
[`docs/VLLM_KNOWN_ISSUES.md`](VLLM_KNOWN_ISSUES.md#2-modelinfo--real-introspection-but-fallback-path-never-confirmed-to-resolve)
for the full, current status of this specific gap and everything else
found running `vllm:` live.

### `ChatAdapter`'s stale source comment

**Where:** `src/auditkit/adapter.py`.

A comment claimed "No backend reads `request.params['messages']`" — no
longer true since `resolve_messages()` was wired into 5 backends
(`openai.py`/`anthropic.py`/`groq.py`/`litellm.py`/chat-mode `api.py`) and
`hf_gen.py`'s own chat-template rendering also reads it. Documentation-only,
never affected runtime behavior — comment corrected to describe the real
current state (flattened text stays the one guaranteed-reachable form for
`EchoModel`/`CallableModel`/bare-callable models, `messages` used by
backends that prefer real chat turns).

### `load_croissant()` called a function that doesn't exist — always crashed

**Where:** `src/auditkit/loaders.py`.

Found while building a real-data test for it: `load_croissant()` called
`mlcroissant.load(path, **kwargs)`, but the real `mlcroissant` package has no
top-level `load()` function at all — **every single call raised
`AttributeError: module 'mlcroissant' has no attribute 'load'`, unconditionally**,
for any input. `tests/test_loaders.py` never caught this because its only
`load_croissant` test checked the "extra not installed" error path (and
lacked the `skipif` guard `load_hf`'s equivalent test has, so it silently
"passed" for the wrong reason whenever `mlcroissant` happened to be absent).

**Fixed** to use the real API — `mlcroissant.Dataset(jsonld=path).records(record_set)`
— which also required two related fixes: a new required `record_set`
parameter (a Croissant file can describe several record sets; there's no
universally correct default), and decoding field values (the real API
returns `bytes`, not `str`). Verified against real data:

```python
from auditkit.loaders import load_croissant
samples = load_croissant(
    "https://huggingface.co/api/datasets/rag-datasets/rag-mini-wikipedia/croissant",
    "question-answer", input_col="question", target_col="answer",
)
len(samples)  # 918 -- matches load_hf() on the same dataset exactly
```

Real functional tests added for both `load_hf` and `load_croissant`
(`tests/test_loaders.py`) — previously neither had one, only "is callable"/
"extra not installed" checks.

One more gotcha found while verifying this against a live dataset:
`mlcroissant` needs `GitPython` for some real datasets' git-based download
operations but doesn't declare it as a dependency itself — `pip install
auditkit[interop]` would hit `ModuleNotFoundError: No module named 'git'`
on those datasets. Added `GitPython` to the `interop` extra in
`pyproject.toml` so it's covered out of the box.

### `bert_score` was thought unfixable from this repo — it wasn't

**Where:** `src/auditkit/metrics/generation.py`, `BertScore.score()`.

Previously documented as a genuine, unfixable `bert_score`/`transformers>=5`
incompatibility: `bert_score.score()` raised `OverflowError: int too big to
convert` deep inside the tokenizer's truncation setup. Root-caused this
session instead of taking that at face value:

`AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli").model_max_length`
is `1000000000000000019884624838656` — HuggingFace's classic "no limit
configured" sentinel value. `bert_score`'s `sent_encode()` passes this
straight into `tokenizer.encode(sent, ..., truncation=True)`, and
`transformers>=5`'s newer Rust-backed truncation setup
(`set_truncation_and_padding` → `self._tokenizer.enable_truncation(**target)`)
tries to convert that sentinel into a bounded integer type and overflows.
`bert_score` itself never clamps this value before use.

**Fixed** with a scoped monkeypatch: `BertScore.score()` temporarily replaces
`bert_score.score.get_tokenizer` (note: `bert_score/__init__.py` does `from
.score import score`, which shadows the `bert_score.score` *attribute* with
that function — the submodule must be reached via
`sys.modules["bert_score.score"]`, not attribute access) with a wrapper that
clamps `tok.model_max_length` to `512` whenever it exceeds `100_000`, calls
the real `bert_score.score()`, then restores the original function in a
`finally` block. Verified live through the real `evaluate()` pipeline, both
for correctness and for discriminating power (not just a degenerate
always-near-1.0 result):

```python
import auditkit as ak
from auditkit.metrics.generation import BertScore
from auditkit.sample import Sample

identical = ak.evaluate(
    [Sample(input="x", target="cat sat on mat quietly")],
    model=lambda p: ["cat sat on mat quietly"], scorers=[BertScore()],
)
identical.headline["bert_score"]   # 0.9999998807907104

unrelated = ak.evaluate(
    [Sample(input="x", target="cat sat on mat quietly")],
    model=lambda p: ["completely unrelated topic about rockets"], scorers=[BertScore()],
)
unrelated.headline["bert_score"]   # 0.4300842583179474 -- correctly low, not masked
```

`tests/test_metrics_via_pipeline.py::test_bert_score` now asserts success
directly instead of skipping on the known failure; a second test,
`test_bert_score_discriminates_unrelated_text`, locks in the discriminating
behavior above.

### `cosine_similarity`/`factual_consistency` had zero functional test coverage

Both metrics load a real model (`sentence-transformers`/`transformers`
respectively) and were only tested for the "extra not installed" error path
and `.name` — never for whether the scoring logic works.
Verified live and confirmed both work correctly (identical text → high
similarity/entailment, unrelated text → low similarity/contradiction); see
`examples/12_coverage_gap_verification.ipynb` sections 4–5 for the real,
executed output.

---

## Real limitations (by design or upstream — not bugs to fix)

- **`vllm:`/`lexsi:` don't apply a chat template to flat prompts.** `hf:`
  (via the model's own `tokenizer.chat_template`) and the hosted chat APIs
  (`openai:`/`anthropic:`/`groq:`/`litellm:`) format an adapter's prompt
  correctly for an instruct model; `vllm:` and `lexsi:` send the raw flat
  prompt, under-formatting an instruct model served that way. Full detail,
  plus everything else found running `vllm:` live, now in
  [`docs/VLLM_KNOWN_ISSUES.md`](VLLM_KNOWN_ISSUES.md).
- **`litellm:` backend has never been verified against the real, installed
  library making a real call.** Real (non-stub) implementation with
  mocked unit-test coverage of its parameter mapping
  (`tests/test_generation_kwargs.py`), but the package isn't installed in
  this dev environment and there's no live-pipeline test for it (unlike
  `tests/test_pipeline_live_hf.py` for `hf:`). Still open.
- **`vllm:` backend HAS now been verified live** (Colab L4 GPU) — real
  generation confirmed working end to end. See
  [`docs/VLLM_KNOWN_ISSUES.md`](VLLM_KNOWN_ISSUES.md) for the full list of
  real implementation gaps and environment/dependency issues found doing
  so, several already fixed directly in `VLLMModel`.
- **`HFGenModel.loglikelihood()`** does one forward pass per request, not
  batched — correctness was prioritized over throughput for the first cut.
- **`representation_skew`** (replaced the mislabeled `bias_score`) measures
  demographic-*representation* balance, not bias — by design it can't see
  meaning (a balanced-but-sexist sentence scores 0.0). Now explicit in the
  name/docstring, not a hidden flaw. Use `bias_judge` for biased content.
- **Bootstrap significance is a conservative heuristic**, not a rigorous
  test — `paired_bootstrap` can report `significant=False` for a large but
  low-variance effect (every sample flips the same way).
- **Model comparison is quality-only unless you supply sizes** —
  `RunComparison.tradeoff()` computes size/latency ratios only from numbers
  measured during the run itself; there's no lineage-based automatic
  baseline selection.

---

## How to re-verify any entry yourself

Every reproduction above is copy-pasteable. If one no longer reproduces by
the time you're reading this, it's fixed — update or remove that entry
rather than leaving a stale bug report in place.
