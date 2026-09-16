# AuditKIT — Stress Tests

Separate from the functional coverage in `tests/`. This folder is about
**robustness under real, heavier conditions**: real gated models, real load,
concurrency, and
deliberately awkward inputs — the kind of testing that surfaces bugs regular
functional tests don't, because it needs actual network access, a real
downloaded model, and adversarial conditions to trigger.

Both scripts here found **new bugs not previously documented** in
`AUDIT_FINDINGS.md`/`KEY_ISSUES.md`. See each script's own output for full
detail; summarized below.

## Files

| Script | What it does |
|---|---|
| `test_stress_gated_model.py` | Attempts a real, currently-gated HuggingFace model (`google/gemma-2b`) with no access token, through both the raw backend and the full `evaluate()` pipeline, to reproduce and explain exactly why gated models don't work. |
| `test_stress_open_model_load.py` | Runs a real, ungated model (`distilgpt2`) under load: 40 real MMLU samples with concurrency=8, deliberately awkward edge-case inputs (empty string, ~2000-word input, unicode/emoji, no-target sample, whitespace-only), repeated back-to-back runs, and an aggressive timeout. |

## How to run

```bash
cd AuditKIT                  # repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[transformers,interop]"

python3 examples/stress_tests/test_stress_gated_model.py
python3 examples/stress_tests/test_stress_open_model_load.py
```

Both need internet access (they download real datasets/models on first run).

## What to expect / key findings

### `test_stress_gated_model.py`

Confirms `HFGenModel` has **no `token`/`use_auth_token` parameter at all** —
gated models (anything requiring HuggingFace license acceptance, e.g. Llama,
Gemma) can only work if the *host machine* already has a valid HF login
cached outside of auditkit entirely; there's no way to supply a token through
auditkit's own API. Two distinct failure modes were reproduced:

1. A bare, tokenizer-only access attempt gets a clean `401` error straight
   from HuggingFace explaining exactly what's wrong.
2. Going through the real model-loading path (and through `evaluate()`)
   produces a *different, more confusing* error — a `ValueError` claiming
   the model "does not appear to have" its weight files, because the
   unauthenticated file-listing call can't see the (present but
   access-restricted) files, so they look missing rather than denied.

Additionally: going through `evaluate()`, the final error the user sees is a
generic `ModelError: generate failed after 3 retries` that says nothing
about authentication — the real cause is preserved via Python's exception
chaining (`e.__cause__`) but isn't surfaced in the top-level message.

**⚠️ Note on environment-specific behavior:** this sandbox happened to have a
globally-cached HF token from a prior session, which made the *first* attempt
at loading a gated model succeed unexpectedly. The script works around this
by explicitly forcing an invalid token so the test reproduces what a typical
user *without* pre-existing access actually experiences. If you run this on
a machine with your own valid, already-approved HF login, gated models may
"just work" for the same underlying reason (ambient credentials, not
anything auditkit provides) — that's expected, not a test failure.

### `test_stress_open_model_load.py`

Found **two new bugs**, neither previously documented:

**Bug A — empty-chunk crash.** `Runner._chunk()` always splits requests into
exactly `concurrency` pieces regardless of how many requests exist. With
`RunConfig`'s default `concurrency=8` and fewer than 8 samples (including the
single-sample case — confirmed deterministic, 100% reproduction), most chunks
are empty lists, and `model.generate([])` crashes for `HFGenModel` with
`IndexError: list index out of range`. This means **almost any small or
single-sample `evaluate()` call against a real local HF model crashes by
default**, unless the caller happens to set `concurrency` low enough or has
enough samples. `EchoModel`/`CallableModel` both tolerate empty lists
gracefully (`generate([]) -> []`), which is why this bug was invisible in
every earlier test in this whole audit — they were the only backends used.

**Bug B — thread-safety violation.** `Runner` never checks `model.threadsafe`
(confirmed zero references anywhere in `runner.py`) before deciding to
parallelize. `HFGenModel` doesn't declare `threadsafe=True`, so it inherits
the base `Model.threadsafe=False` default — yet `Runner` still dispatches
`concurrency` threads to call `generate()` on the same shared, non-threadsafe
pipeline object simultaneously. Confirmed live: a real 40-sample concurrent
run produced many sporadic, non-deterministic internal errors (`list index
out of range`, `index out of range in self`, `probability tensor contains
inf/nan/negative`) — retried away in this run since retries remained, but a
genuine correctness risk (silent corruption or eventual failure) under
sustained real-world load.

Expect: the concurrency section to complete with many visible `Retry` log
lines (this is Bug B manifesting, survived via retries) but a `0` final
`failed_count`; the edge-case section to fail all 5 cases with `ModelError`
(this is Bug A, triggered because there's only 1 sample per case against the
default `concurrency=8`); the repeated-run section to pass cleanly (no state
leakage across repeated calls); and the aggressive-timeout section to
correctly raise `ModelTimeout` (this one is *correct*, expected behavior).

## Updated bug count

These 2 new bugs are not yet folded into `AUDIT_FINDINGS.md`/`KEY_ISSUES.md`'s
numbered list (they were found after that tracking was last updated) — ask
to have them added if you want the running total updated to 15.
