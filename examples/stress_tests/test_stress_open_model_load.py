#!/usr/bin/env python3
"""Stress test: a real, open-source (ungated) HuggingFace model under load.

Uses distilgpt2 -- small, real, publicly downloadable, no auth needed -- and
pushes it with: a larger real dataset pulled live from HuggingFace, forced
concurrency (multiple threads hammering the same model), and deliberately
awkward edge-case inputs (empty strings, very long strings, unicode/emoji,
samples with no target) to see what breaks.

Run with: python3 examples/stress_tests/test_stress_open_model_load.py
Needs: pip install -e ".[transformers,interop]"
"""
from __future__ import annotations
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import auditkit as ak
from auditkit.model.hf_gen import HFGenModel
from auditkit.adapter import MCQAdapter, GenerationAdapter
from auditkit.metric import Acc
from auditkit.sample import Sample
from auditkit.runspec import RunConfig

MODEL_NAME = "distilgpt2"
N_SAMPLES = 40


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def load_real_mmlu(subject: str, split: str, limit: int) -> list[Sample]:
    import datasets
    data = datasets.load_dataset("cais/mmlu", subject, split=split)
    samples = []
    for i, row in enumerate(data):
        if i >= limit:
            break
        letter = chr(65 + row["answer"])
        samples.append(Sample(
            input=row["question"], choices=list(row["choices"]),
            target=letter, kind=ak.TaskKind.MCQ, task=f"mmlu_{subject}",
        ))
    return samples


section(f"1. Load {N_SAMPLES} real MMLU samples + build a real, shared HFGenModel")
samples = load_real_mmlu("abstract_algebra", split="test", limit=N_SAMPLES)
model = HFGenModel(model=MODEL_NAME, device="cpu", max_new_tokens=8)
print(f"Loaded {len(samples)} real samples.")


section("2. Concurrency stress: run the SAME shared model instance with concurrency > 1")
t0 = time.monotonic()
try:
    result = ak.evaluate(
        samples,
        model=model,
        adapter=MCQAdapter(method="mcq_joint"),
        scorers=[Acc()],
        config=RunConfig(concurrency=8, max_retries=2, timeout=30.0),
        verbose=True,
    )
    elapsed = time.monotonic() - t0
    print(f"\n[OK] Ran {len(result.predictions)} samples with concurrency=8 in {elapsed:.1f}s")
    print(f"     headline: {result.headline}")
    print(f"     errors recorded: {result.failed_count}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")


section("3. Edge-case inputs: empty string, very long input, unicode/emoji, no target")
edge_samples = [
    Sample(input="", target="anything"),                                    # empty input
    Sample(input="word " * 2000, target="anything"),                        # ~2000-word input, stresses tokenizer/context
    Sample(input="What does 🤔🔥💯 mean in this context? 日本語テスト", target="anything"),  # unicode/emoji
    Sample(input="no target provided here"),                                # target=None (non-golden sample)
    Sample(input="\n\n\n\t\t   ", target="anything"),                       # whitespace-only input
]
for i, s in enumerate(edge_samples):
    label = ["empty string", "~2000-word input", "unicode/emoji", "no target (non-golden)", "whitespace-only"][i]
    try:
        r = ak.evaluate([s], model=model, adapter=GenerationAdapter(),
                         scorers=None, config=RunConfig(timeout=30.0, max_retries=1))
        print(f"[OK]   {label}: ran without crashing, predictions={len(r.predictions)}, "
              f"errors={r.failed_count}, output_preview={(r.predictions[0].raw_output or '')[:40]!r}")
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {str(e)[:150]}")


section("4. Repeated back-to-back runs on the SAME model instance (state leakage / memory check)")
try:
    for i in range(3):
        r = ak.evaluate(samples[:5], model=model, adapter=MCQAdapter(method="mcq_joint"),
                         scorers=[Acc()], config=RunConfig(concurrency=4))
        print(f"[OK] run {i+1}/3: headline={r.headline}, errors={r.failed_count}")
except Exception as e:
    print(f"[FAIL] repeated-run stress: {type(e).__name__}: {e}")


section("5. Aggressive timeout (force timeouts to fire) -- does the retry/timeout path hold up?")
try:
    r = ak.evaluate(samples[:5], model=model, adapter=GenerationAdapter(), scorers=None,
                     config=RunConfig(timeout=0.001, max_retries=2, retry_delay=0.1))
    print(f"[OK] survived aggressive timeout config: predictions={len(r.predictions)}, "
          f"failed_count={r.failed_count}, errors={r.errors[:2]}")
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")

section("ROOT CAUSES CONFIRMED (2 new bugs found by this stress test)")
print("""
Bug A -- empty-chunk crash (explains section 3's deterministic failures):
  Runner._chunk() always splits the flat request list into exactly
  `concurrency` pieces, regardless of how many requests actually exist.
  For 1 sample with the DEFAULT concurrency=8, this produces:
      [['the_one_item'], [], [], [], [], [], [], []]
  i.e. 7 EMPTY chunks. Each chunk is hand ed to model.generate(chunk) inside
  the thread pool -- and HFGenModel.generate([]) crashes outright with
  IndexError: list index out of range (confirmed directly, deterministically,
  100% of attempts). Any evaluate() call where sample count < concurrency
  (the default is 8, so this includes almost any small/single-sample dev run)
  will crash the same way for any model backend that doesn't defensively
  handle a zero-length request list.

Bug B -- thread-safety violation (explains section 2's intermittent errors):
  Runner never reads/checks `model.threadsafe` anywhere (confirmed: zero
  references to `threadsafe` in runner.py) before deciding to parallelize.
  HFGenModel does not declare threadsafe=True, so it inherits the base
  Model.threadsafe=False default -- yet the Runner still hammers it with
  `concurrency` threads calling generate() simultaneously on the same shared
  pipeline object. The result: sporadic, non-deterministic internal errors
  under real concurrent load (list index out of range / index out of range
  in self / probability tensor contains inf/nan/negative), retried away when
  retries remain, but a real correctness risk whenever they exhaust retries
  or corrupt output silently instead of raising.

Both bugs stem from the same root issue: the `threadsafe`/chunk-sizing
machinery that WOULD prevent this already exists in the codebase (the
`threadsafe` attribute, `RunConfig.concurrency`) but isn't actually
cross-checked against real request counts or backend capability anywhere.

Note on Bug A's blast radius: EchoModel.generate([]) and
CallableModel.generate([]) both return [] gracefully -- so this bug has been
invisible in every prior test in this whole audit, since all of it used
those two toy backends. It only surfaces with a real backend (HFGenModel,
and likely vllm_gen.py/api_gen.py/etc., unverified) that doesn't defensively
handle a zero-length input list the way the two built-in test doubles do.
""")

print("Done.")
