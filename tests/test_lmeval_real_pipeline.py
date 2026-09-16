"""Real, unmocked lm-eval engine pipeline test.

Unlike test_lmeval_engine.py (which injects a fake lm_eval module so the
spec-mapping/result-mapping logic can be tested without the heavy optional
dependency installed), this exercises the real thing end-to-end: a real
"arc_easy" (ARC-Easy) task, a real "gpt2" checkpoint downloaded from the
Hugging Face Hub, real CPU loglikelihood inference via
lm_eval.simple_evaluate(), and the real DiskCache persistence added to
run_benchmark() -- confirming lm-eval runs are now cached exactly like native
engine runs (same fingerprint -> reused RunResult, no re-run).

Needs the `lmeval` extra (`pip install auditkit[lmeval]`, i.e. `lm-eval`
+ `accelerate`) actually installed, plus network access to the HF Hub for the
first run in a given environment (gpt2 + arc_easy are both public, no token
needed). Skipped entirely if either import is missing -- this environment's
default dev install does not include them, so the test only runs when the
lmeval extra has actually been set up.

gpt2 is a tiny, non-instruction-tuned base model chosen purely for speed
(~80s cold for 3 samples on CPU); its near-random accuracy on ARC-Easy below
is the real, expected result for a model this small -- not a test bug.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("lm_eval")
pytest.importorskip("accelerate")

from auditkit.cache import DiskCache
from auditkit.lmeval_engine import run_benchmark
from auditkit.runspec import RunConfig

TASK = "arc_easy"
MODEL = "hf:gpt2"


@pytest.fixture(scope="module")
def lmeval_cache_dir(tmp_path_factory):
    """A throwaway cache dir for the whole module, so this real run never
    touches (or gets confused by) the developer's real ~/.cache/auditkit."""
    d = tmp_path_factory.mktemp("auditkit_lmeval_real_cache")
    os.environ["XDG_CACHE_HOME"] = str(d)
    yield d


@pytest.fixture(scope="module")
def real_result(lmeval_cache_dir):
    """One real lm_eval.simple_evaluate() call, reused by every test below --
    real HF Hub download of gpt2 + arc_easy (first time only), real CPU
    inference, real loglikelihood scoring for 3 real ARC-Easy questions."""
    return run_benchmark(
        TASK, MODEL, config=RunConfig(limit=3, num_fewshot=0), device="cpu",
    )


# --- Exact real input/output captured from a live run ------------------------
#
# Real ARC-Easy prompts (as lm-eval's loglikelihood scoring sends them),
# real gold target indices, and gpt2's real argmax choice per question:
REAL_PROMPTS = [
    "Question: Which statement best explains why photosynthesis is the "
    "foundation of most food webs?\nAnswer:",
    "Question: Which piece of safety equipment is used to keep mold spores "
    "from entering the respiratory system?\nAnswer:",
    "Question: Meiosis is a type of cell division in which germ cells divide "
    "to produce haploid cells. Where does meiosis occur?\nAnswer:",
]
REAL_EXPECTED = ["0", "1", "3"]       # gold choice index, per question
REAL_RAW_OUTPUTS = ["2", "2", "0"]    # gpt2's real argmax choice index, per question


def test_headline_reflects_real_argmax_scoring(real_result):
    assert set(real_result.headline) == {"arc_easy:acc", "arc_easy:acc_norm"}
    assert real_result.headline["arc_easy:acc"] == 0.0
    assert real_result.headline["arc_easy:acc_norm"] == pytest.approx(1 / 3)


def test_real_per_sample_predictions_are_the_answer_browser(real_result):
    assert len(real_result.predictions) == 3
    assert [p.prompt for p in real_result.predictions] == REAL_PROMPTS
    assert [p.expected for p in real_result.predictions] == REAL_EXPECTED
    assert [p.raw_output for p in real_result.predictions] == REAL_RAW_OUTPUTS
    # real gpt2 got all 3 of these real questions wrong -- expected for a
    # tiny, non-instruction-tuned base model on ARC-Easy multiple choice
    assert all(p.correct is False for p in real_result.predictions)
    assert all(
        p.metadata == {"engine": "lmeval", "primary_metric": "acc"}
        for p in real_result.predictions
    )


def test_disk_cache_file_actually_written(real_result, lmeval_cache_dir):
    assert real_result.fingerprint in DiskCache()
    cache_path = DiskCache()._path(real_result.fingerprint)
    assert cache_path.exists()
    assert str(lmeval_cache_dir) in str(cache_path)


def test_second_call_with_identical_config_hits_diskcache_not_lm_eval(real_result, monkeypatch):
    """The core of today's change: an identical run_benchmark() call must
    reuse the cached RunResult and never invoke lm_eval.simple_evaluate again.
    The real function is wrapped (not replaced), so if this assertion is
    wrong the wrapper would still delegate to a real, working call -- this
    only fails if the cache is bypassed, not if lm-eval itself breaks."""
    import lm_eval

    calls = {"n": 0}
    real_simple_evaluate = lm_eval.simple_evaluate

    def counting_wrapper(**kwargs):
        calls["n"] += 1
        return real_simple_evaluate(**kwargs)

    monkeypatch.setattr(lm_eval, "simple_evaluate", counting_wrapper)

    r2 = run_benchmark(TASK, MODEL, config=RunConfig(limit=3, num_fewshot=0), device="cpu")

    assert calls["n"] == 0  # cache hit -- the real lm_eval call never happened
    assert r2.fingerprint == real_result.fingerprint
    assert r2.headline == real_result.headline
    assert len(r2.predictions) == len(real_result.predictions)


def test_different_config_is_a_different_fingerprint_and_a_real_second_run(
    lmeval_cache_dir, monkeypatch
):
    """A genuinely different config (limit=2 instead of 3) must NOT be
    confused with the cached limit=3 result -- and must trigger a real
    second lm_eval.simple_evaluate call, not silently reuse stale data."""
    import lm_eval

    calls = {"n": 0}
    real_simple_evaluate = lm_eval.simple_evaluate

    def counting_wrapper(**kwargs):
        calls["n"] += 1
        return real_simple_evaluate(**kwargs)

    monkeypatch.setattr(lm_eval, "simple_evaluate", counting_wrapper)

    r3 = run_benchmark(TASK, MODEL, config=RunConfig(limit=2, num_fewshot=0), device="cpu")

    assert calls["n"] == 1  # a real, different config -- lm_eval was really re-invoked
    assert r3.headline["arc_easy:acc_norm"] == pytest.approx(0.5)
