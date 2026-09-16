# %% [markdown]
# # Model comparison — testing notebook
#
# Compare a **baseline** model against a **candidate** (base vs pruned/quantized,
# or any two models) on the SAME eval config, for native, lm-eval benchmark, and
# LLM-as-judge runs. Run cell-by-cell (VS Code "Run Cell" or Jupyter).
#
# **The whole thing is one idea:** every run returns a `RunResult`; comparison
# diffs two of them. So it works regardless of how the run was produced.
#
# Prerequisites (only for the cells you run):
# - HF local models:      `pip install -e ".[transformers]"`  (+ pass device="cpu" on a CPU box)
# - lm-eval benchmark:    `pip install -e ".[lmeval]"`
# - Groq (closed API):    `pip install -e ".[requests]"`  and set `GROQ_API_KEY`
# - gated HF models:      pass `hf_token="hf_..."`
#
# The core comparison API:
#   ak.compare(baseline_run, candidate_run)          -> RunComparison
#   ak.compare_models([...], data).pairwise(a, b)    -> RunComparison   (runs both, then diffs)
# RunComparison:  .grade() .grades() .metric_deltas() .per_task_deltas()
#                 .retention(metric) .significance(metric)
#                 .tradeoff(metric=...) .performance()   # size/latency measured automatically
#                 .regressed() .improved() .failure_summary() .coverage_warnings() .summary()

# %%
# --- setup ---------------------------------------------------------------
import os
import auditkit as ak

# A small multi-task dataset (defined ONCE and reused, so sample ids align
# across every run — that's what lets the comparison diff per sample).
DATA = [
    ak.Sample(input="What is 2+2?",            target="4",     task="arith"),
    ak.Sample(input="What is 3+3?",            target="6",     task="arith"),
    ak.Sample(input="What is 5+5?",            target="10",    task="arith"),
    ak.Sample(input="Capital of France?",      target="Paris", task="trivia"),
    ak.Sample(input="Capital of Italy?",       target="Rome",  task="trivia"),
    ak.Sample(input="Capital of Japan?",       target="Tokyo", task="trivia"),
]
print("auditkit", ak.__version__, "| dataset:", len(DATA), "samples, tasks: arith, trivia")


# %% [markdown]
# ## 1. Offline smoke test (no downloads, no keys)
# Proves the comparison machinery works using two plain Python functions as
# "models". Baseline is perfect; the candidate regresses on `trivia` only —
# so per-task grading should isolate that.

# %%
def baseline_fn(prompts):                       # gets everything right
    key = {"What is 2+2?": "4", "What is 3+3?": "6", "What is 5+5?": "10",
           "Capital of France?": "Paris", "Capital of Italy?": "Rome", "Capital of Japan?": "Tokyo"}
    return [key[p] for p in prompts]

def candidate_fn(prompts):                      # arith correct, trivia broken
    key = {"What is 2+2?": "4", "What is 3+3?": "6", "What is 5+5?": "10"}
    return [key.get(p, "WRONG") for p in prompts]

base = ak.evaluate(DATA, model=baseline_fn,  scorers=["exact_match"])
cand = ak.evaluate(DATA, model=candidate_fn, scorers=["exact_match"])

cmp = ak.compare(base, cand)                    # <-- the comparison
print(cmp.summary())
print("\noverall grade:", cmp.grade())
print("per-task:", [(t.task, round(t.delta, 3), t.grade.value) for t in cmp.per_task_deltas()])
print("retention:", cmp.retention("exact_match"))
print("regressed sample ids:", [p.sample_id for p in cmp.regressed()])


# %% [markdown]
# ## 2. HF vs HF (native `ak.evaluate`, exact-match)
# Base-vs-pruned pattern. Swap in your real base/pruned specs. Needs
# `auditkit[transformers]`; `device="cpu"` on a machine without CUDA.

# %%
BASE_HF      = "hf:sshleifer/tiny-gpt2"     # <- your base model
CANDIDATE_HF = "hf:gpt2"                     # <- your pruned/quantized model
try:
    b = ak.evaluate(DATA, model=BASE_HF,      scorers=["exact_match"], device="cpu")
    c = ak.evaluate(DATA, model=CANDIDATE_HF, scorers=["exact_match"], device="cpu")
    cmp = ak.compare(b, c)
    print(cmp.summary())
    print("retention:", cmp.retention("exact_match"))
except ak.ExtraNotInstalled as e:
    print("SKIP — install the extra:", e)
except Exception as e:
    print("SKIP —", type(e).__name__, str(e)[:200])


# %% [markdown]
# ## 3. HF (local) vs Groq (closed API) on the lm-eval benchmark
# Compares two `ak.run_lmeval` runs on the same task. Use a **generative** task
# (gsm8k): `groq:` maps to lm-eval's openai-chat-completions backend, which is
# chat-only (no logprobs), so MCQ-by-loglikelihood tasks like arc/mmlu are
# rejected up front — pick a generation-scored task. Needs `auditkit[lmeval]`
# and `GROQ_API_KEY` (mirrored to the backend's OPENAI_API_KEY for the run only).

# %%
TASK = "gsm8k"
LIMIT = 5
GROQ_MODEL = "groq:llama-3.3-70b-versatile"
try:
    hf_run = ak.run_lmeval(TASK, model="hf:gpt2", limit=LIMIT, device="cpu")
    if os.getenv("GROQ_API_KEY"):
        groq_run = ak.run_lmeval(TASK, model=GROQ_MODEL, limit=LIMIT)
        cmp = ak.compare(hf_run, groq_run)       # baseline=hf, candidate=groq
        print(cmp.summary())
        # significance uses the primary per-sample score (metric=None) — works for benchmark runs
        print("significance:", cmp.significance())
    else:
        print("hf run headline:", hf_run.headline, "\n(set GROQ_API_KEY to compare against Groq)")
except ak.ExtraNotInstalled as e:
    print("SKIP — install the extra:", e)
except Exception as e:
    print("SKIP —", type(e).__name__, str(e)[:200])


# %% [markdown]
# ## 4. LLM-as-judge comparison — HF (local) vs Groq (closed API)
# A judge scores EACH model's answers independently; then we diff the judge
# scores. This is the "open local model vs closed API" comparison, done in the
# native engine where `groq:` works. A Groq model is also the judge here.
# Needs `auditkit[requests]` (+ `[transformers]` for the HF side) and `GROQ_API_KEY`.

# %%
GROQ_MODEL = "groq:llama-3.3-70b-versatile"   # any Groq-hosted model id
QA = [
    ak.Sample(input="Explain why the sky is blue in one sentence.", task="explain"),
    ak.Sample(input="Summarize what a CPU does in one sentence.",   task="explain"),
]
try:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("set GROQ_API_KEY to run the judge")
    judge = ak.LLMJudge(
        judge_model=GROQ_MODEL,
        prompt="Question: {input}\nAnswer: {output}\nIs the answer correct and clear?",
        choices={"yes": 1.0, "partial": 0.5, "no": 0.0},
        use_cot=True, name="quality",
    )
    b = ak.evaluate(QA, model="hf:gpt2", scorers=[judge], device="cpu")   # baseline = local HF
    c = ak.evaluate(QA, model=GROQ_MODEL, scorers=[judge])               # candidate = Groq API
    cmp = ak.compare(b, c)                        # diff on the judge's "quality" score
    print(cmp.summary())
    print("retention(quality):", cmp.retention("quality"))
except Exception as e:
    print("SKIP —", type(e).__name__, str(e)[:200])


# %% [markdown]
# ## 5. N-model bake-off + pairwise + per-task tables
# `compare_models` runs several models on the same dataset (per-model config /
# backend overrides allowed), gives a leaderboard, then `.pairwise()` zooms into
# any two as a baseline-anchored `RunComparison`.

# %%
res = ak.compare_models(
    [baseline_fn, candidate_fn, baseline_fn],
    DATA, scorers=["exact_match"],
    model_names=["base", "pruned", "quantized"],
    # per-model overrides, e.g. a quantized model needing different backend kwargs:
    # model_opts={"quantized": {"dtype": "int4"}},
    # configs={"pruned": ak.RunConfig(limit=6)},
)
print(res.summary())                              # leaderboard + per-task tables + significance
print("\nwinner:", res.winner("exact_match"))
print("coverage warnings:", res.coverage_warnings())   # flags failed/mismatched samples
cmp = res.pairwise("base", "pruned")              # zoom into two -> RunComparison
print("\nbase vs pruned grade:", cmp.grade())


# %% [markdown]
# ## 6. Quality-vs-size/latency tradeoff (pruning/quant payoff)
# Everything here is **measured automatically** — latency/throughput are timed
# by the runner as it runs, and model size is introspected from local HF
# checkpoints (`Model.model_info()`). Nothing is passed in by hand.
#
# - `performance()` — latency + throughput, works for ANY run (incl. these
#   callable runs from cell 1).
# - `tradeoff(metric=...)` — folds in quality + size. `size_ratio`/
#   `quality_per_mb` only appear when BOTH sides are local models with a real
#   on-disk size (i.e. the HF runs in cell 2); callable/API runs show latency
#   only, with `api_based` naming the two models instead.

# %%
cmp = ak.compare(base, cand)                      # reuse the offline callable runs from cell 1
print("performance:", cmp.performance())          # measured latency + throughput
print("tradeoff:   ", cmp.tradeoff(metric="exact_match"))
# For a real size payoff, run cell 2's HF pair, then: ak.compare(b, c).tradeoff(metric="exact_match")
# -> retention, size_ratio (<1 = smaller), quality_per_mb, latency_ratio, speedup (>1 = faster)
