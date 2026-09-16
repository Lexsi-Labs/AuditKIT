# Examples

Colab-ready notebooks demonstrating AuditKit end to end, with real models
and real datasets — not synthetic stand-ins. Each notebook's own first
cells detect Colab and self-install `auditkit` from source (it isn't on
PyPI yet); outside Colab they fall through and assume it's already
installed locally.

| File | Description |
|------|-------------|
| 01_full_evaluation_pipeline.ipynb | Real GSM8K (30 rows) via 3 real, local, gated Llama checkpoints (generation/extraction/judging): `Sample`'s every field, `evaluate()`'s flexible inputs, `GenerationAdapter`, 6 real metrics (`QuasiExactMatch`, `WordCount`, `Regex`, a custom `@scorer`, `LLMJudge`, `EncoderJudge`/`FactualityEncoderJudge`), and `LLMAnnotator` extraction (with a `cast=` regex cleanup, since `Llama-2-13b-chat` routinely preambles its reply despite being told not to). Pass 3 scores the same `LLMAnnotator`-extracted answers with a real BERT-family classifier judge (`EncoderJudge`) at zero extra generation cost -- templated as full question/answer sentences (`text_template=`/`text_pair_template=`) rather than bare numbers, which is what makes an NLI-style judge discriminate reliably. Needs a GPU + HF token |
| 02_generation_across_hf_families.ipynb | Real local generation compared across two different HF model families (`Qwen2.5-3B-Instruct` vs. `Phi-3.5-mini-instruct`) on the same real GSM8K slice, via `compare_models()` -- a `TemplateAdapter`+`RegexAnnotator` extraction pass (`extract_with=`) gets the model's free-form reasoning down to a bare number before `QuasiExactMatch` scores it. Requires a GPU runtime |
| 03_custom_annotators.ipynb | `RegexAnnotator`, `ThinkingStripAnnotator` (strips `<think>` blocks before extracting), `LLMAnnotator`, and a fully custom `Annotator` subclass with no model call at all, wired into scoring via `extract_with=` -- real local HF models (`Qwen2.5-3B`/`1.5B-Instruct` on a Colab GPU), no external API/rate limits |
| 04_metrics_deep_dive.ipynb | Built-in scorers across families (`QuasiExactMatch`, `Contains`, `F1Score`, `Bleu`, `RogueL`, `BertScore`), a custom `@scorer`, LLM-as-judge (`LLMJudge` + rubric-driven `GEval`), and RAG's `LexicalGroundedness`. `BertScore` gets a dedicated section proving (live-verified, not asserted) it needs real sentence-level text to discriminate meaningfully -- a wrong bare short answer scores ~0.80-0.94 (barely below a correct one's ~1.00), while a genuinely unrelated real sentence scores ~0.66 against a correct paraphrase's ~0.86-0.89 |
| 05_data_types.ipynb | Every `Sample`/task shape through its matching adapter on one real model: generative, MCQ (`Acc`), RAG (`LexicalGroundedness`), precomputed (zero new API calls), and chat. The generative and MCQ sections each add an `LLMAnnotator` extraction pass -- `QuasiExactMatch`/`Acc` both require an *exact* match to a bare answer/choice index, which a real free-text reply never gives without one |
| 06_model_comparison.ipynb | `compare_models()` across 3 real models (per-metric winners, significance, measured performance) and `RunComparison` (baseline-vs-candidate `grade()`/`retention()`/`regressed()`/`tradeoff()`) via both `ak.compare()` and `CompareResult.pairwise()`. Uses `compare_models()`'s own `adapter=`/`annotators=`/`extract_with=` support (a `TemplateAdapter`+`RegexAnnotator` extraction pass) -- it does support them, despite what an earlier version of this notebook claimed |
| 07_annotators_across_models.ipynb | 4 real HF families (Qwen/Alibaba, Phi/Microsoft, SmolLM2/HuggingFaceTB, Granite/IBM), each demonstrating a different annotator (`RegexAnnotator`, `LLMAnnotator`, `ThinkingStripAnnotator`, a fully custom `Annotator`), then all 4 compared together in one `compare_models()` call via its native `adapter=`/`annotators=`/`extract_with=` support |
| 08_compare_result_deep_dive.ipynb | `CompareResult` deep dive for what `compare_models()`'s native per-model support still can't express -- most notably different `scorers` per model, deliberately not overridable -- hand-building a `CompareResult` from independent `ak.evaluate()` calls, plus the full method surface (`per_metric()`, `winner()`, `pairwise()`, `coverage_warnings()`, `errors`, `performance_table()`/`size_table()`/`cost_table()`) |
| 09_guard_judge_implementation_check.ipynb | `GuardJudge` (`src/auditkit/metrics/guard.py`) implementation check across all 5 shipped profiles (`llama_guard`, `wildguard`, `harmbench`, `shield_gemma`, `granite_guardian`): **offline** (no model download) verification that each profile's real output-parsing shape (Llama Guard's `unsafe\nS1,S9`, WildGuard's 3 yes/no fields incl. the "refused, so safe even though the request was harmful" distinction, HarmBench's bare yes/no, ShieldGemma/Granite Guardian's policy-parameterized Yes/No) correctly becomes `(value, metadata)`, plus constructor validation (bad `assess=`, `policy=` misuse, malformed custom profiles). **Executed for real** against the one ungated profile, `ibm-granite/granite-guardian-3.1-2b` (Llama Guard/ShieldGemma/WildGuard all require accepting gated access first -- confirmed via `huggingface_hub.model_info` before writing this notebook) -- correctly scored a genuinely harmful prompt+response as unsafe (1.0) and a benign one as safe (0.0), both `assess=` modes, `ak.evaluate()` wiring, and confirmed `policy=` is fingerprint-sensitive. No GPU strictly needed (`device="cpu"`, slow but works for a 2B model) -- real harmful/benign separation confirmed on the live run: harmful=1.0, benign=0.0, in both `assess="response"`/`"prompt"` modes |
| 10_guard_judge_legit_models.ipynb | All 4 named, real guard models (`llama_guard`/`meta-llama/Llama-Guard-3-8B`, `wildguard`/`allenai/wildguard`, `shield_gemma`/`google/shieldgemma-2b`, `granite_guardian`/`ibm-granite/granite-guardian-3.1-2b`) scored together on the same real prompt/response pairs in one `evaluate()` call -- a genuinely harmful case, a genuinely benign case, and (the one that matters most) a harmful **request** met with a genuine **refusal**, which every guard must still score safe. Real weight sizes checked via `huggingface_hub.model_info` before writing this notebook: Llama Guard ~16GB, WildGuard ~29GB (stored fp32) -- both need a real GPU runtime, **Not currently pre-executed** (see note below) -- `granite_guardian`/`shield_gemma` (~5GB each) are CPU-feasible and ungated; `llama_guard`/`wildguard` need a GPU and accepted gated licenses |
| 11_encoder_judge_prebuilts.ipynb | The base `EncoderJudge` mechanism (real tokenizer pair-sequence encoding, one forward pass, softmax over the checkpoint's own labels, both aggregation modes) plus its 2 remaining prebuilt subclasses -- `FactualityEncoderJudge` (entailment/factual-consistency) and `SentimentEncoderJudge` (2-way sentiment) -- scored together in one `evaluate()` call, the auto-detection refusal on a generic-label checkpoint, and the worked example for handling one via explicit `label_map=` (BERT vs. ELECTRA's confirmed-opposite label orders). **Executed for real, every cell** (no GPU needed, largest checkpoint ~140M params) -- real discrimination confirmed: identical-text entailment 0.998 vs. unrelated-text 0.022, positive-sentiment 0.9999 vs. negative-sentiment 0.0004 |

Notebooks 02-08 and 10 are built and syntax-checked but **not
pre-executed** (09 and 11 *were* executed for real; 09 needs no GPU,
running the one ungated guard profile on CPU) -- run the rest yourself
(Colab GPU recommended for the local-HF ones, and required for 10's
`llama_guard`/`wildguard` sections specifically). 01 needs a GPU + an HF
token with the relevant Llama licenses accepted. Everything else runs
entirely on local HF models/CPU, no external API key needed.

**A note on 01/02/05/06/07's fixes**: all five previously had the same
underlying bug -- `QuasiExactMatch`/`Acc` require an *exact* match to a
bare answer/choice index, but the model's raw output is a full
multi-paragraph reasoning chain, so the metric scored 0.0 for every
sample regardless of whether the model actually got the answer right
(visible directly in the saved outputs before this fix: `quasi_exact_match`
flat at `0.0000`, zero variance, across every model in 01/02/05/06). This
looked like every compared model failing outright, when the real problem
was a missing extraction step -- 03 already demonstrated the fix
(`TemplateAdapter`/`RegexAnnotator`/`LLMAnnotator` + `extract_with=`); it's
now applied consistently across all of them. 06/07 additionally corrected
a stale claim that `compare_models()` has no `adapter=`/`annotators=`
support -- it does (`src/auditkit/model_compare.py:478`).

**Two more real bugs found after the first pass, both confirmed against
actual re-runs**:
1. `extract_with=` applies to *every* metric in the same call uniformly
   -- bundling `WordCount(min_words=10)` alongside `QuasiExactMatch()`
   under one `extract_with=` meant `WordCount` scored the tiny
   *extracted* number instead of the raw reasoning it's meant to
   sanity-check, always `<10` words, so it silently went flat `0.0` too.
   02 now uses a proper two-pass split (raw text scored in pass 1, the
   extracted answer re-scored in pass 2 via `model="precomputed"`, zero
   extra generation cost); 06 drops `WordCount` entirely since its focus
   is `compare_models()`/`RunComparison` mechanics, not scorer variety.
2. `max_tokens=300` genuinely wasn't enough headroom for `Qwen2.5-3B`'s
   real chain-of-thought style (heavy LaTeX-formatted step blocks) to
   both finish the reasoning *and* write the required "Final Answer:
   <number>" line -- confirmed by reproducing it directly: at 300 tokens
   the real generation cuts off mid-sentence right before the number
   (`...Therefore, Janet makes \$`), and the exact same prompt at 500
   tokens completes cleanly (`...Final Answer: 18`). Bumped to
   `max_tokens=500` in 02/06/07.
3. `RunConfig.track_performance` defaults to `False` (a deliberate
   reporting gate, not a bug -- most callers don't want the extra
   `model_info()` introspection cost). 01/02/06/07/08 all print/claim a
   "measured performance table" without ever passing
   `track_performance=True`, so `latency_mean_ms`/`throughput_rps`/etc.
   were structurally always `None` -- not missing data, just never
   turned on. Added `track_performance=True` to each notebook's
   `RunConfig` so the performance/size tables they already print show
   real numbers.

`auditkit.yaml` is a sample CLI config file (see `auditkit eval --config
auditkit.yaml`), not a notebook — kept alongside these for reference.

`model_comparison_notebook.py` is a jupytext-style (`# %%` cell) script for
comparing a baseline vs. a candidate model across native/lm-eval/judge
runs cell-by-cell in VS Code or Jupyter — open it directly rather than
running it as a plain script.
