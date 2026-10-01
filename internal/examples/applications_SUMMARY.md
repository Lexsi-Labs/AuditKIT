# Applications — summary

Seven real-world scenarios built on AuditKit, rebuilt to a consistent
shape: **real datasets, capped at 20 samples each**, real Groq-hosted
models for generation and judging, and every notebook exercising an
adapter, an annotator, real performance tracking
(`RunConfig(track_performance=True)`), **4 or more genuinely different
scorers**, and `ak.compare_models()`.

**Every scorer is chosen per-notebook for what that task's actual
failure modes are — not a fixed bundle applied everywhere.** In
particular:

- **`GuardJudge` (safety) is only wired in where the task has a real
  safety/harm axis to screen for**: **02** (clinical advice — unsafe
  medical claims are a real risk) and **04** (e-commerce reviews — real
  toxic content is a real risk). It's deliberately **not** in 03
  (a financial figure has no safety dimension), 06/07 (general trivia
  Q&A has no safety dimension), or 01/05 (see below) — each of those
  gets a scorer that actually fits its task instead (see the table).
- **`BertScore` (semantic similarity) is only wired in where the output
  is genuinely long free text** — a guard model screening a single word,
  or `BertScore` comparing against a one-word reference (a bare MCQ
  letter, a yes/no label), is close to meaningless (a real finding
  confirmed earlier in this project's own testing). So: 01 (BoolQ,
  single-word yes/no) and 05 (ARC, a choice index) don't carry it —
  they get task-appropriate substitutes instead (see the table).

All Groq calls use three real, distinct models throughout: two models
under comparison (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) plus
a third, independent model reserved for judging (`openai/gpt-oss-20b`) so
a judge never grades its own output. Guard scoring is the one exception
to all-Groq: it uses a real **local** HF model (`granite_guardian`,
`ibm-granite/granite-guardian-3.1-2b`, ~2B params, CPU-feasible, ungated)
instead, since Groq serves `llama-guard-3-8b` through a "text
classification" endpoint that rejects the standard user+assistant
message pair `GuardJudge` sends (confirmed live: a 400 "messages must
contains a single user message for text classification models" —
`GuardJudge` now also auto-recovers from this generically via a
`single_message=` mechanism, not hardcoded to Groq/this model).

**`LLMJudge` token budget:** every Groq `LLMJudge` here uses
`max_tokens=200`, not a smaller number — `openai/gpt-oss-20b` is a
reasoning-style model that often emits preamble before its `CHOICE:`
line even when told to be brief. A tighter budget truncates the reply
before the marker, which `LLMJudge` correctly reports as `unknown_score`
(default `0.0`) rather than guessing — but if every sample truncates the
same way, that shows up as a suspicious *flat* 0.0 with zero variance
across an entire run. That's the tell it's a parsing/truncation issue,
not a real unanimous verdict (confirmed directly: a synthetic 16-token
truncated reply parses to `unknown=True`, the same reply with room to
finish parses correctly).

## Per-notebook

| # | Application | Dataset (real) | Samples | Scorers (4+, task-fitted) | Notes |
|---|---|---|---|---|---|
| 01 | Pruning degradation | `google/boolq` | 20 | `boolq_answer_match` (custom), `faithfulness`, `not_degenerate_output` (custom), Groq `LLMJudge` | No guard/BERT — single-word answers aren't meaningful for either. Instead: `not_degenerate_output` catches a real, well-known pruning-specific failure mode (collapsing into an immediate word-repetition loop) — chosen because it fits *this* study, not as a generic substitute. The one exception to "all Groq" for models under test: pruned community checkpoints are inherently local, so the 4 compared models stay local HF; only the judge is Groq. `ak.compare_models()` takes pre-built `Model` instances directly. |
| 02 | Clinical QA | `qiaojin/PubMedQA` | 20 | `decision_match` (custom), `faithfulness`, `guard_judge`, `bert_score`, Groq `LLMJudge` | Guard included: unsafe medical claims are a real risk for a clinical assistant. `target` uses the dataset's real `long_answer` (expert reasoning), not the bare `yes/no/maybe` label, so `BertScore` has real sentence-level text to compare. |
| 03 | Financial QA | `virattt/financial-qa-10K` | 20 | `numeric_fact_match` (custom), `faithfulness`, `bert_score`, Groq `LLMJudge` | No guard — a revenue figure has no safety/harm axis to screen for. `numeric_fact_match` checks the *specific figure* stated, independent of whole-sentence phrasing. |
| 04 | Review triage | `mteb/amazon_polarity` | 20 | `sentiment_match` (custom), `word_count`, `guard_judge`, `bert_score`, Groq `LLMJudge` | Guard included: real reviews can contain real toxic content. Reframed from bare-label classification to "label + one-sentence reason" specifically so the output is long enough for guard/BERT to be meaningful; `target` is a synthesized reference sentence, not the bare label. |
| 05 | Science tutoring | `allenai/ai2_arc` (ARC-Easy) | 20 | `acc`, `explanation_presence` (custom), `explicit_choice_match` (custom), Groq `LLMJudge` | No guard/BERT — MCQ answers are a single choice index, not long free text, and there's no safety axis to a science question. `acc` (built-in, parses free text) and `explicit_choice_match` (custom, via a separate `RegexAnnotator`) are two independent routes to the same correctness signal — a real cross-check on each other. |
| 06 | Enterprise RAG | `rag-datasets/rag-mini-wikipedia` | 20 | `faithfulness`, `bert_score`, `restated_answer_match` (custom), Groq `LLMJudge` | No guard — general trivia Q&A has no safety axis. Real lexical retrieval over the real 3.2k-passage corpus (no vector DB, just a real word-overlap ranker) feeds `retrieval_context`; `restated_answer_match` uses a real *third* Groq model (`LLMAnnotator`) restating each answer as the strictest correctness check. |
| 07 | BERT-NLI judge | `rag-datasets/rag-mini-wikipedia` | 12 pairs (24 samples) + 15 new | `bert_nli_judge` (custom classifier judge), `quasi_exact_match`, `f1_score`, Groq `LLMJudge` | No guard — same reasoning as 06. Two parts: a precomputed correct/incorrect demo isolating the judge mechanism itself (no generation), plus a real Groq generation pass through the full pipeline, paired with real lexical cross-checks (`quasi_exact_match`/`f1_score`) instead — the same judge metric works unchanged in both parts. |

## What's genuinely verified vs. what needs a real run

Every notebook was **dry-run executed end to end** with a mocked Groq
backend (no real network calls) and, for 01, a mocked local model
(matches how every other GPU-only notebook in this repo is verified) —
`compare_models()`, every adapter/annotator/metric wiring and
`RunConfig(track_performance=True)` all confirmed to produce real,
non-crashing output. `BertScore`, `BertNLIJudge`, and `GuardJudge`
(`granite_guardian`) all ran for **real, unmocked** local model calls
against their real checkpoints (`microsoft/deberta-xlarge-mnli`,
`textattack/bert-base-uncased-MNLI`, `ibm-granite/granite-guardian-3.1-2b`)
— including a real, meaningful separation on 07's known correct/incorrect
pairs (mean `bert_nli_judge` 0.917 vs 0.417) and a real harmful=1.0/
benign=0.0 discrimination from `granite_guardian`. Only the Groq
`LLMJudge`/generation calls are mocked in this dry run (with canned
short replies, so the dry run can't reproduce the truncation issue above
— that was confirmed separately with a direct, real `_parse()` test).

**Not executed against the live Groq API** — that needs a real
`GROQ_API_KEY` and will make real billed calls (each notebook makes
roughly `20 samples × 2 compared models × (1 generation + 1 LLM-judge
call)` ≈ 40–60 real Groq requests; `guard_judge` runs locally against
`granite_guardian`, not against Groq, so it doesn't add to that count).
Run each notebook yourself with a real key to get real generated text
and real judge verdicts (guard verdicts are already real on any run,
local or not); the `retention()`/`grade()` numbers from the dry run are
artifacts of the mock returning the same canned answer for both compared
models and should be ignored — they'll be real once run live.

01 additionally needs a GPU runtime and an HF token with the
`meta-llama/Llama-3.2-3B` license accepted (gated model).
