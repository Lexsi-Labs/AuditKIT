# Applications

Real-world scenarios built on AuditKit — not feature tours (see
[`examples/`](../examples/) for those), but a specific question answered
end to end with real data, real models, and a real verdict.

All 7 notebooks are capped at 20 real samples (01/03 currently run at a
larger sample count locally — see each notebook), use real, local,
**gated Llama checkpoints via HF** for generation and judging (a fast
model, a bigger one, and an independent third one reserved for judging
so it never grades its own output — needs an HF token with the relevant
licenses accepted, plus a GPU), and share the same shape: an adapter, an
annotator, real performance tracking
(`RunConfig(track_performance=True)`), and `ak.compare_models()`. Every
notebook carries **4 or more scorers, chosen for that specific task's
actual failure modes** — not a fixed bundle applied everywhere. In
particular, `GuardJudge` (always local `granite_guardian`, ungated) only
appears where the task has a real safety/harm axis (clinical advice,
real user reviews); `BertScore` only where the output is genuinely long
free text. See [`SUMMARY.md`](SUMMARY.md) for the full per-notebook
rationale (some details there may predate the Groq -> HF conversion).

| File | Description |
|------|-------------|
| 01_application_pruned_llama_boolq.ipynb | Does structural pruning (20%/40%/60% of MLP neurons, no quantization) degrade `meta-llama/Llama-3.2-3B` on real BoolQ? `boolq_answer_match` (custom) for correctness, `LexicalGroundedness` for grounding, `not_degenerate_output` (custom) for a real pruning-specific failure mode (collapsing into a word-repetition loop), and a real, local `LLMJudge` cross-check. No guard/BERT — single-word answers aren't meaningful for either. The 4 compared (pruned + baseline) models are real local HF regardless. `ak.compare_models()` across all 4, real perf/size metrics, ship/no-ship verdicts per severity. |
| 02_application_healthcare_pubmedqa.ipynb | Clinical QA on real `qiaojin/PubMedQA`: is the model's yes/no/maybe decision right (`decision_match`, custom) *and* is its reasoning grounded (`lexical_groundedness`) and safe (`guard_judge`, a real local `granite_guardian`) and semantically close to the real expert reasoning (`bert_score`)? Plus a real, local Llama `LLMJudge` opinion. Guard included deliberately: unsafe medical claims are a real risk here. |
| 03_application_finance_10k_qa.ipynb | Financial QA grounded in real SEC 10-K filing excerpts (`virattt/financial-qa-10K`): a real `RegexAnnotator` plus custom `NumericFactMatch` checks the *specific figure* stated, independent of `LexicalGroundedness`/`BertScore`/a real, local judge. No guard — a revenue figure has no safety/harm axis to screen for. |
| 04_application_ecommerce_review_triage.ipynb | Review-sentiment triage on real Amazon reviews (`mteb/amazon_polarity`), reframed to require label + a one-sentence reason (not a bare word) so guard/BERT scoring is meaningful: `sentiment_match` (custom) for raw correctness, `word_count` for a real reason being present, `guard_judge` for unsafe review content, `bert_score`/a real, local judge for reason quality. Guard included deliberately: real reviews can contain real toxic content. |
| 05_application_education_arc_tutor.ipynb | Auto-graded science tutoring on real `allenai/ai2_arc` (ARC-Easy): `Acc` (built-in) and `explicit_choice_match` (custom, via a separate `RegexAnnotator`) are two independent routes to the same correctness signal; `explanation_presence` (custom) and a real, local `LLMJudge` both check the tutor actually reasons, not just guesses. No guard/BERT — MCQ answers are a single choice index, not long free text, and there's no safety axis to a science question. |
| 06_application_enterprise_search_rag.ipynb | Internal knowledge-assistant RAG: real Wikipedia-derived Q&A (`rag-datasets/rag-mini-wikipedia`) retrieved via a real lexical retriever over a real 3.2k-passage corpus, scored with `LexicalGroundedness`/`bert_score`/`restated_answer_match` (custom, via a real *third*, independent local model restating each answer) plus a real, local `LLMJudge`. No guard — general trivia Q&A has no safety axis. |
| 07_application_bert_nli_judge.ipynb | LLM-as-judge via a real **BERT NLI classifier**, not a generative model: `BertNLIJudge` (`textattack/bert-base-uncased-MNLI`) scores entailment against the real reference. Part 1 isolates the judge mechanism on 24 precomputed correct/incorrect pairs (mean 0.917 vs 0.417 -- real separation); Part 2 adds a real, local generation pass through the full pipeline, paired with real lexical cross-checks (`quasi_exact_match`/`f1_score`) and a real, local `LLMJudge` instead of guard -- same reasoning as 06. |

All 7 notebooks are HF-first now — no Groq/external API key needed for
any of them, only an HF token (gated Llama licenses) and a GPU. 04 was
converted from an earlier all-Groq version to match the rest of this set
exactly (same 3-checkpoint pattern: `meta-llama/Llama-3.1-8B-Instruct` /
`meta-llama/Llama-2-13b-chat-hf` / `meta-llama/Meta-Llama-3-8B-Instruct`),
dry-run verified end to end with a mocked `HFGenModel` (real, unmocked
`GuardJudge`/`BertScore` computation against their real checkpoints).

**Every `LLMJudge` here uses a generous `max_tokens` budget (150–200),
not a small number** — a real reasoning-capable model can emit preamble
before its verdict line even when told to be brief; too tight a budget
truncates before the marker, which silently (but correctly) scores
`unknown_score` (0.0) for every sample. If you ever see a judge metric
flat at 0.0 with zero variance across a whole run, that's the tell — not
a real unanimous verdict.

**Not executed against real gated Llama checkpoints in this
environment** (04 was verified with a mocked `HFGenModel`; 01/02/03/05/06/07
are the user's own in-progress/local work and weren't independently
re-verified here) — run each notebook yourself with a GPU runtime and an
HF token with the relevant Llama licenses accepted.
