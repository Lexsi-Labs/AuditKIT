# Pipeline suite: independent tests for PRs #1–#4, plus edge cases for every new metric

Offline tests for the evaluation features in the open PRs. Each expected number is computed by hand in a
comment next to the assertion. Judges are scripted and models are stubs; there are no network or GPU calls.

| File | Covers | Runs when |
|---|---|---|
| `test_pr1_additions.py` | PR #1 additions not covered by `tests/agentic_suite`: `agent_loop_detection`, `tool_permission`, `tool_selection`, `hallucination`, `answer_relevancy`, `response_groundedness`, `context_relevance`, the RAG-stress metrics, `agent_eval` reliability (A3), the harness loop (A4) and the sidecar (A5) | always, on `feat/rag-agent-evals` |
| `test_pr2_redteam.py` | PR #2's attack suite: ASR with the unknown and judge-error accounting, target errors, query budgets, method labels, determinism, the manifest, `RefusalJudge` | once `AttackSuite` exists (#2 merged) |
| `test_pr3_mrm.py` | PR #3's banking model-risk code: ECE, Brier, log loss, dated backtest, paired control, the action oracle (pass / blocked / fail / unknown / not_tested), idempotency, the recipient bypass, maker-checker, the benign/attack pair | once `auditkit.mrm` exists (#3 merged) |
| `test_pr4_uncertainty.py` | PR #4's uncertainty code: the UQ importer, ECE, false-pass rate, review capacity, selective risk (AURC), claims, the code-execution checks | once `auditkit.uncertainty` exists (#4 merged) |
| `test_edge_rag_stress.py` | Edge cases for all seven RAG-stress metrics: alternate and partial evidence sets, changed or missing pinned digests, supersession on the decision date, fail-closed ACLs, spans that fall between declared spans, wrong versions, touching-but-not-overlapping citations, abstention signal precedence, number parsing, context dropped after retrieval, and every `not_tested`/`unknown` path | always |
| `test_edge_agent_and_rag_judges.py` | Edge cases for `agent_loop_detection` (threshold vs cycle, argument canonicalisation, stagnation, malformed calls), `tool_permission` (schema formats, empty allowlist, constructor vs sample denylist), `tool_selection` (per-call history, partial unreadable verdicts), the shared verdict parser, and the four reference-free RAG judges (unreadable replies, `NONE`, `<think>` stripping, skips that never call the judge) | always |
| `test_edge_mrm.py` | Edge cases for calibration, Brier, log loss, dated backtest (maturity boundary, bands, buckets), paired control, `missing_data_sensitivity`, the action oracle (limit boundary, maker-checker, idempotency, unexplained diffs, utility), and the tampering track (`tamper_delta`, lineage identities, budget summary, report fingerprint) | once `auditkit.mrm` exists (#3 merged) |
| `test_edge_uncertainty.py` | Edge cases for the UQLM importer (labels, confidence problems, orientation), the four reports at their boundaries, claims, and code/SQL execution (guard, timeout, read-only and single-statement SQL, equivalence vs correctness, `code_confidence_vs_correctness`) | once `auditkit.uncertainty` exists (#4 merged) |
| `test_edge_redteam_judges.py` | Edge cases for `ModelJudge` (rubric and HarmBench), `StrongRejectJudge` (formula, injected blocks, unparseable fields, `mean_score`), `GuardBackedJudge`, `DetectorJudge`, `RefusalJudge` and `validate_judge` (confusion counts, kappa, unknown and crashing rows) | once the #2 judges exist (#2 merged) |

The three files for #2, #3 and #4 **skip** on branches that don't have those features, so the suite can
land on any branch.

```bash
python -m pytest -o addopts= tests/pipeline_suite -q -rs -rx
```

## Findings fixed on branch `fix/unknown-scores-and-review-gaps` (non-strict `xfail`)

These assert the correct behaviour. They are XFAIL on the base branch and XPASS once the fix branch is
merged. They are non-strict so this suite passes either way; remove the markers after the fix lands.

| Test | Finding |
|---|---|
| `test_pr1_additions.py::test_tool_selection_unreadable_verdict_is_an_error_not_a_zero` | An unreadable `tool_selection` verdict scores 0.0 and is averaged into the headline with no error. `task_completion` was already fixed for this. |
| `test_pr1_additions.py::test_not_tested_results_are_not_averaged_into_the_headline` | A `not_tested`/`unknown` RAG-stress result is a Score with a placeholder 0.0 that the Runner averages: 1.0 plus not_tested gives 0.5. |
| `test_pr3_mrm.py::test_unknown_brier_score_is_not_averaged_into_the_headline` | The same placeholder in `mrm`: an unparseable output makes Brier look *better* (0.81 plus unknown gives 0.405). |

All three share one root cause: the Runner aggregates Scores flagged `metadata["unknown"]`.

## New findings from the edge-case files (strict `xfail`, not fixed by any open PR)

Each one fails today. When a fix lands the test XPASSes, the suite fails, and that is the prompt to remove the marker.

| Test | Finding |
|---|---|
| `test_edge_rag_stress.py::test_a_document_not_yet_in_force_is_not_current` | `freshness` only checks `superseded_date`. A document whose `effective_date` is after the decision date is scored as current. |
| `test_edge_rag_stress.py::test_a_real_answer_that_mentions_a_refusal_phrase_is_not_an_abstention` | The `abstention` keyword fallback reads any answer containing "not able to", "no information" and similar phrases as a refusal. |
| `test_edge_rag_stress.py::test_a_numeric_range_matches_its_spelled_out_form` | `numeric_accuracy` parses "5-10" as 5 and -10, so the correct claim "between 5 and 10" fails. |
| `test_edge_agent_and_rag_judges.py::test_progress_messages_about_different_entities_are_not_stagnation` | Loop stagnation uses raw `difflib`: "…weather in Paris." and "…weather in Rome." (ratio 0.87) are flagged at the default 0.85. |
| `test_edge_mrm.py::test_a_nan_probability_is_dropped_not_propagated` | `_to_float` accepts `"nan"`, so Brier (and `BrierScoreMetric`'s headline) becomes NaN with status `ok` and no error. |
| `test_edge_mrm.py::test_an_out_of_range_probability_is_rejected` | Probabilities outside [0, 1] and labels other than 0/1 are scored silently (`tamper_delta` rejects them; the probability metrics do not). |
| `test_edge_mrm.py::test_a_missing_score_is_skipped_not_a_crash` | `paired_control` crashes with `TypeError` on a missing champion/challenger score (missing labels are skipped). |
| `test_edge_mrm.py::test_a_negative_amount_is_a_violation` | The action oracle never checks the sign of `amount_cents`: a negative transfer passes the limit and moves money the other way. |
