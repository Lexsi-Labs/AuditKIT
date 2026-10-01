# Findings from the PR #1 review, and where each one stands

This suite found every issue below. They were reported on PR #1
([review comment](https://github.com/aryapratinavseth/AuditKIT-Internal/pull/1#issuecomment-5848428995)).
The tests here assert the behaviour as fixed on `feat/rag-agent-evals`.

## Fixed on `feat/rag-agent-evals` (`f6fd6ef`, `bab4ea3`, `e68e47e`, earlier)

| Item | Issue | How it was fixed | Covered here by |
|---|---|---|---|
| 1 | User sample ids not in the dataset hash: a cached run returned another dataset's `sample_id`s | User ids hashed; the Runner's auto `str(index)` excluded | `test_fingerprint_offline.py` (`user sample id`, `auto id`, `test_cached_run_never_reports_another_datasets_ids`, `test_id_less_samples_still_hit_the_cache_on_rerun`) |
| 2 | AuditKit version not in the fingerprint: stale scores after an upgrade | Version folded into the key (and into the lm-eval fingerprint) | `test_auditkit_version_is_part_of_identity` |
| 3 | Judge identity hashed raw args incl. `api_key`/`timeout` | `judge_identity_args()` strips transport and secret keys; `api_base` kept | `test_pairs_that_must_match[judge timeout/api_key]`, `[...server]` |
| 4 | `api:` sent `OPENAI_API_KEY` to any host | Only to OpenAI's host | `test_openai_key_only_goes_to_openai` (6 hosts) |
| 5 | Answer-only `agent:` reply scored as "made no calls" | Marked `tool_calls_unavailable` (including a post-loop chat-completion reply); tool metrics skip. An explicit `"tool_calls": []` or a transcript is observed-zero. | `test_tool_metrics_skip_runs_whose_tool_use_is_unobservable`, `test_agent_answer_only_reply_is_not_scored`, `test_agent_explicit_empty_tool_calls_scores_irrelevance`, `test_post_loop_chat_completion_reply_is_unobservable` |
| 6 | Unreadable `task_completion` verdict averaged in as 0 | Raises → recorded error; `agent_eval` still reports `unknown` | `test_task_completion_parse_failure_is_not_averaged_as_zero`, `test_task_completion_unreadable_verdict_raises` |
| 7 | Nested schema types unchecked | Recursive: array `items`, nested `properties`/`required`, closed objects (declared `properties` or `additionalProperties: false`) | `test_validity_checks_nested_types`, `test_validity_nested_cases`, `test_validity_nested_required_and_additional_properties` |
| 8 | `trajectory_in_order` = 1.0 on an irrelevance case with a call | 0.0 | `test_trajectory_match[irrelevance-call-made]` |
| 9 | `1` vs `1.0` not a repeat in `redundant_tool_calls` | One JSON-value-canonical call key (large ints guarded) | `test_redundant_uses_same_equality_as_matching` |
| 10 | Unasked optional args failed `exact` | Schema-aware `exact`: declared optional args tolerated, undeclared still fail | `test_exact_tolerates_declared_optional_args` and siblings |
| 11 | Items after a duplicate moved up a rank | A duplicate keeps its slot at zero relevance (it also counts as a precision slot) | `test_retrieval_metrics[duplicate-*]` |
| 12 | The answer's `<think>` judged as claims | Stripped before claim extraction | `test_faithfulness_does_not_judge_the_answers_own_reasoning` |
| 13 | Lexical RAG metrics ignored live retrieval | Read the trace's `retrieved_contexts` first | `test_lexical_metrics_read_live_retrieved_contexts`, `test_lexical_trace_wins_over_dataset_contexts`, `test_coverage_reads_live_contexts` |
| 14 | Lexical tokenization inconsistent / case-sensitive | One shared tokenizer | `test_lexical_metrics[*]` |
| — | Conflicting duplicate judge verdicts | Recorded error | `test_faithfulness_conflicting_duplicate_verdict_is_an_error` |
| — | One failing `api:` request aborted the run | Every failing request (4xx included) fails only its sample; retries follow `RunConfig.max_retries`; no batch replay | `test_api_backend_server_error_is_recorded_not_scored`, `test_api_backend_transient_error_recovers_on_retry`, `test_api_backend_client_error_is_recorded_per_sample` |

## Still open: fixed in PR #6

| Item | Issue | PR #6 |
|---|---|---|
| 15 (partial) | The 600 s judge timeout reached only the LLM judges; `faithfulness`, `context_precision` and `context_recall` kept 120 s | shared `judge_resolve_args()` helper + tests in `tests/test_rag_judge.py` |
| FP-3 | `task` / `kind` label cached predictions but aren't hashed: a cached run can show another dataset's task labels | hashed when non-default + tests in `tests/test_fingerprint_stability.py` |

The corresponding tests are in PR #6, not here, so this suite passes on `feat/rag-agent-evals` as it is.

## Observed in live runs (model behaviour, not code)

- **Optional arguments:** qwen3:8b and gemma3:12b add optional arguments unasked (`unit="celsius"`); item 10 handles this.
- **A fabricated tool result:** in the agent loop, qwen3:8b stated a local time it never fetched. `trajectory_strict` caught it; the `task_completion` judge said *partial*.
- **Judges disagree on generalised claims:** qwen3:8b rated "Laptops are replaced…" UNSUPPORTED against "Employee laptops are replaced…", while gemma3:12b rated it SUPPORTED.
