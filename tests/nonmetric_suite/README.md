# Non-metric suite: everything else PRs #1–#4 added

These are offline tests for the parts of #1–#4 that are **not** metrics: evidence and review workflows, red-team machinery, redaction, oracles, loaders and version checks. The metrics themselves are covered by `tests/agentic_suite` and `tests/pipeline_suite` (#12).

Targets, attackers and judges are scripted. There are no network or GPU calls.

| File | Covers | Runs when |
|---|---|---|
| `test_agent_eval_and_loaders.py` | #1: secret redaction (`redact`, `detect_secrets`), the outcome oracles (state, artifact, answer, custom predicate, assertion and its state override, `from_spec`), `AgentCase` validation and digest, episodes from OpenAI messages, `load_jsonl`, `load_agenttune` (dataset, trace and trajectory records), and the compat version checker | always |
| `test_redteam_machinery.py` | #2: case loading and OWASP tags, the metered target, budgets for all 12 query-only methods, seeded reproducibility, `Transform`, injection surfaces and applicability, `dry_run`, sessions, oracles, `SessionRunner`, finding → regression → replay, `paired_run` + `ci_report`, and the 4 scenario packs | once `AttackSuite` exists (#2 merged) |
| `test_mrm_evidence.py` | #3: `ModelUse`/`ValidationPlan`/`EvidenceBundle`, jurisdiction views and the perimeter report, findings, reviews, regression freeze/replay, change triggers, the dossier export, and the fixtures | once `auditkit.mrm` exists (#3 merged) |

PR #4 adds no non-metric code beyond its UQ importer, which `tests/pipeline_suite` covers.

```bash
python -m pytest -o addopts= tests/nonmetric_suite -q -rx
```

## Findings (strict `xfail`)

Each test asserts the correct behaviour and fails today. When a fix lands, the test XPASSes, the suite fails, and that is the prompt to remove the marker.

| PR | Test | Finding |
|---|---|---|
| #2 | `test_redteam_machinery.py::test_a_crashing_target_is_never_fixed` | `replay` ignores the episode status. A target that raises leaves an empty trace and unchanged state, the oracle reads that as success, and the regression is reported **fixed**. |
| #2 | `test_redteam_machinery.py::test_a_down_variant_fails_the_ci_gate` | Same root cause: with the variant endpoint down, `paired_run` says **fixed** with complete evidence and `ci_report` returns **PASS**. |
| #2 | `test_redteam_machinery.py::test_attacker_budget_exhaustion_is_budget_exhausted` | Running out of the attacker query budget is recorded as `method_error` / `error`, not `budget_exhausted`. |
| #3 | `test_mrm_evidence.py::test_replay_does_not_call_a_worse_safety_drop_fixed` | `review.replay` assumes lower is better. A `tamper_delta` safety drop that got six times worse (−0.1 → −0.6) is reported **fixed**. |
| #3 | `test_mrm_evidence.py::test_duplicate_finding_ids_are_rejected` | `export_evidence` accepts duplicate finding ids, and the dossier fingerprint then depends on input order. |
| #3 | `test_mrm_evidence.py::test_a_frozen_bundle_cannot_change_under_its_fingerprint` | `EvidenceBundle` is "immutable", but its `provenance` dict can be edited after creation, which silently changes the fingerprint. |
| #1 | `test_agent_eval_and_loaders.py::test_a_secret_used_as_a_key_is_redacted` | Auto redaction scans values only. A secret used as a dict key is neither redacted nor detected. |
| #1 | `test_agent_eval_and_loaders.py::test_an_int_under_a_credential_like_key_is_left_alone` | The code comment says ints under credential-like keys are kept, but they are redacted (`password_min_length: 12`). |
| #1 | `test_agent_eval_and_loaders.py::test_banking_prose_about_bearer_instruments_is_kept` | The bearer-token pattern redacts "bearer instruments" in ordinary banking text. |
| #1 | `test_agent_eval_and_loaders.py::test_a_thousands_separator_does_not_break_a_normalized_match` | A normalized answer match maps `,` to a space, so "1,000" never matches "1000". |
| #1 | `test_agent_eval_and_loaders.py::test_contains_does_not_match_inside_a_longer_number` | `contains` is a raw substring test: the reference "5" matches "The fee is 15%". |
| #1 | `test_agent_eval_and_loaders.py::test_a_criterion_read_from_metadata_is_part_of_the_digest` | `AgentCase.digest()` leaves out `metadata`, but the answer and assertion oracles read their reference from it. Opposite criteria share one digest and one run identity. |
