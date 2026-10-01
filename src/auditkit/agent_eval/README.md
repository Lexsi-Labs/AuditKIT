# Agent evaluation

End-to-end evaluation of what an agent did and achieved across a complete task
episode. The headline is a verified outcome (final state, artifact, answer, or a
custom predicate); tool, trace and judge scores are separate diagnostics.
Stdlib only -- AgentTune, torch and transformers are never imported here.

- **`types.py`**: `AgentCase` (task, tools, budgets, oracle, stable digest),
  `AgentEvent` (one ordered event), `AgentEpisode` (one recorded/deployed run,
  ordered events, final answer/state/artifacts, loss-aware `coverage`),
  `validate_cases`.
- **`importers.py`**: recorded runs to episodes: `episodes_from_agenttune`,
  `episode_from_eventlog` (AgentTune `EventLog` object or JSON, duck-typed),
  `episode_from_openai_messages`, `episode_from_sample`, `inspect_agenttune`,
  and `supported_scorers_from_coverage`. Coverage is set honestly per source.
- **`outcome.py`**: outcome oracles `FinalStateAssertion`, `ArtifactAssertion`,
  `AnswerAssertion`, `CustomPredicate` and the `Outcome` verdicts
  (`success`/`failure`/`unknown`/`error`); `resolve_outcome` and the diagnostic
  `judge_outcome`. A judge never overrides a verified state assertion.
- **`runner.py`**: `AgentEvalSpec`, `AgentEvalRunner` (recorded and deployed
  modes, per-case status, scorer eligibility), and `rescore` (score saved
  episodes with no agent call).
- **`report.py`**: `AgentCaseResult` (per-case row) and `AgentEvalResult`
  (the full run plus a both-denominators summary), with JSON round-trip.
- **`lexsi.py`**: read-only evidence importer for sibling Lexsi Labs products
  (AgentTune, AlignTune, SafeTune, CuratorKIT, CircuitKIT). Readers wrap an
  exported artifact in a `SourceEvidence` envelope with provenance and missing
  join keys; `join_evidence` attaches it to an episode only on explicit ids;
  `lineage_sidecar` summarizes sample lineage. No producer library is imported.

See [`docs/agent_eval.md`](../../../docs/agent_eval.md) for the guide and
[`examples/agent_eval_offline.py`](../../../examples/agent_eval_offline.py) for
a runnable offline example.
