# Lexsi library integration

AuditKit reads evaluation evidence from the other Lexsi Labs libraries through a
**read-only importer**. It never imports those libraries (they pull torch,
transformers, numpy or pydantic); it reads an artifact each product already wrote
to disk, as plain JSON, and wraps it in a `SourceEvidence` envelope that records
provenance, the join keys the artifact actually carries, the join keys it is
**missing**, and a field coverage map. A producer's reward, verdict or score is
kept as **provenance about what that product measured**, never promoted to an
AuditKit verified outcome.

```python
from auditkit.agent_eval import (
    SourceEvidence, join_evidence, lineage_sidecar,
    episodes_from_agenttune,
    aligntune_audit_report, aligntune_regression_report,
    safetune_audit_event, safetune_bench_row,
    curatorkit_data_sample, curatorkit_manifest, curatorkit_trainer_export_row,
    circuitkit_faithfulness_report,
)
```

## What each library contributes

| Library | Importer | Granularity | What it carries |
|---|---|---|---|
| AgentTune | `episodes_from_agenttune` (run-eval rows, trajectories, reports, `trace.jsonl`, `EventLog`) | per case / per event | ordered events, per-call ids, retrieved contexts; coverage marks what is observed vs missing |
| AlignTune | `aligntune_audit_report`, `aligntune_regression_report` | aggregate | alignment scorecards and baseline-vs-variant regression verdicts |
| SafeTune | `safetune_audit_event`, `safetune_bench_row` | per event / per row | runtime policy decisions and scored benchmark rows |
| CuratorKIT | `curatorkit_data_sample`, `curatorkit_manifest`, `curatorkit_trainer_export_row` | per row / aggregate | dataset rows with provenance chains, run manifests, trainer-export rows |
| CircuitKIT | `circuitkit_faithfulness_report` | aggregate | mechanistic faithfulness of a discovered circuit (not a task or safety outcome) |

The fixtures behind the compatibility gate are produced by running each library's
own serializer at a pinned commit, so the importer is checked against real output
rather than a guess.

## Joining evidence to an episode

`join_evidence(episode, evidence, by)` attaches a piece of evidence to an
`AgentEpisode` **only through identity keys the caller supplies explicitly**. The
join is refused, with the missing key named, when:

- the `by` names no identity key (a timestamp, tenant, stage, format or source
  file never carries a join on its own);
- the evidence is **aggregate** and the key is per-case: a scorecard over many
  cases cannot be pinned to one case or call;
- per-event evidence does not name its `call_id`;
- a value contradicts the episode's own ids, or a list-valued artifact key does
  not match one record.

The central finding of the integration spike is that **no producer shares an
episode id today**, so every cross-product join needs ids captured at emission
time. `lineage_sidecar()` builds a `{row_id -> provenance}` map from CuratorKIT
`DataSample`, checkpoint or rejected rows, so a trainer export that dropped its
id (Alpaca or ShareGPT) can be re-joined by an id recorded at export time.

Nothing here changes another product's training loop or runtime policy: the
integration begins and stays with read-only files and sidecar reports.
