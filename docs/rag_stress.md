# RAG stress testing (deterministic, model-free)

Deterministic RAG-stress checks for banking model-risk validation. They score a
**versioned corpus, question, retrieval trace, and answer** against a gold
oracle — no LLM judge — and locate *where* a failure first occurred. A high
faithfulness score can still coexist with a wrong answer if the retrieved
document was stale, incomplete, unauthorized, or misattributed; each of those is
a separate, deterministic check here.

This is build-order steps (1) and (2) of the RAG-stress PRD
(`docs/notes/rag-stress-model-risk-research-prd-2026-09.md`): the trace/ID +
corpus-manifest contract, and the evidence-set, freshness, ACL and citation
checks. Judge calibration, stress-fixture generation and load/chaos are later
steps.

## The contract (`auditkit.metrics.rag_stress`)

Three stdlib dataclasses, all with strict-JSON `to_dict()` / `from_dict()`:

- **`CorpusSnapshot`** — the frozen, replayable manifest. Per document: a content
  `digest` (sha256), `version`, `effective_date`, `superseded_date`, an `acl`
  (list of permitted tenants/identities), and a `spans` page/span map. Plus
  `parser_version` / `chunker_version` / `embedding_version` / `index_version` /
  `reranker_version` and `ingested_at`. It stores **digests, not raw text**
  (redaction seam). `changed(other)` returns doc ids whose digest differs;
  `invalidated(case)` returns doc ids whose pinned evidence digest no longer
  matches (RST-01: a changed source invalidates its evidence view).
- **`RAGCase`** — one question and its gold: `answerability`
  (`answerable`/`unanswerable`), `expected_action` (`answer`/`abstain`),
  `sufficient_evidence_sets` (a list of id sets — any one *complete* set is
  enough, so a valid alternate source passes), `evidence_digests` (the snapshot
  the gold was authored against), `gold_citations`, `prohibited_sources`,
  `tenant`/`identity`, `decision_date`, `locale`, `risk_tier`.
- **`RAGTrace`** — what the pipeline did: `query` + `rewrites`, ranked
  `retrieved` (`[{id, score, digest?, version?}]`), `final_context` offsets,
  `cited_spans`, `answer_claims`, `abstained`, and a per-event `coverage` label.
  A `None` field means the event was never captured (`missing`); `[]` means
  captured-but-empty (`observed`). `to_dict()` also emits `retrieved_contexts`
  (ranked ids) so the existing `retrieval` metric runs on the same trace.

The metrics read the case and snapshot from `sample.metadata["rag_case"]` and
`sample.metadata["corpus"]` (a dataclass or its dict), and the trace from
`context["trace"]` — the Runner copies `Sample.actual_trace` there — or
`sample.actual_trace`.

## Metrics

| Name (`ak.evaluate(scorers=...)`) | Needs | Reports |
| --- | --- | --- |
| `evidence_set_recall` (RST-02) | case `sufficient_evidence_sets`, trace `retrieved` | 1.0 iff one **complete** set is retrieved; ranking one chunk first is not enough; an alternate set passes |
| `freshness` (RST-01/03) | case `decision_date`, corpus dates/versions, trace `retrieved` | `freshness` (fraction current) + `stale_hit_rate`; stale iff `superseded_date <= decision_date` or trace digest ≠ snapshot |
| `acl_compliance` (RST-05) | case `tenant`/`identity`, corpus `acl`, trace `retrieved` | `acl_compliance` (fraction authorized) + `unauthorized_hit_rate`; prohibited / cross-tenant / not-in-snapshot hits fail |
| `citation_support` (RST-06) | case `gold_citations`, corpus `spans`/`version`, trace `cited_spans` | `citation_support`/`citation_precision`/`citation_recall` + `wrong_version_rate`; a nonexistent or superseded-version span fails |
| `abstention` (RST-04) | case `answerability`/`expected_action`, trace `abstained`/`answer_claims` or output | 1.0 for correct abstain / correct answer; 0.0 for answering an unanswerable case or false-refusing an answerable one |

Every metric returns a score in `[0, 1]` **or** an explicit `unknown` /
`not_tested` result (`label` set, `metadata["unknown"] = True`) when the required
gold or trace is missing — never a silent `0`. Decisions worth knowing:

- **acl**: a retrieved doc *absent from the snapshot* is a violation
  (fail-closed); a doc present but with no `acl` label is untestable.
- **freshness**: only docs carrying a version/date are testable; dates parse
  with strict `date.fromisoformat`.
- **abstention**: the oracle is `expected_action` (derived from `answerability`
  if unset). Abstention is read from `trace.abstained`, else empty
  `answer_claims`, else a naive keyword scan of the output (an LLM judge
  replaces the keyword fallback in build step 4).
- **citation**: existence = the cited offsets fall inside a declared source
  span; a gold match = same-doc overlap.

### Aggregation caveat

`Runner.aggregate` averages every score's `value` and does **not** yet exclude
`unknown`-flagged scores (it has no hook to, and the runner is out of scope for
this step). Read the per-score `label` / `metadata["unknown"]` rather than
trusting the mean when any sample is `unknown`/`not_tested`. A first-class
`not_tested` run status (the PRD's `StressResult`) arrives with the paired
runner below (build-order step (3), `auditkit.metrics.rag_stress_runner`).

## Example

```python
import auditkit as ak
from auditkit.metrics.rag_stress import RAGTrace, synthetic_bank_kb

snapshot, cases = synthetic_bank_kb()   # tiny benign synthetic bank KB
case = cases[0]                          # a multi-hop LTV + Q1-NII question

trace = RAGTrace(query=case.question, retrieved=[
    {"id": "LP-CURRENT", "score": 0.9}, {"id": "FIN-TABLE", "score": 0.8}])

sample = ak.Sample(
    input=case.question,
    actual_output="LTV cap is 80 percent; Q1 NII 1200.",
    actual_trace=trace.to_dict(),
    metadata={"rag_case": case.to_dict(), "corpus": snapshot.to_dict()})

result = ak.evaluate(dataset=[sample], model="precomputed",
                     scorers=["evidence_set_recall", "freshness", "acl_compliance"])
print(result.stats["evidence_set_recall"].mean)  # 1.0 — a complete set retrieved
```

`synthetic_bank_kb()` returns a current + superseded lending policy, a financial
table/footnote doc, a private-tenant doc, and a poisonable (prohibited) doc —
enough to prove each check locates its failure independently.

## Stress-fixture generator and paired runner (step 3)

`auditkit.metrics.rag_stress_runner` builds benign synthetic stress scenarios
and scores the deterministic metrics above on a **stress (attack) arm** and a
**benign control arm** of each case, then reports *where* the first failure
occurred and whether the control still passed — never collapsing the metrics
into one index.

```python
from auditkit.metrics.rag_stress_runner import generate_stress_cases, run_stress

cases = generate_stress_cases(seed=0)          # 22 paired cases, reproducible under the seed
report = run_stress(cases)                      # scores both arms of every case
d = report.to_dict()                            # strict JSON: matrix + coverage, no single index
print(d["matrix"]["poisoned_doc"]["acl_compliance"])   # {'flagged': 2, 'pass': 0, ...}
print(d["benign_utility_loss"])                 # case ids where the control arm failed anyway
```

`generate_stress_cases(seed=0, kinds=None, repeats=2)` returns a list of
`StressCase` (each iterable as the documented `(case, stress_trace, snapshot)`
triple), carrying a `stress_kind` tag and the `expected_failure` metric that
*should* flag it. `repeats` re-emits each scenario with seed-driven benign
variation (a rotated safe distractor, jittered retrieval scores) so the default
run is >= 20 cases; the structural failure is identical across repeats.

### Stress kinds and the metric that flags each

| `stress_kind` | Benign scenario | `expected_failure` | First-failure stage |
| --- | --- | --- | --- |
| `stale_policy` | superseded policy retrieved beside a valid alternate set | `freshness` | retrieval |
| `conflicting_policies` | current + superseded policy retrieved together | `freshness` | retrieval |
| `missing_evidence` | unanswerable question; stress arm answers anyway | `abstention` | answer |
| `false_refusal` | answerable question; stress arm refuses | `abstention` | answer |
| `financial_table_citation` | right number, citation to a nonexistent span | `citation_support` | citation |
| `financial_numeric` | wrong number, correct citations | `numeric_accuracy` | answer |
| `long_context` | both docs retrieved, one dropped from final context | `context_retention` | context |
| `multilingual` | synthetic non-English query misses the gold set | `evidence_set_recall` | retrieval |
| `poisoned_doc` | a prohibited (poisonable) source is retrieved | `acl_compliance` | retrieval |
| `cross_tenant` | a private-only doc retrieved for a retail identity | `acl_compliance` | retrieval |
| `partial_retrieval` | operational failure: the retrieval event never fired | *(none)* | — (`not_tested`) |

Each scenario is built so that **only** its `expected_failure` metric flags on
the stress arm — every other metric passes or is `unknown`/`not_tested` — so the
matrix locates each failure independently (RST-02..06). Two locators live in the
runner, not in `rag_stress.py`, so the report can separate an answer-stage
numeric error from a citation error and a context-assembly drop from a retrieval
miss:

- `numeric_accuracy` — gold-answer numbers ⊆ answer-claim numbers (naive; a
  unit/currency-aware judge is build step 4).
- `context_retention` — a complete sufficient evidence set survives into
  `trace.final_context` (the `context` stage the retrieval metrics do not read).

### The report

`StressReport.to_dict()` is strict JSON (round-trips with `allow_nan=False`):

- `cases` — per case: the `stress` and `control` metric results (each with
  `value`, `label`, `status`, `flagged`, `reason`), the `first_failure_stage`,
  the `stress_flagged` metric names, and `benign_control_passed`.
- `matrix` — `kind -> metric -> {flagged, pass, unknown, not_tested}` counts on
  the stress arm. Never a product of scores.
- `coverage` — per metric across all stress arms: `scored` (= `flagged` +
  `pass`), `unknown` and `not_tested` kept **separate** (a `not_tested` case is
  counted, never scored 0).
- `benign_utility_loss` — case ids whose benign control arm failed a metric,
  reported apart from detected stress failures.

The `ingestion` stage is declared in `STAGE_ORDER` but has no deterministic
metric yet: a changed source digest surfaces as an `evidence_set_recall`
`unknown` (RST-01), not a stage failure. Judge calibration, human review and
load/chaos are the next section (build steps 4-5).

## Judge calibration, human review, and operational checks (steps 4-5)

`auditkit.metrics.rag_stress_ops` adds three stdlib, model-free pieces that
consume **supplied data** — an adjudicated label set, a `StressReport`, or
scripted timing/error samples — and follow the same `unknown` convention (value
`None`, never a silent `0`) as the metrics above.

### `judge_calibration(labeled_rows)` — test the judge against a local set (RST-08)

RAGBench/TRACe and ARES both warn that an automated RAG judge can disagree with
humans substantially, so the judge is calibrated against a small
human-adjudicated set before it is trusted — it is not itself ground truth.

Each row is `{"human": <verdict>, "judge": <verdict>, "dimension"?: str}` where a
verdict is a bool or `"pass"`/`"fail"` (case-insensitive; anything else raises —
a verdict is a trust boundary). `pass` is the positive class. Returns a
strict-JSON dict:

| Field | Meaning |
| --- | --- |
| `agreement` | observed agreement (fraction where judge == human) |
| `cohens_kappa` | chance-corrected agreement; `None` when chance agreement is 1.0 (a degenerate all-one-class set) |
| `false_pass_rate` | P(judge=pass \| human=fail) — the judge lets a human-failed answer through (the model-risk-dangerous miss); `None` when there are no human-fail rows |
| `false_negative_rate` | P(judge=fail \| human=pass) — the judge rejects a human-passed answer (a false alarm; the PRD's false-fail); `None` when there are no human-pass rows |
| `confusion`, `n`, `n_human_pass`, `n_human_fail` | the 2x2 counts and denominators (the PRD asks to record denominators; a rate is uninterpretable without them) |
| `by_dimension` | the same block per `dimension` when rows carry it — RST-08's per-slice agreement for routing low-agreement slices to review |

Rows missing a `human` label are not counted (`n_unlabeled`); rows with a human
label but no `judge` verdict are counted separately (`n_missing_judge`) and left
out of the confusion. When no row carries both, the result is
`status: "unknown"` with every rate `None`.

### `HumanReview` + `review_queue(stress_report, *, budget)` — no auto-resolution

`review_queue` surfaces, within `budget`, the `StressReport` cases a reviewer
should look at. A `StressReport` carries no per-case confidence number, so the
proxy for "lowest-confidence" is a genuinely unresolvable score. A case is a
candidate when any signal fires:

- **disagreement** — the harness disagrees with the injected expectation (the
  `expected_failure` metric did not flag, or something unexpected did);
- **unknown** — a metric with status `unknown` (gold present, unresolvable);
- **nothing scored** — no metric scored at all (the operational /
  partial-retrieval shape: retrieval never fired);
- **utility loss** — the benign control arm failed a metric.

`not_tested` (absent gold for a metric a case does not target) is deliberately
**not** a signal — otherwise every fixture case would queue and the budget would
be meaningless. Candidates are ordered by priority (disagreement > nothing
scored > unknown > utility loss), ties broken by `case_id`.

Each returned `HumanReview` carries the auto fields plus reviewer fields
(`detected` / `correction` / `override` / `final_answer_changed`) that default to
`None`. The queue **records** work; it never resolves it — the reviewer fills the
decision via `hr.record(detected=..., override=..., final_answer_changed=...)`.

```python
from auditkit.metrics.rag_stress_ops import review_queue
from auditkit.metrics.rag_stress_runner import generate_stress_cases, run_stress

report = run_stress(generate_stress_cases(seed=0))
for hr in review_queue(report, budget=5):
    print(hr.case_id, hr.queued_reason)          # partial-* (nothing scored), missing-* (acl unknown)
    hr.record(detected=True, override="fail", final_answer_changed=True)
```

### `operational_stress(scenarios)` — load / chaos, deterministic from samples (RST-07)

A deterministic replay of scripted `concurrency` / `burst` / `slow_store` /
`timeout` / `partial_search_failure` / `rate_limit` / `judge_failure` scenarios
(`OPS_KINDS`). **Nothing is timed, slept, or threaded** — every number is
computed from the supplied samples. Each scenario is a dict with `name`, `kind`,
and any of:

- `latencies_ms` — supplied timing samples → `p50_ms` / `p95_ms` (reusing
  `Stat.percentile`); `None` when absent;
- `outcomes` — per-request `"ok"`/`"error"`/`"timeout"`/`"rate_limited"` →
  `error_rate` / `timeout_rate`; `None` when absent;
- `error_budget`, `slo_ms` → `within_error_budget`, `p95_within_slo` (`None`
  when the budget/SLO or the measure is absent);
- `partial_retrieval` + `answered`/`abstained` → the safe-degradation check;
- observed `retried` / `idempotent` / `recovered` flags — recorded, not computed.

`false_answer_under_partial_retrieval(scenario)` is the standalone
safe-degradation check: `True` (a false answer — unsafe) if retrieval partially
failed and the system answered anyway, `False` (safe: it abstained), or `None`
when the check does not apply (no partial failure) or applies but no answer
signal is present. The rollup reports `any_false_answer` (`None` if none were
determinable) beside `n_partial_retrieval_checked` / `n_partial_retrieval_unknown`
so a pass cannot hide an untested slice.

Jurisdiction-specific evidence packs (also PRD step 5) are out of scope here:
they select which evidence view is requested per the jurisdiction matrix and are
additive on top of these checks.
