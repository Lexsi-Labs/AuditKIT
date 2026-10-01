"""Read-only evidence importer for sibling Lexsi Labs products (X1 compat gate).

Each reader takes an artifact another product already wrote to disk -- parsed as
a plain dict -- and wraps it in a :class:`SourceEvidence` envelope that records
**provenance** (which product, which revision, which artifact kind), the
granularity of the evidence, the id fields that are actually present, and the id
fields that are **missing**. Nothing is invented: a reader never synthesizes an
episode/call/case id, never promotes a producer's reward/verdict/score to an
AuditKit outcome, and never mutates the producer's payload.

No producer library is imported. AgentTune, AlignTune, SafeTune, CuratorKIT and
CircuitKIT all pull torch / transformers / numpy / pydantic at module import, so
their artifacts are read as JSON only (same rule as :mod:`.importers`).

Shapes were checked against output from each library's own serializer at these
producer revisions (compat spike, 2026-09-25)::

    agenttune  36d4724   aligntune  a5708bc   safetune   a99d08a
    curatorkit 2aab65a   circuitkit 6430809

Canonical join-key vocabulary (snake_case, normalized across producers so one
name means one thing): ``episode_id``, ``call_id``, ``case_id``, ``trial_id``,
``row_id``, ``run_id``, ``stage``, ``policy_revision``, ``checkpoint_digest``,
``probe_set_id``, ``probe_id``, ``artifact_name``, ``artifact_path``,
``artifact_format``, ``model_name``, ``model_revision``, ``model_digest``,
``circuit_artifact_id``, ``source_uri``, ``source_file``, ``row``,
``provenance_chain``, ``timestamp``.

Joining is deliberately hostile: :func:`join_evidence` attaches evidence to an
episode only against ids the caller supplies **explicitly**, requires at least one
identity key, never pins aggregate evidence to one case or call, and refuses
unknown keys. An artifact whose required id sits
in ``missing_join_keys`` cannot be joined until the integration captures that id
at emission time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .types import OBSERVED, UNAVAILABLE, AgentEpisode

SCHEMA_VERSION = "lexsi_evidence/1"

#: Producer revisions the artifact shapes were read from.
REVISIONS = {
    "agenttune": "36d4724",
    "aligntune": "a5708bc",
    "safetune": "a99d08a",
    "curatorkit": "2aab65a",
    "circuitkit": "6430809",
}

GRANULARITIES = ("aggregate", "per_case", "per_event", "per_row")

# Keys that identify one episode, call, row, case, or model/artifact. A join must
# name at least one of them. Everything else (timestamp, tenant, decision, stage,
# task, format, source file, provenance step, ...) only describes evidence: it is
# shared across runs, rows or variants, so it may accompany an identity key and is
# then verified, but it never carries a join on its own.
IDENTITY_KEYS = ("episode_id", "call_id", "case_id", "trial_id", "row_id", "run_id",
                 "probe_id", "artifact_name", "model_digest", "model_revision",
                 "checkpoint_digest", "circuit_artifact_id")

# Per-case keys. Aggregate evidence summarizes many cases, so it can never join at
# this level, whatever id the caller supplies.
PER_CASE_KEYS = ("episode_id", "call_id", "case_id", "trial_id", "row_id", "probe_id")

# Canonical join key -> the AgentEpisode field it is verified against.
_EPISODE_FIELDS = {"episode_id": "source_id", "case_id": "case_id", "trial_id": "trial_id"}


def _strict(obj: Any) -> Any:
    """Non-finite floats -> ``None`` so ``to_dict()`` is strict JSON.

    CircuitKIT's report encoder inherits ``allow_nan=True`` and emits bare
    ``NaN``/``Infinity`` tokens, which strict parsers reject. The payload itself
    is kept verbatim; only the serialized view is sanitized.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


@dataclass
class SourceEvidence:
    """One artifact from another Lexsi product, read as-is.

    ``payload`` is the producer's dict, preserved verbatim. ``join_keys`` holds
    only the id fields the artifact actually carries; every id the artifact
    lacks is named in ``missing_join_keys`` rather than fabricated.
    ``coverage`` labels each field ``observed`` / ``unavailable`` -- including
    fields the producer computes but drops on the way to disk.

    ``is_outcome`` is ``False`` and stays ``False``: a producer's reward,
    verdict (AlignTune ``PASS``/``WARN``/``FAIL``), safety ``decision`` or
    faithfulness score is **provenance about what that product measured**, never
    an AuditKit verified outcome. An AuditKit outcome comes only from an oracle
    or judge run over a frozen case (AG-05).
    """

    source_product: str
    source_revision: str
    artifact_kind: str
    granularity: str
    payload: dict[str, Any] = field(default_factory=dict)
    join_keys: dict[str, Any] = field(default_factory=dict)
    missing_join_keys: list[str] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    is_outcome: bool = False
    # The producer's lexsi_provenance.json (read it with
    # auditkit.provenance.read_provenance(<output dir>)); None when absent.
    provenance: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {GRANULARITIES}, got {self.granularity!r}")

    def to_dict(self) -> dict[str, Any]:
        """Strict-JSON view (non-finite floats become ``None``)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "source_product": self.source_product,
            "source_revision": self.source_revision,
            "artifact_kind": self.artifact_kind,
            "granularity": self.granularity,
            "join_keys": _strict(self.join_keys),
            "missing_join_keys": list(self.missing_join_keys),
            "coverage": dict(self.coverage),
            "is_outcome": self.is_outcome,
            "payload": _strict(self.payload),
            "provenance": _strict(self.provenance),
        }


def _evidence(
    product: str,
    kind: str,
    granularity: str,
    payload: dict[str, Any],
    *,
    present: dict[str, Any],
    missing: Iterable[str] = (),
    observed: Iterable[str] = (),
    dropped: Iterable[str] = (),
    revision: Optional[str] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> SourceEvidence:
    """Build evidence: ``present`` entries that resolved to nothing become missing.

    ``dropped`` names fields the producer computes but never serializes -- the
    lost-field half of the coverage map.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"{kind}: expected a dict payload, got {type(payload).__name__}")
    empty = [k for k, v in present.items()
             if v is None or v == {} or (isinstance(v, list) and all(x in (None, "") for x in v))]
    join_keys = {k: v for k, v in present.items() if k not in empty}
    missing_keys = list(missing) + empty
    coverage = {k: OBSERVED for k in join_keys}
    coverage.update({k: OBSERVED for k in observed})
    coverage.update({k: UNAVAILABLE for k in missing_keys})
    coverage.update({k: UNAVAILABLE for k in dropped})
    return SourceEvidence(
        source_product=product,
        source_revision=revision or REVISIONS[product],
        artifact_kind=kind,
        granularity=granularity,
        payload=payload,
        join_keys=join_keys,
        missing_join_keys=missing_keys,
        coverage=coverage,
        provenance=provenance,
    )


def _keys(payload: dict[str, Any], names: Iterable[str]) -> list[str]:
    return [n for n in names if n in payload]


# --------------------------------------------------------------------------
# AlignTune @ a5708bc
# --------------------------------------------------------------------------
_AUDIT_FIELDS = (
    "reward_hacking", "sycophancy", "refusal_collapse", "verbosity_gain", "timestamp",
    "avg_response_tokens", "repetition_ratio", "unique_token_ratio", "degenerate_rate",
)


def aligntune_audit_report(payload: dict[str, Any], *,
                           revision: Optional[str] = None,
                           provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``AuditReport.to_json()`` -- an aggregate alignment scorecard.

    Aggregate over the whole probe set: no per-probe row, no per-case evidence,
    and no record of *which* artifact was audited (no model name, path or
    checkpoint digest). The only present key is a tz-less ``timestamp``, which
    orders reports but identifies nothing, so this artifact can never support a
    case-level join. ``verbosity_gain`` has two meanings: it is 0.0 unless a
    drift check ran, and ``check_drift`` writes a relative token delta into it --
    both for a ``RegressionReport`` variant and for a standalone
    ``audit_report_step_{N}.json`` whenever ``audit_baseline_report`` is set (the
    callback checks drift before saving). The training step lives only in that
    filename. ``per_case_evidence``/``probe_responses`` are AuditKit labels for
    what never reaches disk: per-probe responses are discarded locals.
    """
    return _evidence(
        "aligntune", "aligntune_audit_report", "aggregate", payload,
        present={"timestamp": payload.get("timestamp")},
        missing=("episode_id", "call_id", "case_id", "artifact_name",
                 "checkpoint_digest", "probe_set_id", "probe_id"),
        observed=_keys(payload, _AUDIT_FIELDS),
        dropped=("per_case_evidence", "probe_responses"),
        revision=revision, provenance=provenance,
    )


def aligntune_regression_report(payload: dict[str, Any], *,
                                revision: Optional[str] = None,
                                provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``RegressionReport.to_json()`` -- baseline vs export/quant variants.

    Still aggregate, but it is the one AlignTune artifact with an artifact
    identity: ``artifact.name`` (the key used in ``deltas``/``verdicts``),
    ``artifact.path`` and ``artifact.format``. These are list-valued (baseline
    plus every variant) and index-aligned, so a join naming several of them must
    match the same artifact record, never one variant's name with another's format.

    The runner hardcodes the baseline as ``name="baseline"``, ``format="hf"``.
    There is no checkpoint digest anywhere: ``artifact.metadata`` is the only
    pass-through slot and every CLI construction site leaves it ``{}``.
    ``per_expert_audit`` is computed on ``ArtifactResult`` but dropped by
    ``to_dict`` -- yet a per-expert ``refusal_collapse`` can still flip a verdict
    to ``FAIL``. A failing variant aborts the whole run (no report is written);
    what fails silently is evaluation: ``eval_results`` can be ``{}``, and its
    ``<metric>_delta`` keys are then absent. The top-level ``timestamp`` is taken
    at serialization time, not run time. Verdicts are the producer's regression
    verdict, never an AuditKit outcome.
    """
    results = [payload.get("baseline")] + list(payload.get("variants") or [])
    artifacts = [r.get("artifact") or {} for r in results if isinstance(r, dict)]
    return _evidence(
        "aligntune", "aligntune_regression_report", "aggregate", payload,
        present={
            "artifact_name": [a.get("name") for a in artifacts],
            "artifact_path": [a.get("path") for a in artifacts],
            "artifact_format": [a.get("format") for a in artifacts],
            "timestamp": payload.get("timestamp"),
        },
        missing=("episode_id", "call_id", "case_id", "checkpoint_digest",
                 "probe_set_id", "probe_id", "stage"),
        observed=_keys(payload, ("baseline", "variants", "deltas", "verdicts",
                                "thresholds", "timestamp")),
        dropped=("per_expert_audit", "per_case_evidence", "probe_responses"),
        revision=revision, provenance=provenance,
    )


# --------------------------------------------------------------------------
# SafeTune @ a99d08a
# --------------------------------------------------------------------------
def safetune_audit_event(payload: dict[str, Any], *,
                         revision: Optional[str] = None,
                         provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``SafetyAuditEvent.as_dict`` -- one runtime policy decision.

    Per *event*, and the event is anonymous: no episode id, no call/request id,
    no ``pre_check``/``post_check`` stage marker (the ``stage`` kwarg exists in
    ``_classify`` but is only logged), and no policy revision (``TenantPolicy``
    has no version field). ``tenant_id`` is tenant-level, not per-episode.
    Under the default ``redact_audit=True`` texts of 32 characters or fewer are
    fully masked and longer ones keep their first 16 and last 8 characters, so
    there is no usable content-text fallback either, and nothing in the event
    says whether it was redacted. On ``pre_check`` the ``response`` field is the
    middleware's enforced refusal text (``None`` on allow), not a model reply.

    A ``decision`` is a policy action, not a verified safety outcome -- and the
    keyword fallback that fires when no classifier is supplied leaves no flag in
    the event, so an uncalibrated block is indistinguishable from a real one.
    Call-level attribution needs ``episode_id`` + ``call_id`` captured at the
    invocation boundary and passed to :func:`join_evidence` explicitly.
    """
    meta = payload.get("metadata") or {}
    return _evidence(
        "safetune", "safetune_audit_event", "per_event", payload,
        present={
            "tenant_id": payload.get("tenant_id"),
            "decision": payload.get("decision"),
            "reason": payload.get("reason"),
            "timestamp": payload.get("timestamp"),
        },
        missing=("episode_id", "call_id", "case_id", "stage", "policy_revision"),
        observed=([k for k in ("prompt", "response", "metadata") if payload.get(k) is not None]
                  + _keys(meta, ("category", "matched_keyword", "tool_name"))),
        dropped=(("classifier_provenance", "used_default_detector", "redaction_applied")
                 + (("response",) if payload.get("response") is None else ())),
        revision=revision, provenance=provenance,
    )


def safetune_bench_row(payload: dict[str, Any], *,
                       revision: Optional[str] = None,
                       provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``write_jsonl`` scored row / ``write_bench_jsonl`` raw row.

    Per row, with no case id: the only in-band handle is the raw ``prompt``
    text, and model identity lives in the *filename*
    (``<model>__base__<bench>_scored.jsonl``), never in the row. Only the
    library helper ``write_jsonl`` writes a ``bench`` field, and nothing in
    SafeTune calls it: the raw dump and the production ``eval_safety`` scored
    rows have no ``bench``. The AdvBench inline path writes ``judge``
    (``"string_match"``); every other path drops it, and the
    ``asr``/``refusal_rate``/``hard1k_*`` aggregates are never written.
    ``score``/``refused`` are SafeTune's grade, kept as provenance for judge
    disagreement analysis -- not an AuditKit outcome.
    """
    return _evidence(
        "safetune", "safetune_bench_row", "per_row", payload,
        present={"bench": payload.get("bench"), "prompt": payload.get("prompt")},
        missing=("case_id", "episode_id", "call_id", "model_name", "timestamp"),
        observed=_keys(payload, ("prompt", "response", "score", "refused", "bench", "judge")),
        dropped=[k for k in ("judge", "asr", "refusal_rate") if k not in payload],
        revision=revision, provenance=provenance,
    )


# --------------------------------------------------------------------------
# CuratorKIT @ 2aab65a
# --------------------------------------------------------------------------
def curatorkit_data_sample(payload: dict[str, Any], *,
                           revision: Optional[str] = None,
                           provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """A full ``DataSample`` row: ``rejected.jsonl`` or a checkpoint stage row.

    These are the only CuratorKIT row artifacts that keep ``id`` **and** the
    append-only ``provenance_chain``. ``rejected.jsonl`` is always emitted but
    covers rejected rows only. There is no ``passed.jsonl``: checkpoints are
    stage snapshots ``.checkpoints/<StepClass>.jsonl`` (``_after_readers``,
    ``SchemaGate``, ``ExactDeduplicator``, ``TextCleaner``), and the kept set is
    the last non-exporter stage file. They are written only on the async
    ``Pipeline.run_async`` path; ``Curator.run()`` without a generation task
    takes the sync path and leaves ``.checkpoints/`` empty even with
    ``enable_checkpoint=True``. So by default no emitted artifact carries lineage
    for *kept* rows.

    The chain's first step is the reader: ``Connector`` for every tabular input
    (JSONL/CSV/JSON/parquet/HF; ``JSONLReader`` is only a manifest stage name)
    or ``PDFReader``, followed by ``SchemaGate``, ``ExactDeduplicator``,
    ``TextCleaner``. The row locator lives in that first step's notes:
    ``notes.line_number`` (1-based physical line) for tabular rows -- also
    copied to ``metadata.line_number`` on Connector rejections -- and
    ``source_file`` (+ ``page``/``chunk_index``) for PDF rows, where kept chunks
    also carry ``metadata.source_file``. Nothing writes ``metadata.row``.
    Tabular rows therefore have ``row`` but no ``source_file`` (``source_uri``
    names the input); PDF rows have ``source_file`` but no ``row``.

    Caveats: ``id`` is a fresh uuid4 per run (a Connector-rejected row always
    gets one) and is only promoted from an input ``id`` column; the timestamp
    format diverges between writers (``rejected.jsonl`` uses a space separator,
    the checkpoint ISO ``T``).
    """
    meta = payload.get("metadata") or {}
    chain = payload.get("provenance_chain") or []
    first = chain[0] if chain and isinstance(chain[0], dict) else {}
    notes = first.get("notes") if isinstance(first.get("notes"), dict) else {}
    rejected = "rejection_reason" in payload
    return _evidence(
        "curatorkit",
        "curatorkit_rejected_sample" if rejected else "curatorkit_data_sample",
        "per_row", payload,
        present={
            "row_id": payload.get("id"),
            "source_uri": payload.get("source_uri"),
            "source_file": meta.get("source_file") or notes.get("source_file"),
            "row": meta.get("line_number", notes.get("line_number")),
            "provenance_chain": [s.get("step_name") for s in chain if isinstance(s, dict)],
        },
        observed=_keys(payload, ("instruction", "input", "output", "chosen", "rejected",
                                 "label", "task_type", "metadata", "provenance_chain",
                                 "rejection_reason", "rejecting_step", "diagnosis")),
        revision=revision, provenance=provenance,
    )


def curatorkit_manifest(payload: dict[str, Any], *,
                        revision: Optional[str] = None,
                        provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``ProvenanceManifest.write`` -- ``manifest.json``, always emitted.

    Aggregate: stage counts, rejection breakdown, dedup/token stats, tool
    versions. It has no per-row id and no provenance chain, so it cannot
    recover row lineage after the fact. ``source_files`` entries are
    ``{"path": ...}`` only (the path as passed, not a basename) -- the
    docstring's ``sha256`` is not emitted -- and the list is empty for the
    common tabular (JSONL/CSV/parquet) runs because only the PDF reader writes
    the ``source_file`` provenance note. ``pipeline_config_hash`` hashes the
    config, so every run of one config shares it: it is a descriptor, not a run
    id. ``stage_counts`` names the reader stage ``JSONLReader``/``PDFReader``
    (never the provenance ``Connector``), and an exporter's ``exported_count`` is
    the rows handed to it, not rows written (a task-incompatible exporter can
    report 2 and write an empty file).
    """
    files = payload.get("source_files") or []
    return _evidence(
        "curatorkit", "curatorkit_manifest", "aggregate", payload,
        present={
            "pipeline_config_hash": payload.get("pipeline_config_hash"),
            "source_files": [f.get("path") for f in files if isinstance(f, dict)],
            "run_timestamp": payload.get("run_timestamp"),
        },
        missing=("row_id", "provenance_chain", "source_file_sha256"),
        observed=_keys(payload, ("stage_counts", "rejected_breakdown", "dedup_stats",
                                 "token_stats", "wall_clock_seconds", "tool_versions",
                                 "diversity_stats", "diagnostic_stats")),
        # CuratorKIT embeds its lexsi_provenance.json as manifest["provenance"].
        revision=revision, provenance=provenance or payload.get("provenance"),
    )


def curatorkit_trainer_export_row(payload: dict[str, Any], *, fmt: Optional[str] = None,
                                  revision: Optional[str] = None,
                                  provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``sft_alpaca.jsonl`` / ``sft_sharegpt.jsonl`` row -- the lineage-loss case.

    **Zero join keys.** Both exporters rebuild a fresh record dict and drop
    ``id``, ``source_uri``, ``provenance_chain``, ``task_type`` and
    ``metadata``. ``CorpusExporter`` rows (not read here) keep ``source_uri``,
    ``source_file``, ``page`` and ``chunk_index`` but no ``id`` or chain; only
    PDF runs fill ``source_file``/``chunk_index``. Line position is
    *not* a join either: exporters skip rows whose ``task_type`` does not match,
    so output line N is not input sample N in a mixed-task run.

    The only way back is out-of-band: keep a :func:`lineage_sidecar` built from
    the ``DataSample``/checkpoint/rejected rows and have the export step record
    the row id it wrote, then pass that id to :func:`join_evidence` explicitly.
    """
    if fmt is None:
        if "instruction" in payload:
            fmt = "alpaca"
        elif "conversations" in payload:
            fmt = "sharegpt"
        else:
            raise ValueError(
                "unrecognized CuratorKIT trainer export row: expected an alpaca "
                f"{{instruction,input,output}} or sharegpt {{conversations}} row, got keys "
                f"{sorted(payload)}")
    return _evidence(
        "curatorkit", f"curatorkit_{fmt}_export_row", "per_row", payload,
        present={},
        missing=("row_id", "source_uri", "provenance_chain", "task_type",
                 "source_file", "row"),
        observed=_keys(payload, ("instruction", "input", "output", "conversations")),
        revision=revision, provenance=provenance,
    )


# --------------------------------------------------------------------------
# CircuitKIT @ 6430809
# --------------------------------------------------------------------------
_CIRCUIT_SCORES = ("patching_score", "ablation_score", "stability", "robustness",
                   "baseline_comparison", "generalization", "intervention_reliability")


def circuitkit_faithfulness_report(payload: dict[str, Any], *,
                                   revision: Optional[str] = None,
                                   provenance: Optional[dict[str, Any]] = None) -> SourceEvidence:
    """``FaithfulnessReport.to_json()`` -- aggregate MECHANISTIC evidence.

    Patching / ablation / stability / robustness / baseline / generalization /
    intervention-reliability of a discovered circuit: whether the circuit
    faithfully explains the model's behavior. It is **not** task accuracy and
    **not** a safety verdict, so ``is_outcome`` stays ``False`` and these scores
    never stand in for a state-based oracle.

    Aggregate over the whole dataloader -- there are no per-case rows, so
    pairing it to a held-out AuditKit case is impossible by construction, not
    merely for want of an id. ``asdict`` always writes all eight fields, so a
    pillar that was not computed is ``null``: only non-null scores count as
    observed (the IBCircuit paths never copy ``intervention_reliability``). The
    present keys are descriptors (``metadata.model`` is a TransformerLens alias
    on one path and a config name on another, while ``CircuitArtifact`` uses the
    HF repo id); ``run_full_faithfulness`` writes the sentinel ``"unknown"`` for
    an unresolved model/algorithm/task/scope, which counts as missing here, and
    ``sparsity`` may be a real 0.0. ``metadata.timestamp`` is an epoch float and
    is absent when no graph pillar ran. The IBCircuit fast path replaces
    ``metadata`` wholesale (``random_avg``/``eval_mode`` only), so every
    descriptor can be absent. No CircuitKIT artifact carries an id or digest --
    ``CircuitArtifact.save_json`` included -- so ``circuit_artifact_id`` and
    ``model_digest`` must be minted by the caller.
    """
    meta = payload.get("metadata") or {}

    def _named(key: str) -> Any:
        v = meta.get(key)
        return None if v == "unknown" else v

    return _evidence(
        "circuitkit", "circuitkit_faithfulness_report", "aggregate", payload,
        present={
            "model_name": _named("model"),
            "task": _named("task"),
            "algorithm": _named("algorithm"),
            "level": _named("level"),
            "scope": _named("scope"),
            "sparsity": meta.get("sparsity"),
            "timestamp": meta.get("timestamp"),
        },
        missing=("model_revision", "model_digest", "circuit_artifact_id", "run_id",
                 "dataset", "case_id", "episode_id"),
        observed=[k for k in _CIRCUIT_SCORES + ("metadata",) if payload.get(k) is not None],
        dropped=("per_case_scores", "checkpoint_backreference")
        + tuple(k for k in _CIRCUIT_SCORES if payload.get(k) is None),
        revision=revision, provenance=provenance,
    )


# --------------------------------------------------------------------------
# joining
# --------------------------------------------------------------------------
@dataclass
class JoinResult:
    """Outcome of one join attempt. ``basis`` records which id matched what."""

    joined: bool
    basis: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


def _withheld(value: Any) -> bool:
    """``None``, blank strings, empty containers and non-finite floats are no id."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


def join_evidence(episode: AgentEpisode, evidence: SourceEvidence,
                  by: dict[str, Any]) -> JoinResult:
    """Attach ``evidence`` to ``episode`` using EXPLICITLY-supplied ids only.

    ``by`` is the join contract: ``{canonical_key: value}``, where each value is
    a string or integer id. The join is refused, with the reason naming the key,
    when:

    - ``by`` names no :data:`IDENTITY_KEYS` key (descriptors such as timestamp,
      tenant, stage, task, format or source file never carry a join alone);
    - a value is withheld (``None``, blank, empty, non-finite) or not a scalar id;
    - the key is neither carried by the artifact nor declared missing by it;
    - the evidence is aggregate and the key is per-case (:data:`PER_CASE_KEYS`):
      a scorecard over many cases cannot be pinned to one case or call;
    - a carried value does not match (list-valued keys, such as a regression
      report's artifact name/path/format, must all match the same record);
    - the value contradicts the episode's own ``source_id``/``case_id``/
      ``trial_id`` or names a call the episode does not contain;
    - the evidence is per-event and ``by`` does not name its ``call_id``.

    On success the evidence and its join basis are appended to
    ``episode.metadata["lexsi_evidence"]``, and the episode's outcome is left
    untouched -- a producer verdict is provenance, not a verified outcome.
    """
    kind = evidence.artifact_kind
    if not by:
        return JoinResult(False, {}, (
            f"no explicit join key supplied for {kind}; it carries "
            f"{sorted(evidence.join_keys) or 'no id at all'} and is missing "
            f"{evidence.missing_join_keys}"))
    for key in by:
        if evidence.granularity == "aggregate" and key in PER_CASE_KEYS:
            return JoinResult(False, {}, (
                f"{kind} is aggregate evidence over many cases; it cannot join at "
                f"{key!r} level"))
        if key not in evidence.join_keys and key not in evidence.missing_join_keys:
            return JoinResult(False, {}, (
                f"unknown join key {key!r} for {kind}: it carries "
                f"{sorted(evidence.join_keys)} and declares missing "
                f"{evidence.missing_join_keys}"))
    if not any(k in IDENTITY_KEYS for k in by):
        return JoinResult(False, {}, (
            f"refusing to join {kind} on {sorted(by)} alone: these keys describe the "
            f"evidence, they do not identify one episode/call/row (name one of "
            f"{list(IDENTITY_KEYS)})"))

    basis: dict[str, Any] = {}
    records: Optional[set[int]] = None   # artifact records all list-valued keys agree on
    for key, value in by.items():
        in_artifact = key in evidence.join_keys
        if _withheld(value):
            if in_artifact:
                return JoinResult(False, {}, f"no value supplied for join key {key!r}")
            return JoinResult(False, {}, (
                f"{kind} carries no {key!r} and the caller supplied none: {key!r} "
                "must be captured at emission time for this join"))
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return JoinResult(False, {}, (
                f"join key {key!r} must be a string or integer id, got "
                f"{type(value).__name__}"))
        if in_artifact:
            carried = evidence.join_keys[key]
            if isinstance(carried, list):
                hits = {i for i, v in enumerate(carried)
                        if v is not None and str(v) == str(value)}
                if not hits:
                    return JoinResult(False, {}, (
                        f"{key}={value!r} does not match the artifact's {key}={carried!r}"))
                records = hits if records is None else records & hits
                if not records:
                    listed = [k for k in by if isinstance(evidence.join_keys.get(k), list)]
                    return JoinResult(False, {}, (
                        f"{listed} each match the artifact but not the same record: "
                        f"{kind} has no single record with all of {by}"))
                artifact_value = value
            else:
                if _withheld(carried) or str(carried) != str(value):
                    return JoinResult(False, {}, (
                        f"{key}={value!r} does not match the artifact's {key}={carried!r}"))
                artifact_value = carried
            basis[key] = {"value": value, "matched": "artifact",
                          "artifact_value": artifact_value}
        else:
            basis[key] = {"value": value, "matched": "caller_supplied",
                          "artifact_value": None}

        ep_field = _EPISODE_FIELDS.get(key)
        if ep_field is not None:
            ep_val = getattr(episode, ep_field, None)
            if ep_val is not None:
                if str(ep_val) != str(value):
                    return JoinResult(False, {}, (
                        f"{key}={value!r} does not match episode.{ep_field}={ep_val!r}"))
                basis[key]["episode"] = f"{ep_field}={ep_val}"
        if key == "call_id":
            call_ids = [str(e.call_id) for e in episode.events if e.call_id is not None]
            if call_ids:
                if str(value) not in call_ids:
                    return JoinResult(False, {}, (
                        f"call_id={value!r} is not a call in this episode "
                        f"(call ids: {sorted(set(call_ids))})"))
                basis[key]["episode"] = f"events.call_id={value}"

    if evidence.granularity == "per_event" and "call_id" not in basis:
        return JoinResult(False, {}, (
            f"{kind} is one event; the join must name the 'call_id' it belongs to"))

    episode.metadata.setdefault("lexsi_evidence", []).append(
        {"evidence": evidence.to_dict(), "join_basis": _strict(basis)})
    return JoinResult(True, basis, None)


def lineage_sidecar(samples: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """``{row_id -> provenance}`` from CuratorKIT ``DataSample``-shaped rows.

    Accepts raw dicts or :class:`SourceEvidence` from
    :func:`curatorkit_data_sample` (rejected sidecar rows and checkpoint
    stage rows both work). The result is the sidecar an exporter needs so a
    lineage-dropping trainer export can be re-joined: alpaca and ShareGPT rows
    carry **no id, source_uri, provenance_chain or metadata at all**, so the
    export step must record which ``row_id`` it wrote for each line and pass it
    to :func:`join_evidence` explicitly. Re-joining by line position is wrong --
    exporters skip task-incompatible rows.

    A row without an ``id`` raises: there is nothing to key it by, and guessing
    would be exactly the fabricated lineage this module refuses. So does an id
    that repeats with different provenance; an exact repeat is harmless.
    """
    out: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(samples):
        row = s.payload if isinstance(s, SourceEvidence) else s
        if not isinstance(row, dict):
            raise TypeError(f"samples[{i}] is not a dict or SourceEvidence")
        row_id = row.get("id")
        if not row_id:
            raise ValueError(
                f"samples[{i}] has no 'id': a trainer export row (alpaca/sharegpt) "
                "carries no lineage -- build the sidecar from DataSample, checkpoint "
                "stage (.checkpoints/<Step>.jsonl) or rejected.jsonl rows instead")
        entry = {
            "source_uri": row.get("source_uri"),
            "task_type": row.get("task_type"),
            "metadata": row.get("metadata") or {},
            "provenance_chain": row.get("provenance_chain") or [],
            "rejected": "rejection_reason" in row,
            "rejection_reason": row.get("rejection_reason"),
            "rejecting_step": row.get("rejecting_step"),
        }
        prior = out.get(str(row_id))
        if prior is not None and prior != entry:
            raise ValueError(
                f"samples[{i}] repeats row id {row_id!r} with different provenance: "
                "one id cannot map to two lineages")
        out[str(row_id)] = entry
    return out
