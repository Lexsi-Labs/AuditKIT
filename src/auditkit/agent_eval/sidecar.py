"""AgentTune-compatible evaluation sidecar (A5, AG-19).

A **read-only emit**: it writes AuditKit's per-case outcome + coverage + trial
reliability as a compact JSONL sidecar keyed by the source trajectory id, so a
downstream AgentTune analysis can join it back to the original trajectory. It
never imports AgentTune and never mutates any source record -- the trajectory
and its reward are untouched (AG-19).

One JSON object per line (AgentTune's own on-disk shape is JSONL), strict JSON
and deterministic, so a run round-trips: :func:`write_sidecar` then
:func:`read_sidecar` reproduce the same records.

What still needs AgentTune-side support (documented seam): writing these records
*back into* AgentTune's training data, and a `Project` intake that reconstructs
episodes from a Project on disk, both need an AgentTune serializer that does not
exist at the pinned mirror commit (the only file AgentTune writes is the lossy
heal audit JSONL, which :mod:`.importers` refuses). This emit is the achievable
half: score AuditKit-side, hand AgentTune a keyed sidecar.
"""

from __future__ import annotations

from typing import Any, Optional

from .types import dumps_strict

SIDECAR_SCHEMA = "agent_eval_sidecar/1"


def sidecar_records(result: Any) -> list[dict[str, Any]]:
    """One record per case row, keyed by ``source_id`` (falls back to case/trial id).

    Each record carries the verified outcome, the coverage label, the imported
    source provenance (reward/verdict/scores -- diagnostic, never promoted), and
    the trial reliability when the run had repeated trials. It deliberately does
    not copy the full event trace: the sidecar joins to the original trajectory
    by key, it does not replace it.
    """
    records: list[dict[str, Any]] = []
    for row in getattr(result, "rows", []):
        key = row.source_id or row.case_id
        oc = row.outcome or {}
        rec: dict[str, Any] = {
            "schema_version": SIDECAR_SCHEMA,
            "source_id": row.source_id,
            "case_id": row.case_id,
            "key": key,
            "status": row.status,
            "outcome": {"verdict": oc.get("verdict"), "reason": oc.get("reason"),
                        "oracle": oc.get("source"), "diagnostic": oc.get("diagnostic", False)},
            "coverage_label": row.coverage_label,
            "coverage": dict(row.coverage or {}),
            "source_provenance": dict(row.provenance or {}),
        }
        if row.reliability:
            rec["reliability"] = dict(row.reliability)
        if row.disagreement:
            rec["disagreement"] = dict(row.disagreement)
        records.append(rec)
    return records


def write_sidecar(result: Any, path: Optional[str] = None) -> str:
    """Serialize :func:`sidecar_records` as strict JSONL; write it when ``path`` is given."""
    text = "\n".join(dumps_strict(r) for r in sidecar_records(result))
    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + ("\n" if text else ""))
    return text


def read_sidecar(path: str) -> list[dict[str, Any]]:
    """Read a sidecar JSONL back into records (round-trips :func:`write_sidecar`)."""
    import json
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
