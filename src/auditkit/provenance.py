"""``lexsi_provenance.json``: the lineage record every Lexsi library writes into
each output folder (dataset export, checkpoint/adapter, results).

AuditKIT reads it from ``hf:<dir>`` models and dataset folders into
``RunResult.metadata["inputs"]`` and writes its own next to saved results, with
those inputs (and their provenance) embedded, so lineage chains across a track.
Stdlib only; a missing or unreadable file reads as ``None``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

PROVENANCE_FILE = "lexsi_provenance.json"
SCHEMA = "lexsi.provenance/1"


def read_provenance(path: Any) -> Optional[dict[str, Any]]:
    """The provenance object of a folder (or of the file itself), else ``None``.

    Hub ids, URLs and folders without the file all give ``None``.
    """
    if not isinstance(path, (str, os.PathLike)):
        return None
    path = os.fspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, PROVENANCE_FILE)
    if os.path.basename(path) != PROVENANCE_FILE or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def input_ref(kind: str, ref: Any, config: Optional[str] = None) -> dict[str, Any]:
    """One ``inputs[]`` entry: what was consumed, plus its own provenance."""
    entry: dict[str, Any] = {"kind": kind, "ref": os.fspath(ref) if isinstance(ref, os.PathLike) else ref}
    if config is not None:
        entry["config"] = config
    entry["provenance"] = read_provenance(ref)
    return entry


def make_provenance(*, method: str = "evaluate", base_model: Optional[str] = None,
                    inputs: Iterable[dict[str, Any]] = (), params: Optional[dict[str, Any]] = None,
                    library: str = "auditkit") -> dict[str, Any]:
    """A provenance object for an output this library produced."""
    from importlib import metadata

    try:
        version: Optional[str] = metadata.version(library)
    except metadata.PackageNotFoundError:
        version = None
    return {
        "schema": SCHEMA, "library": library, "version": version, "git_sha": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_model, "method": method, "inputs": list(inputs), "params": params or {},
    }


def write_provenance(out_dir: str, provenance: dict[str, Any]) -> str:
    """Write *provenance* to ``<out_dir>/lexsi_provenance.json``; returns the path."""
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, PROVENANCE_FILE)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=str)
    return path
