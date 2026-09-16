"""Simple disk cache for evaluation results keyed by fingerprint."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .report import RunResult


class DiskCache:
    """Persist RunResult objects keyed by run fingerprint.

    Defaults to ``~/.cache/auditkit/runs/``.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self._dir = Path(cache_dir or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "auditkit", "runs",
        ))
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str) -> Path:
        return self._dir / f"{fingerprint}.json"

    def get(self, fingerprint: str) -> RunResult | None:
        path = self._path(fingerprint)
        if path.exists():
            with open(path, "r") as fh:
                return RunResult.from_dict(json.load(fh))
        return None

    def set(self, fingerprint: str, result: RunResult) -> None:
        path = self._path(fingerprint)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        tmp.rename(path)

    def clear(self) -> None:
        for p in self._dir.iterdir():
            if p.suffix == ".json":
                p.unlink()

    def __contains__(self, fingerprint: str) -> bool:
        return self._path(fingerprint).exists()
