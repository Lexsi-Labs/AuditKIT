"""Experiment tracking for multi-run comparison and persistence."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .diff import RunDiff
from .errors import ExtraNotInstalled
from .report import RunResult


@dataclass
class Experiment:
    name: str
    runs: list[RunResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, run: RunResult) -> None:
        self.runs.append(run)

    def aggregate(self, metric: str | None = None) -> dict[str, float]:
        if not self.runs:
            return {}
        if metric:
            vals = [r.headline.get(metric, 0.0) for r in self.runs if r.headline]
            return {metric: sum(vals) / len(vals) if vals else 0.0}
        all_metrics = set()
        for r in self.runs:
            all_metrics.update(r.headline.keys())
        result = {}
        for m in sorted(all_metrics):
            vals = [r.headline.get(m, 0.0) for r in self.runs if r.headline]
            result[m] = sum(vals) / len(vals) if vals else 0.0
        return result

    def leaderboard(self) -> list[dict[str, Any]]:
        if not self.runs:
            return []
        first_metric = next(iter(self.runs[0].headline.keys()), None)
        rows = []
        for r in self.runs:
            row = {"run_id": r.run_id, "fingerprint": r.fingerprint}
            row.update(r.headline)
            rows.append(row)
        if first_metric:
            rows.sort(key=lambda x: x.get(first_metric, 0), reverse=True)
        return rows

    def pairwise_diff(self, run_a_index: int = 0, run_b_index: int = -1) -> RunDiff:
        return RunDiff(self.runs[run_a_index], self.runs[run_b_index])

    def log_mlflow(self, experiment_name: str | None = None, tracking_uri: str | None = None) -> None:
        try:
            import mlflow
        except ImportError:
            raise ExtraNotInstalled("mlflow", "pip install auditkit[mlflow]")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        exp_name = experiment_name or self.name
        mlflow.set_experiment(exp_name)
        for run in self.runs:
            with mlflow.start_run(run_name=run.run_id):
                for metric_name, value in run.headline.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.log_param("fingerprint", run.fingerprint)
                mlflow.log_param("model", run.model_spec or "")
                if run.config:
                    for k in ("temperature", "seed", "limit", "num_fewshot", "concurrency"):
                        v = getattr(run.config, k, None)
                        if v is not None:
                            mlflow.log_param(k, v)
                if run.errors:
                    mlflow.log_metric("failed_count", run.failed_count)
                    mlflow.log_text(json.dumps(run.errors, default=str), "errors.json")

    def significance(self, metric: str, method: str = "bootstrap", n_resamples: int = 1000) -> dict[str, Any]:
        """Compare the best two runs and return a p-value.

        Aligns per-sample scores **by sample_id** via the shared paired bootstrap
        (see :mod:`auditkit._bootstrap`), rather than zipping by list position.
        """
        if len(self.runs) < 2:
            return {"error": "Need at least 2 runs for significance test"}
        from ._bootstrap import paired_bootstrap, score_pairs

        scored = []
        for r in self.runs:
            vals = [p.score for p in r.predictions if p.score is not None]
            if vals:
                scored.append((sum(vals) / len(vals), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) < 2:
            return {"error": "Need at least 2 runs with predictions for significance test"}
        baseline_run = scored[0][1]
        contrast_run = scored[1][1]
        bs = paired_bootstrap(score_pairs(baseline_run, contrast_run, None), n_resamples=n_resamples)
        if "error" in bs:
            return {"error": "Need at least 2 paired predictions per run"}
        return {
            "baseline": baseline_run.run_id,
            "contrast": contrast_run.run_id,
            "delta": round(bs["mean_baseline"] - bs["mean_candidate"], 6),
            "p_value": bs["p_value"],
            "significant": bs["significant"],
            "method": method,
            "n_resamples": n_resamples,
            "n_samples": bs["n"],
        }


class ExperimentDB:
    def __init__(self, path: str | None = None) -> None:
        self._dir = Path(path or os.path.join(
            os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
            "auditkit", "experiments",
        ))
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self._dir / f"{safe}.json"

    def save(self, experiment: Experiment) -> None:
        data = {
            "name": experiment.name,
            "metadata": experiment.metadata,
            "run_count": len(experiment.runs),
            "runs": [r.to_dict() for r in experiment.runs],
        }
        with open(self._path(experiment.name), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, name: str) -> Experiment:
        with open(self._path(name)) as f:
            data = json.load(f)
        exp = Experiment(name=data["name"], metadata=data.get("metadata", {}))
        for rd in data.get("runs", []):
            exp.add(RunResult.from_dict(rd))
        return exp

    def list_experiments(self) -> list[str]:
        return [p.stem for p in self._dir.iterdir() if p.suffix == ".json"]
