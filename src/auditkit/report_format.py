"""Pretty-printed and markdown reports for RunResult."""
from __future__ import annotations
from typing import Any
from .report import RunResult


class Report:
    def __init__(self, result: RunResult) -> None:
        self.result = result

    def metric_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name, stat in self.result.stats.items():
            rows.append({
                "name": name,
                "mean": stat.mean,
                "stderr": stat.stderr,
                "std": stat.std,
                "n": stat.count,
            })
        return rows

    def __str__(self) -> str:
        lines = [f"Run {self.result.run_id}  (fingerprint {self.result.fingerprint})"]
        lines.append("")
        rows = self.metric_rows()
        if rows:
            lines.append(f"{'Metric':<30} {'Mean':>8} {'±Stderr':>8} {'Std':>8} {'N':>6}")
            lines.append("-" * 62)
            for r in rows:
                lines.append(
                    f"{r['name']:<30} {r['mean']:>8.4f} {r['stderr']:>8.4f} "
                    f"{r['std']:>8.4f} {r['n']:>6}"
                )
            lines.append("")
        lines.append(f"{len(self.result.predictions)} predictions")
        wrong = len(self.result.wrong_only())
        if wrong:
            lines.append(f"{wrong} wrong / {len(self.result.predictions)} total")
        return "\n".join(lines)

    def markdown(self) -> str:
        rows = self.metric_rows()
        if not rows:
            return ""
        lines = ["| Metric | Mean | ±Stderr | Std | N |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['name']} | {r['mean']:.4f} | {r['stderr']:.4f} "
                f"| {r['std']:.4f} | {r['n']} |"
            )
        return "\n".join(lines)
