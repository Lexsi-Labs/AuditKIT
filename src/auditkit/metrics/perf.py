from __future__ import annotations

import math


class LatencyStats:
    def __init__(self) -> None:
        self._values: list[float] = []

    def record(self, latency_ms: float) -> None:
        self._values.append(latency_ms)

    def stats(self) -> dict[str, float]:
        if not self._values:
            return {
                "count": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0, "std": 0.0,
            }
        ordered = sorted(self._values)
        n = len(ordered)
        mean = sum(ordered) / n
        variance = sum((v - mean) ** 2 for v in ordered) / n

        def percentile(p: int) -> float:
            return ordered[int(p / 100 * (n - 1))]

        return {
            "count": n, "mean": mean,
            "min": ordered[0], "max": ordered[-1],
            "p50": percentile(50), "p95": percentile(95), "p99": percentile(99),
            "std": math.sqrt(variance),
        }


class Throughput:
    def __init__(self) -> None:
        self._total_time_ms = 0.0
        self._total_requests = 0

    def record(self, total_ms: float, num_requests: int = 1) -> None:
        self._total_time_ms += total_ms
        self._total_requests += num_requests

    def stats(self) -> dict[str, float]:
        if self._total_time_ms == 0:
            rps = 0.0
        else:
            rps = self._total_requests / (self._total_time_ms / 1000.0)
        return {
            "rps": rps,
            "total_requests": self._total_requests,
            "total_time_ms": self._total_time_ms,
        }
