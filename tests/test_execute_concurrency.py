"""Runner.execute concurrency safety: respect model.threadsafe, no empty chunks.

Regression for the chunking bug: when a caller sets concurrency > 1, execute()
used to (1) call a NON-thread-safe model from that many threads at once, and
(2) call generate([]) on empty chunks when there were fewer requests than the
configured concurrency. The default was also lowered to 1 (runspec) so the
common path avoids fan-out entirely; this covers the explicit opt-in path.
"""

from __future__ import annotations

import threading

from auditkit.model import Model, Request, Result_, Generated
from auditkit.runner import Runner
from auditkit.sample import Sample


class _RecordingModel(Model):
    """Records how many times generate() is called and the batch sizes it saw,
    with a declarable threadsafe flag."""

    def __init__(self, threadsafe: bool):
        self.threadsafe = threadsafe
        self.name = "rec"
        self.calls = 0
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()

    def generate(self, requests):
        with self._lock:
            self.calls += 1
            self.batch_sizes.append(len(requests))
        return [Result_(completions=[Generated(text="x")]) for _ in requests]


def _batch(n):
    return [(Sample(input=f"q{i}"), [Request(prompt=f"q{i}")]) for i in range(n)]


def test_non_threadsafe_model_is_called_once_even_at_high_concurrency():
    m = _RecordingModel(threadsafe=False)
    out = Runner().execute(m, _batch(5), concurrency=8)
    assert m.calls == 1                      # single batched call, not fanned out
    assert m.batch_sizes == [5]              # whole batch in one go
    assert len(out) == 5


def test_threadsafe_model_fans_out_without_empty_chunks():
    m = _RecordingModel(threadsafe=True)
    out = Runner().execute(m, _batch(3), concurrency=8)
    # 3 requests, concurrency 8 -> at most 3 chunks, never an empty generate([])
    assert m.calls == 3
    assert all(size >= 1 for size in m.batch_sizes)
    assert len(out) == 3


def test_concurrency_one_is_a_single_call():
    m = _RecordingModel(threadsafe=True)
    Runner().execute(m, _batch(4), concurrency=1)
    assert m.calls == 1
    assert m.batch_sizes == [4]


def test_results_realign_to_samples():
    m = _RecordingModel(threadsafe=False)
    out = Runner().execute(m, _batch(3), concurrency=8)
    assert [len(results) for _s, results in out] == [1, 1, 1]


# ---- throughput must not be understated ~concurrency-fold -----------------
# Parallel chunks overlap in wall-clock; summing each chunk's own elapsed (a
# per-chunk throughput.record) inflates total_time_ms and collapses rps to the
# serial rate. execute() must record ONE span for the whole parallel region.

import time

from auditkit.metrics.perf import LatencyStats, Throughput


class _SleepyModel(Model):
    """Threadsafe backend that spends real (small) wall time per request, so
    parallel chunks genuinely overlap."""

    threadsafe = True
    name = "sleepy"

    def generate(self, requests):
        time.sleep(0.05 * len(requests))
        return [Result_(completions=[Generated(text="x")]) for _ in requests]


def test_parallel_rps_is_not_divided_by_concurrency():
    lat, thr = LatencyStats(), Throughput()
    Runner().execute(_SleepyModel(), _batch(8), concurrency=4,
                     latency=lat, throughput=thr)
    lstats, tstats = lat.stats(), thr.stats()
    # One latency sample per chunk (workers still record latency per call).
    assert lstats["count"] == 4
    assert tstats["total_requests"] == 8
    # total_time_ms is the single parallel span, NOT the sum of the 4 chunk
    # durations. The buggy sum would be ~= mean*count; the real span is far below.
    summed = lstats["mean"] * lstats["count"]
    assert tstats["total_time_ms"] < summed * 0.75, (tstats, lstats)
    # And rps reflects the parallelism instead of collapsing to the serial rate.
    serial_rps = tstats["total_requests"] / (summed / 1000.0)
    assert tstats["rps"] > serial_rps * 1.5, (tstats, lstats)


def test_serial_throughput_unchanged():
    """concurrency=1 stays on the summing path -- behavior identical."""
    lat, thr = LatencyStats(), Throughput()
    Runner().execute(_SleepyModel(), _batch(4), concurrency=1,
                     latency=lat, throughput=thr)
    # Single batched call: one latency sample, all 4 requests counted once.
    assert lat.stats()["count"] == 1
    assert thr.stats()["total_requests"] == 4
