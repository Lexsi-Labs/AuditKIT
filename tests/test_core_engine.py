"""Tests for T1 core engine: split, concurrency, retry, error handling."""

from __future__ import annotations

import threading


from auditkit.runspec import SplitConfig, RunConfig
from auditkit.errors import ModelError, MetricError, SampleSkipped, ModelTimeout
from auditkit.score import Stat


class TestSplitConfig:
    def test_sequential_split(self):
        sc = SplitConfig(strategy="sequential", train_ratio=0.5, test_ratio=0.5)
        assert sc.strategy == "sequential"
        assert sc.train_ratio == 0.5

    def test_random_split_with_seed(self):
        sc = SplitConfig(strategy="random", train_ratio=0.8, seed=42)
        assert sc.strategy == "random"

    def test_stratified_split(self):
        sc = SplitConfig(strategy="stratified", train_ratio=0.5, test_ratio=0.5)
        assert sc.strategy == "stratified"

    def test_defaults(self):
        sc = SplitConfig()
        assert sc.train_ratio == 0.0
        assert sc.test_ratio == 1.0


class TestRunConfig:
    def test_timeout_field(self):
        cfg = RunConfig(timeout=30.0)
        assert cfg.timeout == 30.0

    def test_split_field(self):
        sc = SplitConfig(strategy="sequential")
        cfg = RunConfig(split=sc)
        assert cfg.split.strategy == "sequential"

    def test_concurrency_field(self):
        cfg = RunConfig(concurrency=4)
        assert cfg.concurrency == 4


class TestModelTimeout:
    def test_error_type(self):
        assert issubclass(ModelTimeout, Exception)
        assert "timeout" in str(ModelTimeout("timeout error"))


class TestModelError:
    def test_error_type(self):
        assert issubclass(ModelError, Exception)


class TestMetricError:
    def test_error_type(self):
        assert issubclass(MetricError, Exception)


class TestSampleSkipped:
    def test_error_type(self):
        assert issubclass(SampleSkipped, Exception)


class TestStatThreadSafety:
    def test_concurrent_add(self):
        st = Stat("test")
        ts = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [st.add(1.0) for _ in range(100)])
            ts.append(t)
            t.start()
        for t in ts:
            t.join()
        assert st.count == 1000

    def test_to_dict_roundtrip(self):
        st = Stat("test")
        st.add(0.5)
        st.add(1.0)
        d = st.to_dict()
        st2 = Stat.from_dict(d)
        assert st2.count == 2
        assert round(st2.mean, 4) == 0.75
