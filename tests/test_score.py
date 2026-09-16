import math

import pytest

from auditkit.score import Score, Stat, pass_at_k, pass_hat_k
from auditkit.types import ScoreKind, DataType, Direction, Source


# ---- Score ---------------------------------------------------------------

def test_score_defaults():
    s = Score(name="acc", value=1.0)
    assert s.kind == ScoreKind.BENCHMARK
    assert s.data_type == DataType.NUMERIC
    assert s.direction == Direction.MAXIMIZE
    assert s.weight == 1.0
    assert s.source == Source.SDK


def test_passed_is_none_without_threshold():
    assert Score(name="acc", value=0.9).passed is None


def test_passed_respects_direction():
    # MAXIMIZE: value must be >= threshold
    assert Score(name="acc", value=0.8, threshold=0.5).passed is True
    assert Score(name="acc", value=0.3, threshold=0.5).passed is False
    # MINIMIZE: value must be <= threshold (e.g. latency, toxicity)
    lat = Score(name="p95_latency", value=120.0, threshold=200.0,
                direction=Direction.MINIMIZE)
    assert lat.passed is True
    assert Score(name="asr", value=0.4, threshold=0.1,
                 direction=Direction.MINIMIZE).passed is False


# ---- Stat ----------------------------------------------------------------

def test_stat_accumulates_and_reports_mean():
    st = Stat("acc")
    for v in (1.0, 0.0, 1.0, 1.0):
        st.add(v)
    assert st.count == 4
    assert st.mean == 0.75
    assert st.min == 0.0
    assert st.max == 1.0


def test_stat_add_returns_self_for_chaining():
    st = Stat("x")
    assert st.add(1.0) is st


def test_stat_stderr_and_std():
    st = Stat("x")
    for v in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        st.add(v)
    assert st.mean == 5.0
    # population std of this classic set is 2.0
    assert st.std == pytest.approx(2.0)
    assert st.stderr == pytest.approx(2.0 / math.sqrt(8))


def test_stat_empty_is_safe():
    st = Stat("x")
    assert st.count == 0
    assert st.mean == 0.0
    assert st.stderr == 0.0


def test_stat_percentile():
    st = Stat("lat")
    for v in range(1, 101):  # 1..100
        st.add(float(v))
    assert st.percentile(50) == pytest.approx(50.5, abs=1.0)
    assert st.percentile(99) == pytest.approx(99.0, abs=1.5)


def test_stat_to_dict_roundtrips_summary():
    st = Stat("acc")
    st.add(1.0).add(0.0)
    d = st.to_dict()
    assert d["name"] == "acc"
    assert d["count"] == 2
    assert d["mean"] == 0.5


# ---- pass@k / pass^k -----------------------------------------------------

def test_pass_at_k_capability():
    # n=5 samples, c=2 correct: probability >=1 correct in a draw of k
    assert pass_at_k(5, 0, 1) == 0.0            # nothing correct -> never
    assert pass_at_k(5, 5, 1) == 1.0            # all correct -> always
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)   # 2/5
    # k=2 draw from n=5 with c=2: 1 - C(3,2)/C(5,2) = 1 - 3/10
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)


def test_pass_hat_k_reliability():
    # pass^k = C(c,k)/C(n,k): all k draws correct
    assert pass_hat_k(5, 5, 2) == 1.0
    assert pass_hat_k(5, 2, 2) == pytest.approx(0.1)   # C(2,2)/C(5,2)=1/10
    assert pass_hat_k(5, 1, 2) == 0.0                  # can't draw 2 correct


def test_pass_at_k_guards_k_greater_than_n():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)
