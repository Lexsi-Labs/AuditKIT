"""CLI seed wiring: --seed / --split-seed must not silently override
RunConfig/SplitConfig's own defaults with an unset-flag None.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from auditkit.cli import _build_eval_parser, _run_eval


def _parse(argv):
    return _build_eval_parser().parse_args(argv)


def test_no_seed_flags_leaves_runconfig_seed_none(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate(dataset, **kwargs):
        captured.update(kwargs)
        result = MagicMock()
        result.summary.return_value = "ok"
        return result

    monkeypatch.setattr("auditkit.cli.evaluate", fake_evaluate)
    csv = tmp_path / "d.csv"
    csv.write_text("input,target\nhi,hi\n")

    args = _parse(["--model", "echo", "--csv", str(csv)])
    _run_eval(args)

    assert captured["config"].seed is None


def test_seed_flag_reaches_runconfig(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate(dataset, **kwargs):
        captured.update(kwargs)
        result = MagicMock()
        result.summary.return_value = "ok"
        return result

    monkeypatch.setattr("auditkit.cli.evaluate", fake_evaluate)
    csv = tmp_path / "d.csv"
    csv.write_text("input,target\nhi,hi\n")

    args = _parse(["--model", "echo", "--csv", str(csv), "--seed", "7"])
    _run_eval(args)

    assert captured["config"].seed == 7


def test_split_strategy_without_split_seed_keeps_splitconfig_default_zero(monkeypatch, tmp_path):
    """--split-strategy without --split-seed used to pass seed=None straight
    into SplitConfig, silently overriding its own reproducibility-motivated
    seed=0 default (None draws fresh OS entropy every call, reshuffling
    differently each run)."""
    captured = {}

    def fake_evaluate(dataset, **kwargs):
        captured.update(kwargs)
        result = MagicMock()
        result.summary.return_value = "ok"
        return result

    monkeypatch.setattr("auditkit.cli.evaluate", fake_evaluate)
    csv = tmp_path / "d.csv"
    csv.write_text("input,target\nhi,hi\n")

    args = _parse(["--model", "echo", "--csv", str(csv), "--split-strategy", "random"])
    _run_eval(args)

    assert captured["config"].split.seed == 0


def test_split_seed_flag_reaches_splitconfig(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate(dataset, **kwargs):
        captured.update(kwargs)
        result = MagicMock()
        result.summary.return_value = "ok"
        return result

    monkeypatch.setattr("auditkit.cli.evaluate", fake_evaluate)
    csv = tmp_path / "d.csv"
    csv.write_text("input,target\nhi,hi\n")

    args = _parse(["--model", "echo", "--csv", str(csv), "--split-strategy", "random", "--split-seed", "99"])
    _run_eval(args)

    assert captured["config"].split.seed == 99
