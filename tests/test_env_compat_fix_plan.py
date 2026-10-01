"""Regression tests for G15, G17 and G19 (fix plan #15): the check_compat rules for an old
accelerate under transformers 5, sglang on Python 3.13, and a torch/torchaudio CUDA mismatch."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import auditkit.compat as compat


# -- G15, G17, G19: check_compat -------------------------------------------------------------------------------------

def run(monkeypatch, versions, py=(3, 11, 9), **kw):
    versions = {compat._norm(k): v for k, v in {"auditkit": "1.0.0", **versions}.items()}
    monkeypatch.setattr(compat, "_dist_version", lambda n: versions.get(compat._norm(n)))
    monkeypatch.setattr(compat, "_dist_requires", lambda n: [])
    monkeypatch.setattr(compat, "_system", lambda: "Linux")
    monkeypatch.setattr(compat, "_libc_ver", lambda: "2.35")
    monkeypatch.setattr(compat, "_py_version", lambda: py)
    return compat.check_compat(**kw)


def pkgs(report, level):
    return [f.package for f in report.findings if f.level == level]


@pytest.mark.parametrize("acc,warned", [("0.34.2", True), ("1.0.0", False), ("1.10.1", False)])
def test_g15_old_accelerate_under_transformers5_is_a_warning(monkeypatch, acc, warned):
    r = run(monkeypatch, {"transformers": "5.17.0", "accelerate": acc})
    assert ("accelerate" in pkgs(r, "warn")) is warned


def test_g15_old_accelerate_under_transformers4_is_fine(monkeypatch):
    assert "accelerate" not in pkgs(run(monkeypatch, {"transformers": "4.57.1", "accelerate": "0.34.2"}), "warn")


def test_g15_the_lmeval_extra_has_an_accelerate_floor():
    import pathlib
    import re
    toml = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^lmeval\s*=.*"accelerate>=1\.0"', toml, re.M)


SGL = {"sglang": "0.5.20", "torch": "2.13.0+cu130"}


def test_g17_sglang_on_python_313_is_a_warning(monkeypatch):
    r = run(monkeypatch, SGL, py=(3, 13, 1))
    warn = [f for f in r.findings if f.package == "python" and f.level == "warn"]
    assert r.ok and warn and "3.12" in warn[0].message


@pytest.mark.parametrize("py", [(3, 10, 0), (3, 12, 7)])
def test_g17_sglang_on_python_312_and_below_is_fine(monkeypatch, py):
    assert "python" not in pkgs(run(monkeypatch, SGL, py=py), "warn")


def test_g17_python_313_without_sglang_is_fine(monkeypatch):
    assert "python" not in pkgs(run(monkeypatch, {"torch": "2.11.0"}, py=(3, 13, 1)), "warn")


def _fake_torch(monkeypatch, audio_error=None):
    fake = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                           cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "L4"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    if audio_error is None:
        monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())
    else:
        import builtins
        real = builtins.__import__

        def imp(name, *a, **k):
            if name == "torchaudio":
                raise RuntimeError(audio_error)
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", imp)


def test_g19_a_torchaudio_cuda_mismatch_is_an_error(monkeypatch):
    _fake_torch(monkeypatch, "Detected that PyTorch and TorchAudio were compiled with different CUDA versions. "
                             "PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8.")
    r = run(monkeypatch, {"vllm": "0.30.0", "torch": "2.11.0+cu130", "torchaudio": "2.9.0+cu128"}, check_cuda=True)
    assert "torchaudio" in pkgs(r, "error") and not r.ok


def test_g19_a_matching_torchaudio_is_fine(monkeypatch):
    _fake_torch(monkeypatch)
    r = run(monkeypatch, {"vllm": "0.30.0", "torch": "2.11.0+cu130", "torchaudio": "2.11.0+cu130"}, check_cuda=True)
    assert "torchaudio" not in pkgs(r, "error")


def test_g19_no_torchaudio_is_not_checked(monkeypatch):
    _fake_torch(monkeypatch, "compiled with different CUDA versions")
    r = run(monkeypatch, {"vllm": "0.30.0", "torch": "2.11.0+cu130"}, check_cuda=True)
    assert "torchaudio" not in pkgs(r, "error")
