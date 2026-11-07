from __future__ import annotations

import builtins
import importlib.util

import pytest

from auditkit.errors import ExtraNotInstalled
from auditkit.model.hf_gen import HFGenModel


def _fail_transformers_import(monkeypatch, message):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError(message)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_broken_transformers_import_surfaces_the_real_error(monkeypatch):
    pytest.importorskip("torch")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("needs transformers installed")
    _fail_transformers_import(monkeypatch, "PIL.Image has no attribute Resampling")
    with pytest.raises(ImportError, match="Resampling") as exc:
        HFGenModel("gpt2")._ensure_pipeline()
    assert not isinstance(exc.value, ExtraNotInstalled)


def test_missing_transformers_still_says_install_the_extra(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "transformers" else real_find_spec(name, *a, **k))
    _fail_transformers_import(monkeypatch, "No module named 'transformers'")
    pytest.importorskip("torch")
    with pytest.raises(ExtraNotInstalled):
        HFGenModel("gpt2")._ensure_pipeline()
