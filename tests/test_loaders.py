"""Tests for T2 dataset loaders (load_hf, load_croissant)."""

from __future__ import annotations

import importlib.util

import pytest

from auditkit.errors import ExtraNotInstalled


def _importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# ============================================================================
# load_hf
# ============================================================================

class TestLoadHf:
    @pytest.mark.skipif(
        _importable("datasets"),
        reason="datasets is installed; test designed for missing-datasets scenario",
    )
    def test_not_installed_raises_extra_not_installed(self):
        from auditkit.loaders import load_hf
        with pytest.raises(ExtraNotInstalled):
            load_hf("some/dataset")

    def test_is_callable(self):
        from auditkit.loaders import load_hf
        assert callable(load_hf)

    @pytest.mark.skipif(not _importable("datasets"), reason="needs the datasets package")
    def test_real_dataset_load(self):
        # Real HF Hub load -- needs network access. rag-mini-wikipedia's
        # `question-answer` config is small (918 rows) and already used
        # for real elsewhere in this repo (examples/applications/06, 07).
        from auditkit.loaders import load_hf
        samples = load_hf(
            "rag-datasets/rag-mini-wikipedia", name="question-answer", split="test[:5]",
            input_col="question", target_col="answer",
        )
        assert len(samples) == 5
        assert all(isinstance(s.input, str) and s.input for s in samples)
        assert all(isinstance(s.target, str) and s.target for s in samples)


# ============================================================================
# load_croissant
# ============================================================================

class TestLoadCroissant:
    @pytest.mark.skipif(
        _importable("mlcroissant"),
        reason="mlcroissant is installed; test designed for missing-mlcroissant scenario",
    )
    def test_not_installed_raises_extra_not_installed(self):
        from auditkit.loaders import load_croissant
        with pytest.raises(ExtraNotInstalled):
            load_croissant("some/path", "some_record_set")

    def test_is_callable(self):
        from auditkit.loaders import load_croissant
        assert callable(load_croissant)

    @pytest.mark.skipif(not _importable("mlcroissant"), reason="needs the mlcroissant package")
    def test_real_dataset_load(self):
        # Real Croissant (JSON-LD) load via HuggingFace's croissant endpoint --
        # needs network access. Same rag-mini-wikipedia dataset as the load_hf
        # test above, loaded through the Croissant format instead.
        from auditkit.loaders import load_croissant
        samples = load_croissant(
            "https://huggingface.co/api/datasets/rag-datasets/rag-mini-wikipedia/croissant",
            "question-answer", input_col="question", target_col="answer",
        )
        assert len(samples) == 918
        assert all(isinstance(s.input, str) and s.input for s in samples[:5])
        assert all(isinstance(s.target, str) and s.target for s in samples[:5])
