"""lexsi_provenance.json round trip (INTEROP §2): read from an ``hf:<dir>`` model
and a dataset folder into run metadata, shown in reports, written next to saved
results with the inputs embedded, uploaded by push_to_hub, surfaced by the
Lexsi importers. CPU only, tiny random ``cohere`` built in-test."""

from __future__ import annotations

import json
import sys
import types

import pytest

import auditkit as ak
from auditkit.agent_eval.lexsi import curatorkit_manifest, safetune_bench_row
from auditkit.provenance import PROVENANCE_FILE, read_provenance
from auditkit.report import RunResult
from auditkit.report_format import Report


def _prov(library, method, base_model=None, inputs=()):
    return {"schema": "lexsi.provenance/1", "library": library, "version": "0.2.0", "git_sha": None,
            "created_at": "2026-09-28T12:00:00Z", "base_model": base_model, "method": method,
            "inputs": list(inputs), "params": {}}


DATA_PROV = _prov("curatorkit", "Curator.run")
MODEL_PROV = _prov("safetune", "harden.SafeGrad", base_model="CohereLabs/aya-expanse-8b",
                   inputs=[{"kind": "dataset", "ref": "./curated_out", "config": "sft_alpaca",
                            "provenance": DATA_PROV}])


def test_reader_tolerates_missing_and_bad_files(tmp_path):
    assert read_provenance(tmp_path) is None
    assert read_provenance("CohereLabs/aya-expanse-8b") is None
    (tmp_path / PROVENANCE_FILE).write_text("{not json")
    assert read_provenance(tmp_path) is None
    (tmp_path / PROVENANCE_FILE).write_text(json.dumps(DATA_PROV))
    assert read_provenance(tmp_path) == read_provenance(tmp_path / PROVENANCE_FILE) == DATA_PROV


def test_hf_model_and_dataset_provenance_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from auditkit.model.hf_gen import HFGenModel
    from tests.test_cohere_aya import SMALL, _ids, _tokenizer
    import torch
    import transformers

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    model_dir = tmp_path / "hardened"
    tok = _tokenizer()
    torch.manual_seed(0)
    transformers.CohereForCausalLM(transformers.CohereConfig(**SMALL, **_ids(tok))).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    (model_dir / PROVENANCE_FILE).write_text(json.dumps(MODEL_PROV))
    data_dir = tmp_path / "curated_out"
    data_dir.mkdir()
    (data_dir / "eval.jsonl").write_text(json.dumps({"input": "what is two plus two", "target": "a"}) + "\n")
    (data_dir / PROVENANCE_FILE).write_text(json.dumps(DATA_PROV))

    r = ak.evaluate(str(data_dir / "eval.jsonl"), HFGenModel(model=str(model_dir), device="cpu"),
                    ["exact_match"], config=ak.RunConfig(max_tokens=4))
    ds, model = r.metadata["inputs"]
    assert (ds["kind"], ds["provenance"]) == ("dataset", DATA_PROV)
    assert (model["kind"], model["ref"], model["provenance"]) == ("model", str(model_dir), MODEL_PROV)

    # shown in reports
    for text in (r.summary(), str(Report(r)), Report(r).markdown()):
        assert "safetune 0.2.0 harden.SafeGrad, base CohereLabs/aya-expanse-8b" in text
        assert "curatorkit 0.2.0 Curator.run" in text

    # written next to the results, lineage chained, and the run reloads with it
    out = tmp_path / "results"
    out.mkdir()
    r.save(str(out / "run.json"))
    own = json.loads((out / PROVENANCE_FILE).read_text())
    assert own["schema"] == "lexsi.provenance/1" and own["library"] == "auditkit"
    assert own["base_model"] == str(model_dir) and own["inputs"] == r.metadata["inputs"]
    assert own["inputs"][1]["provenance"]["inputs"][0]["provenance"] == DATA_PROV
    assert RunResult.load(str(out / "run.json")).metadata == r.metadata

    # a cache hit still reports this call's inputs
    (model_dir / PROVENANCE_FILE).unlink()
    monkeypatch.setattr(HFGenModel, "generate", lambda *a, **k: 1 / 0)  # a cache miss would fail
    again = ak.evaluate(str(data_dir / "eval.jsonl"), HFGenModel(model=str(model_dir), device="cpu"),
                        ["exact_match"], config=ak.RunConfig(max_tokens=4))
    assert again.metadata["inputs"][1]["provenance"] is None


def test_push_to_hub_uploads_provenance(monkeypatch):
    datasets = pytest.importorskip("datasets")
    uploads = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, **k):
            data = path_or_fileobj
            if isinstance(data, str):
                with open(data, "rb") as fh:
                    data = fh.read()
            uploads[path_in_repo] = data

    monkeypatch.setattr(datasets.Dataset, "push_to_hub", lambda self, *a, **k: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    r = RunResult(run_id="r", fingerprint="f", stats={}, predictions=[], headline={},
                  metadata={"inputs": [{"kind": "model", "ref": "./hardened", "provenance": MODEL_PROV}]})
    r.push_to_hub("org/run", token="x")
    pushed = json.loads(uploads[PROVENANCE_FILE])
    assert pushed["inputs"][0]["provenance"] == MODEL_PROV
    assert "harden.SafeGrad" in uploads["README.md"].decode()


def test_lexsi_importers_surface_provenance(tmp_path):
    ev = curatorkit_manifest({"stage_counts": {}, "provenance": DATA_PROV})
    assert ev.provenance == DATA_PROV and ev.to_dict()["provenance"] == DATA_PROV
    (tmp_path / PROVENANCE_FILE).write_text(json.dumps(MODEL_PROV))
    ev = safetune_bench_row({"prompt": "x", "response": "no", "score": 0.0},
                            provenance=read_provenance(tmp_path))
    assert ev.to_dict()["provenance"]["method"] == "harden.SafeGrad"
    assert safetune_bench_row({"prompt": "x"}).provenance is None
