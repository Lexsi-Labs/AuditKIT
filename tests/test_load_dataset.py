"""A CuratorKIT-style export folder + config name is a dataset (INTEROP §1):
``datasets.load_dataset(<dir>, <config>)`` layout, README ``configs:`` mapping
each format to train/validation files; non-data files are never splits."""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.errors import AuditKitError
from auditkit.loaders import _row_sample

PROV = {"schema": "lexsi.provenance/1", "library": "curatorkit", "version": "0.4.0", "git_sha": None,
        "created_at": "2026-09-28T12:00:00Z", "base_model": None, "method": "Curator.run",
        "inputs": [], "params": {}}
README = """---
configs:
- config_name: sft_alpaca
  data_files:
  - split: train
    path: sft_alpaca/train.jsonl
  - split: validation
    path: sft_alpaca/validation.jsonl
- config_name: dpo
  data_files:
  - split: train
    path: dpo/train.jsonl
  - split: validation
    path: dpo/validation.jsonl
---
# curated
"""


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def curated(tmp_path):
    d = tmp_path / "curated_out"
    _jsonl(d / "sft_alpaca" / "train.jsonl", [{"instruction": "t", "input": "", "output": "T"}])
    _jsonl(d / "sft_alpaca" / "validation.jsonl", [
        {"instruction": "capital of france?", "input": "", "output": "Paris"},
        {"instruction": "add", "input": "2 and 3", "output": "5"}])
    _jsonl(d / "dpo" / "train.jsonl", [{"prompt": "p1", "chosen": "c1", "rejected": "r1"},
                                        {"prompt": "p2", "chosen": "c2", "rejected": "r2"}])
    _jsonl(d / "dpo" / "validation.jsonl", [{"prompt": "pv", "chosen": "cv", "rejected": "rv"}])
    _jsonl(d / "rejected.jsonl", [{"instruction": "junk", "rejection_reason": "dup"}])
    (d / "manifest.json").write_text(json.dumps({"stage_counts": {}, "provenance": PROV}))
    (d / "lexsi_provenance.json").write_text(json.dumps(PROV))
    (d / "README.md").write_text(README)
    return d


def test_folder_and_config(curated):
    pytest.importorskip("datasets")
    sc = ak.load_dataset(str(curated), "sft_alpaca")  # no test split -> validation
    samples = sc.samples()
    assert [(s.input, s.target) for s in samples] == [("capital of france?", "Paris"), ("add\n\n2 and 3", "5")]
    assert sc.metadata["lexsi_input"] == {"kind": "dataset", "ref": str(curated), "config": "sft_alpaca",
                                          "split": "validation", "provenance": PROV}


def test_evaluate_takes_folder_config_and_split(curated, monkeypatch, tmp_path):
    pytest.importorskip("datasets")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    r = ak.evaluate(str(curated), lambda ps: ["c1" for _ in ps], ["exact_match"],
                    dataset_config="dpo", dataset_split="train")
    assert [p.expected for p in r.predictions] == ["c1", "c2"]
    assert r.headline["exact_match"] == 0.5
    (ds,) = r.metadata["inputs"]
    assert ds["config"] == "dpo" and ds["provenance"] == PROV


def test_single_jsonl_file_reads_its_folder_provenance(curated):
    sc = ak.load_dataset(str(curated / "dpo" / "train.jsonl"))
    assert [s.target for s in sc.samples()] == ["c1", "c2"]
    assert sc.metadata["lexsi_input"]["provenance"] is None  # dpo/ has none; the export root does
    sc = ak.load_dataset(str(curated / "rejected.jsonl"), input_col="instruction")
    assert sc.metadata["lexsi_input"]["provenance"] == PROV


def test_row_layouts():
    sg = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
    assert (_row_sample(sg, None, None).input, _row_sample(sg, None, None).target) == ("hi", "hello")
    s = _row_sample({"prompt": "p", "responses": ["a"], "rewards": [1.0]}, None, None)  # grpo
    assert (s.input, s.target, s.metadata) == ("p", None, {"responses": ["a"], "rewards": [1.0]})
    s = _row_sample({"question": "q", "answer": "a", "tools": [{"type": "function"}]}, None, None)
    assert (s.input, s.target, s.tools) == ("q", "a", [{"type": "function"}])
    with pytest.raises(ValueError, match="input_col"):
        _row_sample({"text": "x"}, None, None)


def test_unknown_name_still_errors():
    with pytest.raises(AuditKitError, match="Unknown dataset"):
        ak.evaluate("no_such_scenario", lambda ps: ps)
