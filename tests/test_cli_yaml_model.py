"""`auditkit eval --config FILE` takes the model from the YAML (docs/cli.md); --model is still
required when neither the flag nor the file gives one."""
from __future__ import annotations

import json

import pytest

from auditkit.cli import main


def test_the_documented_yaml_example_runs_with_the_model_from_the_file(tmp_path):
    out = tmp_path / "results.json"
    cfg = tmp_path / "auditkit.yaml"
    cfg.write_text(f"model: echo\ntemperature: 0.0\nmax_tokens: 256\nseed: 42\nconcurrency: 8\n"
                   f"prompts:\n  - \"What is the capital of France?\"\n  - \"Explain quantum computing.\"\n"
                   f"output: {out}\ntags: [test, v1]\n")
    main(["eval", "--config", str(cfg)])
    assert out.exists() and json.loads(out.read_text())


def test_a_flag_still_wins_over_the_file(tmp_path):
    out = tmp_path / "r.json"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"model: not-a-backend:x\nprompts: [\"hi\"]\noutput: {out}\n")
    main(["eval", "--config", str(cfg), "--model", "echo"])
    assert out.exists()


def test_no_model_anywhere_is_still_an_error(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("prompts: [\"hi\"]\n")
    with pytest.raises(SystemExit) as e:
        main(["eval", "--config", str(cfg)])
    assert e.value.code == 2 and "--model is required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["eval"])
