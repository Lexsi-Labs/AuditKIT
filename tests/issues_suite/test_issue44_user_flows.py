"""#44 the way a user meets it: a tool-call dataset kept in a spreadsheet or JSONL, reloaded,
and run again. The references are real BFCL answers (single-valued ones, as a user would write).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

import auditkit as ak
from auditkit.agent_eval.types import AgentCase

from .conftest import SRC


def user_rows(bfcl_root, n=6):
    """(question, reference) pairs from real BFCL parallel answers, first accepted value of each arg."""
    rows = []
    for s in ak.load_bfcl(str(bfcl_root / "BFCL_v3_parallel.json"), limit=n):
        calls = [{"name": c["name"], "arguments": {k: next(a for a in v if a != "") for k, v in c["arguments"].items()
                                                   if any(a != "" for a in v)}}
                 for c in s.expected_tool_calls[0]]
        rows.append((s.id, s.input, [calls], s.tools))
    return rows


class Counting:
    """A deterministic 'model' that answers every tool question with the reference calls, counting calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def __call__(self, prompts):
        self.calls += len(prompts)
        # prompt mode: the question is the end of the rendered prompt
        return ["".join(f"<tool_call>{json.dumps(c)}</tool_call>"
                        for c in next(v for q, v in self.answers.items() if p.endswith(q))) for p in prompts]


def as_samples(rows, ref_of):
    return [ak.Sample(id=i, input=q, tools=t, expected_tool_calls=ref_of(ref)) for i, q, ref, t in rows]


def test_the_spreadsheet_copy_of_a_dataset_reuses_the_cached_run(bfcl_root, tmp_path):
    rows = user_rows(bfcl_root)
    model = Counting({q: ref[0] for _, q, ref, _ in rows})
    model.__name__ = "counting"
    first = ak.evaluate(as_samples(rows, lambda r: r), model=model, adapter=ak.ToolCallAdapter(mode="prompt"),
                        scorers=[ak.ToolCallF1()])
    # the user exports to CSV (Excel, pandas, Sheets...) and comes back the next day
    path = tmp_path / "dataset.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "input", "tools", "expected_tool_calls"])
        for i, q, ref, t in rows:
            w.writerow([i, q, json.dumps(t), json.dumps(ref, indent=2)])      # pretty-printed cells, too
    with path.open(newline="", encoding="utf-8") as f:
        reloaded = [ak.Sample(id=r["id"], input=r["input"], tools=json.loads(r["tools"]),
                              expected_tool_calls=r["expected_tool_calls"]) for r in csv.DictReader(f)]
    assert all(isinstance(s.expected_tool_calls, str) for s in reloaded)
    second = ak.evaluate(reloaded, model=model, adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
    assert model.calls == len(rows), "the reloaded dataset must hit the cache, not call the model again"
    assert second.headline == first.headline and first.headline["tool_call_f1"] == 1.0


def test_jsonl_with_the_reference_as_a_string_field(bfcl_root, tmp_path):
    # two exports of one dataset: one tool writes the field as JSON, another as a JSON string
    rows = user_rows(bfcl_root, 3)
    names = []
    for encode in (lambda r: r, json.dumps):
        path = tmp_path / "data.jsonl"
        path.write_text("".join(json.dumps({"id": i, "input": q, "tools": t, "expected_tool_calls": encode(ref)}) + "\n"
                                for i, q, ref, t in rows))
        names.append(ak.ListScenario(ak.load_jsonl(str(path))).name)
    assert names[0] == names[1]


def test_a_different_reference_is_still_a_different_dataset(bfcl_root):
    rows = user_rows(bfcl_root, 2)
    edited = [(i, q, [[{**c, "arguments": {**c["arguments"], "_edited": 1}} for c in ref[0]]], t) for i, q, ref, t in rows]
    assert ak.ListScenario(as_samples(rows, json.dumps)).name != ak.ListScenario(as_samples(edited, json.dumps)).name


def test_expect_no_call_as_a_string_is_not_a_missing_reference():
    s = lambda ref: ak.ListScenario([ak.Sample(id="q", input="What is 2+2?", expected_tool_calls=ref)]).name
    assert s("[]") == s([]) != s(None)


def test_a_broken_cell_fails_its_sample_with_a_clear_error_not_the_fingerprint():
    bad = ak.Sample(id="bad", input="Weather in Paris?", expected_tool_calls='[[{"name": "get_weather", "arguments": {"city": "Paris"}]',
                    actual_output="", actual_trace={"tool_calls": []})
    ak.ListScenario([bad]).name                                             # hashing never raises
    r = ak.evaluate([bad], model="precomputed", scorers=[ak.ToolCallF1()])
    assert r.failed_count == 1
    assert any("unparseable" in json.dumps(e) for e in r.errors)


def test_agent_cases_from_a_config_file_digest_like_the_python_ones(bfcl_root):
    rows = user_rows(bfcl_root, 2)
    for i, q, ref, _ in rows:
        assert AgentCase(id=i, task=q, reference_turns=json.dumps(ref)).digest() == \
            AgentCase(id=i, task=q, reference_turns=ref).digest()


def test_agent_dry_run_says_an_empty_reference_is_a_reference(tmp_path):
    cfg = {"mode": "deployed", "agent": "http://127.0.0.1:9/agent", "scorers": ["tool_call_f1"],
           "cases": [{"id": "no-call", "task": "What is 2+2?", "reference_turns": []},
                     {"id": "no-ref", "task": "Say hi"}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(cfg))
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-m", "auditkit", "agent", "eval", "--config", str(path), "--dry-run"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = {l.split(":")[0].strip("- ").strip(): l for l in out.stdout.splitlines() if l.strip().startswith("- ")}
    assert "refs=yes" in lines["no-call"] and "refs=no" in lines["no-ref"]
