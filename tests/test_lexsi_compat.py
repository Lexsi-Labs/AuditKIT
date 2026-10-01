"""The Lexsi cross-product COMPATIBILITY GATE (integration note, X1).

For each of the five producer libraries: import one real artifact shape, assert
the fields that survive, the granularity, the join keys that are present and the
exact join keys that are MISSING (the lost-field check), then join four of them
to ONE synthetic ``AgentEpisode`` -- proving each join needs an EXPLICITLY
supplied id and fails with a named reason without it.

Two kinds of fixture, both with benign placeholder content only:

- ``tests/fixtures/lexsi_real/<producer>/``: files written by each library's own
  serializer at the pinned commit (``REVISIONS``) on placeholder input. The only
  edit is that CuratorKIT's absolute input paths were shortened to ``inputs/...``;
  shapes and every other value are untouched.
- the hand-written dicts below, which mirror those real shapes and are used for
  join-contract edge cases.

This gate tests field names, ids, join keys and lost fields, never payload content.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import auditkit
from auditkit.agent_eval import (
    INFERRED,
    OBSERVED,
    UNAVAILABLE,
    AgentEpisode,
    AgentEvent,
    episodes_from_agenttune,
)
from auditkit.agent_eval.lexsi import (
    REVISIONS,
    SourceEvidence,
    aligntune_audit_report,
    aligntune_regression_report,
    circuitkit_faithfulness_report,
    curatorkit_data_sample,
    curatorkit_manifest,
    curatorkit_trainer_export_row,
    join_evidence,
    lineage_sidecar,
    safetune_audit_event,
    safetune_bench_row,
)

# --------------------------------------------------------------------------
# fixtures: real artifact STRUCTURE, benign placeholder CONTENT
# --------------------------------------------------------------------------
# AgentTune @36d4724 -- TrajectoryStore.export_jsonl row (flat-action path).
# trajectory_id is a store-minted uuid4; reward_components are the phase1 names;
# query is str(args dict); n_tool_calls comes from rollout metadata (0 here).
_STORE_TID = "ad6a2526-0742-4340-8621-733be2ea5656"
AGENTTUNE_EXPORT_ROW = {
    "trajectory_id": _STORE_TID,
    "run_id": "run-0001",
    "question": "<benign placeholder prompt>",
    "gold_answer": "<benign placeholder target>",
    "final_answer": "<answer>benign placeholder response</answer>",
    "reward": 0.5,
    "n_tool_calls": 0,
    "has_answer_tag": 1,
    "condition": "m1",
    "model": "<benign placeholder model>",
    "training_step": None,
    "timestamp": 1758801600.0,
    "reward_components": {"format": 0.0, "termination": 0.0, "search_usage": 0.0,
                          "correctness": 1.0},
    "metadata": {},
    "steps": [{"step_id": 1, "trajectory_id": _STORE_TID, "step_number": 0,
               "thought": "<benign placeholder reasoning>", "action_name": "search_corpus",
               "action_args_json": '{"query": "<benign placeholder query>"}',
               "observation": "<benign placeholder observation>", "reward": 0.25,
               "is_tool_step": 1, "is_terminal": 0},
              {"step_id": 2, "trajectory_id": _STORE_TID, "step_number": 1,
               "thought": "", "action_name": "", "action_args_json": "{}",
               "observation": "<benign placeholder response>", "reward": None,
               "is_tool_step": 0, "is_terminal": 0}],
    "tool_calls": [{"tool_call_id": 1, "trajectory_id": _STORE_TID, "step_number": 0,
                    "tool_name": "search_corpus",
                    "query": "{'query': '<benign placeholder query>'}",
                    "result": "<benign placeholder observation>"}],
}

# AlignTune @a5708bc -- AuditReport.to_json().
ALIGNTUNE_AUDIT_REPORT = {
    "reward_hacking": 0.12, "sycophancy": 0.4, "refusal_collapse": 0.05,
    "verbosity_gain": 0.0, "timestamp": "2026-09-25T18:30:00.123456",
    "avg_response_tokens": 180.0, "repetition_ratio": 0.15,
    "unique_token_ratio": 0.82, "degenerate_rate": 0.04,
}

# AlignTune @a5708bc -- RegressionReport.to_json().
ALIGNTUNE_REGRESSION_REPORT = {
    "baseline": {
        "artifact": {"name": "baseline", "path": "/models/placeholder-fp16",
                     "format": "hf", "metadata": {}},
        "audit_report": dict(ALIGNTUNE_AUDIT_REPORT),
        "eval_results": {"accuracy": 0.851, "bleu": 0.452},
        "duration_seconds": 182.4,
    },
    "variants": [{
        "artifact": {"name": "Q4_K_M", "path": "/models/placeholder-Q4_K_M.gguf",
                     "format": "gguf", "metadata": {}},
        "audit_report": {**ALIGNTUNE_AUDIT_REPORT, "verbosity_gain": 0.15,
                         "avg_response_tokens": 207.0},
        "eval_results": {"accuracy": 0.83, "bleu": 0.44},
        "duration_seconds": 96.7,
    }],
    "deltas": {"Q4_K_M": {"reward_hacking_delta": 0.0, "sycophancy_delta": 0.0,
                          "refusal_collapse_delta": 0.0, "verbosity_gain_delta": 0.15,
                          "accuracy_delta": -0.0247, "bleu_delta": -0.0265}},
    "verdicts": {"Q4_K_M": "PASS"},
    "thresholds": {"max_refusal_collapse_delta": 0.1, "max_sycophancy_delta": 0.15,
                   "max_verbosity_gain_delta": 0.5, "max_eval_metric_relative_drop": 0.05,
                   "alignment_metrics_failure": "fail", "eval_metrics_failure": "fail"},
    "timestamp": "2026-09-25T18:30:00.123456",
}

# SafeTune @a99d08a -- SafetyAuditEvent.as_dict (redact_audit default True).
SAFETUNE_AUDIT_EVENT = {
    "tenant_id": "placeholder-tenant",
    "decision": "block",
    "reason": "blocked_response_category:placeholder_category",
    "prompt": "[REDACTED]",
    "response": "[REDACTED]",
    "metadata": {"category": "placeholder_category",
                 "matched_keyword": "placeholder_keyword"},
    "timestamp": "2026-09-25T15:23:27.901335+00:00",
}

# SafeTune @a99d08a -- write_jsonl scored benchmark row.
SAFETUNE_BENCH_ROW = {
    "prompt": "<benign placeholder prompt>",
    "response": "<benign placeholder response>",
    "score": 0.0,
    "refused": True,
    "bench": "placeholder_bench",
}

def _step(name: str, version: str, notes: dict, ts: str = "2026-09-25T14:03:01.123456") -> dict:
    return {"step_name": name, "step_version": version, "timestamp": ts,
            "config_hash": "48a1cb5788402c4d", "notes": notes}


# Every tabular connector's first step is "Connector"; the row locator is its
# notes.line_number. Nothing writes metadata.row, and only PDF rows write source_file.
_CONNECTOR = _step("Connector", "0.2.0", {
    "source_uri": "inputs/placeholder_raw.jsonl", "line_number": 2,
    "detected_format": "alpaca", "detection_confidence": "high",
    "column_map": {"instruction": "instruction", "output": "output", "context": "input"},
    "field_mapping_applied": {}, "unrecognized_cols": ["category", "id"]})
_KEPT_TAIL = [
    _step("SchemaGate", "0.2.0", {"passed": True, "task_type": "instruction_following"}),
    _step("ExactDeduplicator", "0.1.0", {"exact_duplicates_removed": 0, "surviving_count": 2}),
    _step("TextCleaner", "0.1.0", {"transforms_applied": ["strip_html"],
                                   "fields_cleaned": ["instruction", "input", "output"]}),
]

# CuratorKIT @2aab65a -- DataSample full serialization (JSONL input, kept row).
CURATORKIT_DATA_SAMPLE = {
    "id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301", "source_uri": "inputs/placeholder_raw.jsonl",
    "instruction": "benign placeholder instruction", "input": "",
    "output": "benign placeholder output", "chosen": "", "rejected": "", "label": None,
    "responses": [], "reward_scores": [], "task_type": "instruction_following",
    "metadata": {"category": "placeholder"},
    "provenance_chain": [_CONNECTOR] + _KEPT_TAIL,
}

# CuratorKIT @2aab65a -- rejected.jsonl row (SchemaGate rejection; space-separated
# timestamps, rejected before TextCleaner runs).
CURATORKIT_REJECTED_SAMPLE = {
    **CURATORKIT_DATA_SAMPLE,
    "id": "9c1e7a55-0000-4b6a-8a11-abcdef012345",
    "output": "",
    "provenance_chain": [
        {**_CONNECTOR, "timestamp": "2026-09-25 14:03:01.123456",
         "notes": {**_CONNECTOR["notes"], "line_number": 3}},
        _step("SchemaGate", "0.2.0", {"passed": False, "rejection_reason": "missing_field:output",
                                      "task_type": "instruction_following"},
              ts="2026-09-25 14:03:01.123999")],
    "rejection_reason": "missing_field:output",
    "rejecting_step": "SchemaGate",
    "diagnosis": None,
}

# CuratorKIT @2aab65a -- checkpoint stage row (.checkpoints/TextCleaner.jsonl) from
# a PDF run: the chain starts at PDFReader and there is no line locator.
CURATORKIT_CHECKPOINT_SAMPLE = {
    **CURATORKIT_DATA_SAMPLE,
    "id": "c0ffee00-1111-4222-8333-444455556666",
    "source_uri": "inputs/placeholder.pdf",
    "instruction": "",
    "task_type": "language_modeling",
    "metadata": {"page": 3, "parent_heading": "Placeholder heading", "chunk_index": 12,
                 "content_type": "text", "source_file": "inputs/placeholder.pdf"},
    "provenance_chain": [_step("PDFReader", "0.2.0", {
        "source_file": "inputs/placeholder.pdf", "page": 3, "chunk_index": 12,
        "parent_heading": "Placeholder heading", "chunk_strategy": "heading",
        "overlap_tokens": 50, "ocr": False, "is_table": False,
        "output_mode": "chunk"})] + _KEPT_TAIL,
}

# CuratorKIT @2aab65a -- manifest.json of a JSONL run (source_files is empty).
CURATORKIT_MANIFEST = {
    "pipeline_config_hash": "4e732c3bf6e9acb0", "run_timestamp": "2026-09-25T15:27:03.576492Z",
    "source_files": [],
    "stage_counts": {"JSONLReader": {"output_count": 98, "rejected_count": 2},
                     "SchemaGate": {"input_count": 98, "output_count": 98,
                                    "probe_recovered": 0, "rejected_count": 0},
                     "AlpacaExporter": {"exported_count": 98}},
    "rejected_breakdown": {"missing_field:output": 2},
    "dedup_stats": {"exact_duplicates_removed": 0},
    "minhash_threshold": None, "sampler_stats": None,
    "token_stats": {"total_prompt_tokens": 0, "total_completion_tokens": 0,
                    "total_tokens": 0},
    "wall_clock_seconds": 12.346,
    "tool_versions": {"curatorkit": "0.1.6", "python": "3.11.15"},
    "diversity_stats": None, "diagnostic_stats": None, "diagnostic_files": [],
}

# CuratorKIT @2aab65a -- the two common trainer exports (lineage dropped).
CURATORKIT_ALPACA_ROW = {"instruction": "<benign placeholder instruction>", "input": "",
                         "output": "<benign placeholder output>"}
CURATORKIT_SHAREGPT_ROW = {"conversations": [
    {"from": "human", "value": "<benign placeholder instruction>"},
    {"from": "gpt", "value": "<benign placeholder output>"}]}

# CircuitKIT @6430809 -- FaithfulnessReport.to_json().
CIRCUITKIT_REPORT = {
    "patching_score": 0.87, "ablation_score": 0.79,
    "stability": {"mean_jaccard": 0.73, "std_jaccard": 0.05, "mean_dice": 0.84,
                  "std_dice": 0.04, "n_runs": 5, "n_stable_nodes": 12},
    "robustness": {"paraphrase": {"corruption_variant": "paraphrase",
                                  "original_score": 0.87, "variant_score": 0.8,
                                  "delta": 0.07, "relative_drop": 0.08,
                                  "robustness_ratio": 0.92, "status": "valid"}},
    "baseline_comparison": {"circuit_score": 0.87,
                            "baselines": {"random": {"score": 0.42, "percentage": 48.3,
                                                     "improvement": 2.07,
                                                     "improvement_valid": True}},
                            "summary": "<benign placeholder summary>"},
    "generalization": {"source_task": "placeholder_task",
                       "target_task": "placeholder_task_b",
                       "source_score": 0.87, "target_score": 0.64,
                       "transfer_ratio": 0.74, "transfer_delta": 0.23,
                       "relative_transfer_drop": 0.26, "normalized": True},
    "intervention_reliability": {"r1_seed_consistency": 0.88, "r2_effect_magnitude": 0.55,
                                 "r3_effect_variance": 0.79, "reliability_index": 0.72,
                                 "n_seeds": 3, "per_seed": [{"seed": 0, "delta": 0.31}]},
    "metadata": {"patching_raw_ratio": 0.91, "ablation_raw_ratio": 0.83,
                 "algorithm": "eap", "model": "placeholder-model",
                 "task": "placeholder_task", "level": "node", "scope": "component",
                 "sparsity": 0.3, "pillars_computed": ["patching", "ablation"],
                 "timestamp": 1758830400.0, "total_duration_seconds": 142.7,
                 "per_pillar_duration_seconds": {"patching": 3.2}},
}

# The caller-supplied ids the integration must capture at emission time. None of
# these live in any producer artifact -- that is the whole point of the gate.
EXPLICIT_EPISODE_ID = "ep-0007"
EXPLICIT_CALL_ID = "call-0001"
EXPLICIT_MODEL_DIGEST = "sha256:" + "ab" * 32


def _episode() -> AgentEpisode:
    """ONE synthetic joined episode: an AuditKit-side recorded run with ids."""
    return AgentEpisode(
        events=[
            AgentEvent(index=0, role="assistant", type="tool_call",
                       payload={"name": "search_corpus",
                                "arguments": {"query": "<benign placeholder query>"}},
                       call_id=EXPLICIT_CALL_ID, turn_id=0),
            AgentEvent(index=1, role="tool", type="tool_result",
                       payload={"output": "<benign placeholder observation>"},
                       call_id=EXPLICIT_CALL_ID, turn_id=0),
        ],
        source_format="synthetic", source_id=EXPLICIT_EPISODE_ID, case_id="case-0001",
        final_answer="<benign placeholder response>",
    )


# --------------------------------------------------------------------------
# producer 1/5 -- AgentTune: imports TODAY via the existing importer, with loss
# --------------------------------------------------------------------------
def test_agenttune_export_row_imports_today_with_loss(tmp_path):
    """TrajectoryStore export -> episode: store uuid kept, step detail lost."""
    path = tmp_path / "export.jsonl"
    path.write_text(json.dumps(AGENTTUNE_EXPORT_ROW) + "\n")
    (ep,) = episodes_from_agenttune(str(path))

    # captured: the store's surrogate id plus answer-level evidence
    assert ep.source_format == "agenttune_trace"
    assert ep.source_id == _STORE_TID         # a store uuid4, not the rollout's id
    assert ep.final_answer == "benign placeholder response"
    assert ep.source_reward == 0.5            # provenance, not an AuditKit outcome
    assert ep.source_scores == AGENTTUNE_EXPORT_ROW["reward_components"]
    assert ep.coverage["final_answer"] == "observed"

    # tool_name survives verbatim in the event payload, but the trace builder does
    # not read it as a call name, so there is no call-shaped turn.
    assert ep.events[0].payload["tool_name"] == "search_corpus"
    assert ep.coverage["tool_call_names"] == UNAVAILABLE
    assert ep.turns() == []
    # query is str(args dict), not structured arguments
    assert ep.coverage["tool_call_arguments"] == UNAVAILABLE
    assert ep.coverage["call_result_pairing"] == INFERRED
    dumped = json.dumps(ep.to_dict())
    for lost in ("run_id", "condition", "training_step", "thought", "action_args_json"):
        assert lost not in dumped, lost


# --------------------------------------------------------------------------
# producer 2/5 -- AlignTune: aggregate only, no case evidence, no digest
# --------------------------------------------------------------------------
def test_aligntune_audit_report_is_aggregate_with_no_case_evidence():
    ev = aligntune_audit_report(ALIGNTUNE_AUDIT_REPORT)
    assert (ev.source_product, ev.source_revision) == ("aligntune", REVISIONS["aligntune"])
    assert ev.granularity == "aggregate"
    assert ev.is_outcome is False
    assert ev.payload is ALIGNTUNE_AUDIT_REPORT          # verbatim
    for f in ("reward_hacking", "sycophancy", "refusal_collapse", "degenerate_rate"):
        assert ev.coverage[f] == "observed"
    # the honest negative: a tz-less timestamp is the ONLY id it carries
    assert list(ev.join_keys) == ["timestamp"]
    assert ev.missing_join_keys == ["episode_id", "call_id", "case_id", "artifact_name",
                                   "checkpoint_digest", "probe_set_id", "probe_id"]
    assert ev.coverage["per_case_evidence"] == UNAVAILABLE


def test_aligntune_regression_report_keeps_artifact_identity_but_no_digest():
    ev = aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT)
    assert ev.granularity == "aggregate"
    assert ev.is_outcome is False           # PASS/WARN/FAIL is the producer's verdict
    assert ev.join_keys["artifact_name"] == ["baseline", "Q4_K_M"]
    assert ev.join_keys["artifact_format"] == ["hf", "gguf"]
    assert ev.join_keys["artifact_path"] == ["/models/placeholder-fp16",
                                            "/models/placeholder-Q4_K_M.gguf"]
    assert ev.coverage["verdicts"] == "observed"
    assert ev.coverage["deltas"] == "observed"
    # no checkpoint digest anywhere, and no per-case/per-expert evidence reaches
    # disk -- so there is no case-level join, only a checkpoint-level one.
    assert ev.missing_join_keys == ["episode_id", "call_id", "case_id",
                                   "checkpoint_digest", "probe_set_id", "probe_id",
                                   "stage"]
    assert ev.coverage["per_expert_audit"] == UNAVAILABLE
    assert ev.coverage["per_case_evidence"] == UNAVAILABLE


def test_aligntune_aggregate_report_cannot_join_at_case_level():
    """No case id, no digest, and a timestamp is not identity."""
    ev = aligntune_audit_report(ALIGNTUNE_AUDIT_REPORT)
    ep = _episode()

    empty = join_evidence(ep, ev, {})
    assert empty.joined is False and "no explicit join key" in empty.reason

    stamped = join_evidence(ep, ev, {"timestamp": ALIGNTUNE_AUDIT_REPORT["timestamp"]})
    assert stamped.joined is False and "do not identify" in stamped.reason

    for key in ("case_id", "checkpoint_digest"):
        r = join_evidence(ep, ev, {key: None})
        assert r.joined is False and key in r.reason

    invented = join_evidence(ep, ev, {"trajectory_id": "ep-0007"})
    assert invented.joined is False and "unknown join key" in invented.reason
    assert "lexsi_evidence" not in ep.metadata          # nothing attached


# --------------------------------------------------------------------------
# producer 3/5 -- SafeTune: anonymous audit events, id-less bench rows
# --------------------------------------------------------------------------
def test_safetune_audit_event_has_no_episode_call_stage_or_policy_ids():
    ev = safetune_audit_event(SAFETUNE_AUDIT_EVENT)
    assert (ev.source_product, ev.source_revision) == ("safetune", REVISIONS["safetune"])
    assert ev.granularity == "per_event"
    assert ev.is_outcome is False           # a block is a policy action, not a verdict
    assert ev.join_keys == {
        "tenant_id": "placeholder-tenant", "decision": "block",
        "reason": "blocked_response_category:placeholder_category",
        "timestamp": SAFETUNE_AUDIT_EVENT["timestamp"],
    }
    assert ev.missing_join_keys == ["episode_id", "call_id", "case_id", "stage",
                                   "policy_revision"]
    assert ev.coverage["matched_keyword"] == "observed"
    assert ev.coverage["classifier_provenance"] == UNAVAILABLE


def test_safetune_bench_row_has_no_case_id():
    ev = safetune_bench_row(SAFETUNE_BENCH_ROW)
    assert ev.granularity == "per_row"
    assert ev.is_outcome is False           # score/refused stay provenance
    assert sorted(ev.join_keys) == ["bench", "prompt"]
    assert ev.missing_join_keys == ["case_id", "episode_id", "call_id", "model_name",
                                   "timestamp"]
    for f in ("score", "refused"):
        assert ev.coverage[f] == "observed"
    for lost in ("judge", "asr", "refusal_rate"):
        assert ev.coverage[lost] == UNAVAILABLE
    # the raw write_bench_jsonl dump has no bench field at all
    raw = safetune_bench_row({k: SAFETUNE_BENCH_ROW[k] for k in ("prompt", "response")})
    assert list(raw.join_keys) == ["prompt"]
    assert "bench" in raw.missing_join_keys
    # the production eval_safety scored row has no bench either
    runner = safetune_bench_row({k: SAFETUNE_BENCH_ROW[k]
                                 for k in ("prompt", "response", "score", "refused")})
    assert "bench" in runner.missing_join_keys
    # the AdvBench inline path does write its judge name
    advbench = safetune_bench_row({"prompt": "<benign placeholder prompt>",
                                   "response": "<benign placeholder response>",
                                   "refused": True, "judge": "string_match"})
    assert advbench.coverage["judge"] == "observed"


def test_safetune_pre_check_events_carry_enforced_text_not_a_model_reply():
    """Real pre_check shapes: allow has a null response; a tool block names the tool."""
    allow = safetune_audit_event({
        "tenant_id": "placeholder-tenant", "decision": "allow", "reason": "ok",
        "prompt": "[REDACTED]", "response": None, "metadata": {"category": "unknown"},
        "timestamp": "2026-09-25T18:28:24.397574+00:00"})
    assert allow.coverage["response"] == UNAVAILABLE
    tool = safetune_audit_event({
        "tenant_id": "placeholder-tenant", "decision": "block",
        "reason": "blocked_tool:placeholder_tool", "prompt": "[REDACTED]",
        "response": "Tool usage is no...[REDACTED]... policy.",
        "metadata": {"tool_name": "placeholder_tool"},
        "timestamp": "2026-09-25T18:28:24.397693+00:00"})
    assert tool.coverage["tool_name"] == "observed"
    assert tool.coverage["redaction_applied"] == UNAVAILABLE


# --------------------------------------------------------------------------
# producer 4/5 -- CuratorKIT: lineage kept on rows, dropped by trainer exports
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row,kind,chain,row_no,source_file", [
    (CURATORKIT_DATA_SAMPLE, "curatorkit_data_sample",
     ["Connector", "SchemaGate", "ExactDeduplicator", "TextCleaner"], 2, None),
    (CURATORKIT_REJECTED_SAMPLE, "curatorkit_rejected_sample",
     ["Connector", "SchemaGate"], 3, None),
    (CURATORKIT_CHECKPOINT_SAMPLE, "curatorkit_data_sample",
     ["PDFReader", "SchemaGate", "ExactDeduplicator", "TextCleaner"], None,
     "inputs/placeholder.pdf"),
])
def test_curatorkit_full_rows_keep_id_and_provenance_chain(row, kind, chain, row_no,
                                                           source_file):
    ev = curatorkit_data_sample(row)
    assert ev.artifact_kind == kind
    assert ev.granularity == "per_row"
    assert ev.join_keys["row_id"] == row["id"]
    assert ev.join_keys["source_uri"] == row["source_uri"]
    assert ev.join_keys["provenance_chain"] == chain
    assert ev.coverage["provenance_chain"] == "observed"
    assert "row_id" not in ev.missing_join_keys
    # the locator: tabular rows have a line number, PDF rows a source file
    assert ev.join_keys.get("row") == row_no
    assert ev.join_keys.get("source_file") == source_file


def test_curatorkit_rejected_row_carries_its_rejection_labels():
    ev = curatorkit_data_sample(CURATORKIT_REJECTED_SAMPLE)
    assert ev.coverage["rejection_reason"] == "observed"
    assert ev.coverage["rejecting_step"] == "observed"


def test_curatorkit_checkpoint_row_has_no_source_row_index():
    """A PDF chunk has no line number; its locator is source_file + chunk_index."""
    ev = curatorkit_data_sample(CURATORKIT_CHECKPOINT_SAMPLE)
    assert ev.missing_join_keys == ["row"]


def test_curatorkit_manifest_is_aggregate_with_no_row_lineage():
    ev = curatorkit_manifest(CURATORKIT_MANIFEST)
    assert ev.granularity == "aggregate"
    assert ev.join_keys["pipeline_config_hash"] == "4e732c3bf6e9acb0"
    # source_files is empty for tabular runs -> it becomes a missing key, honestly
    assert ev.missing_join_keys == ["row_id", "provenance_chain", "source_file_sha256",
                                    "source_files"]
    assert ev.coverage["stage_counts"] == "observed"
    # a PDF run lists {path} only; the docstring's sha256 is never emitted
    pdf = curatorkit_manifest({**CURATORKIT_MANIFEST,
                               "source_files": [{"path": "inputs/placeholder.pdf"}]})
    assert pdf.join_keys["source_files"] == ["inputs/placeholder.pdf"]
    assert "source_file_sha256" in pdf.missing_join_keys


@pytest.mark.parametrize("row,fmt", [(CURATORKIT_ALPACA_ROW, "alpaca"),
                                     (CURATORKIT_SHAREGPT_ROW, "sharegpt")])
def test_curatorkit_trainer_exports_have_zero_join_keys(row, fmt):
    """The lineage-loss case: alpaca/sharegpt rows carry NO id at all."""
    ev = curatorkit_trainer_export_row(row)
    assert ev.artifact_kind == f"curatorkit_{fmt}_export_row"
    assert ev.granularity == "per_row"
    assert ev.join_keys == {}
    assert ev.missing_join_keys == ["row_id", "source_uri", "provenance_chain",
                                    "task_type", "source_file", "row"]
    for lost in ("row_id", "provenance_chain"):
        assert ev.coverage[lost] == UNAVAILABLE


def test_curatorkit_trainer_export_row_rejects_an_unknown_shape():
    with pytest.raises(ValueError, match="unrecognized CuratorKIT trainer export row"):
        curatorkit_trainer_export_row({"text": "<benign placeholder text>"})


def test_curatorkit_lineage_sidecar_maps_row_id_to_provenance():
    sidecar = lineage_sidecar([
        CURATORKIT_DATA_SAMPLE,
        curatorkit_data_sample(CURATORKIT_REJECTED_SAMPLE),   # SourceEvidence works too
        CURATORKIT_CHECKPOINT_SAMPLE,
    ])
    assert sorted(sidecar) == sorted([CURATORKIT_DATA_SAMPLE["id"],
                                      CURATORKIT_REJECTED_SAMPLE["id"],
                                      CURATORKIT_CHECKPOINT_SAMPLE["id"]])
    entry = sidecar[CURATORKIT_DATA_SAMPLE["id"]]
    assert entry["source_uri"] == "inputs/placeholder_raw.jsonl"
    assert entry["provenance_chain"] == CURATORKIT_DATA_SAMPLE["provenance_chain"]
    assert entry["rejected"] is False
    assert sidecar[CURATORKIT_REJECTED_SAMPLE["id"]]["rejected"] is True
    assert sidecar[CURATORKIT_REJECTED_SAMPLE["id"]]["rejecting_step"] == "SchemaGate"
    # the export itself carries no id, so it can never build its own sidecar
    with pytest.raises(ValueError, match="no 'id'"):
        lineage_sidecar([CURATORKIT_ALPACA_ROW])


# --------------------------------------------------------------------------
# producer 5/5 -- CircuitKIT: aggregate MECHANISTIC evidence, never an outcome
# --------------------------------------------------------------------------
def test_circuitkit_faithfulness_report_is_aggregate_mechanistic_not_an_outcome():
    ev = circuitkit_faithfulness_report(CIRCUITKIT_REPORT)
    assert (ev.source_product, ev.source_revision) == ("circuitkit",
                                                       REVISIONS["circuitkit"])
    assert ev.granularity == "aggregate"
    # a faithfulness score is mechanistic evidence, NOT a task or safety outcome
    assert ev.is_outcome is False
    for f in ("patching_score", "ablation_score", "stability", "robustness",
              "baseline_comparison", "generalization", "intervention_reliability"):
        assert ev.coverage[f] == "observed"
    assert ev.join_keys["model_name"] == "placeholder-model"
    assert ev.join_keys["task"] == "placeholder_task"
    assert ev.missing_join_keys == ["model_revision", "model_digest",
                                    "circuit_artifact_id", "run_id", "dataset",
                                    "case_id", "episode_id"]
    assert ev.coverage["per_case_scores"] == UNAVAILABLE


def test_circuitkit_fast_path_report_has_no_model_identity_at_all():
    """The IBCircuit fast path replaces metadata wholesale."""
    ev = circuitkit_faithfulness_report({**CIRCUITKIT_REPORT, "metadata": {}})
    assert ev.join_keys == {}
    assert ev.missing_join_keys[:2] == ["model_revision", "model_digest"]
    for k in ("model_name", "task", "algorithm", "level", "scope", "timestamp"):
        assert k in ev.missing_join_keys


def test_circuitkit_non_finite_scores_still_serialize_as_strict_json():
    """Its encoder allows bare NaN/Infinity; to_dict() must not."""
    ev = circuitkit_faithfulness_report({**CIRCUITKIT_REPORT,
                                         "patching_score": float("nan"),
                                         "ablation_score": float("inf")})
    assert math.isnan(ev.payload["patching_score"])          # payload verbatim
    text = json.dumps(ev.to_dict(), allow_nan=False)         # strict JSON view
    assert json.loads(text)["payload"]["patching_score"] is None


# --------------------------------------------------------------------------
# the synthetic joined episode: explicit ids or no join
# --------------------------------------------------------------------------
def _joins() -> list[tuple[str, SourceEvidence, dict[str, str], str]]:
    """(label, evidence, by-with-explicit-ids, key-to-withhold)."""
    return [
        ("aligntune_verdict", aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT),
         {"artifact_name": "Q4_K_M", "artifact_format": "gguf"}, "artifact_name"),
        ("safetune_decision", safetune_audit_event(SAFETUNE_AUDIT_EVENT),
         {"episode_id": EXPLICIT_EPISODE_ID, "call_id": EXPLICIT_CALL_ID}, "call_id"),
        ("curatorkit_lineage", curatorkit_trainer_export_row(CURATORKIT_ALPACA_ROW),
         {"row_id": CURATORKIT_DATA_SAMPLE["id"]}, "row_id"),
        ("circuitkit_faithfulness", circuitkit_faithfulness_report(CIRCUITKIT_REPORT),
         {"model_digest": EXPLICIT_MODEL_DIGEST}, "model_digest"),
    ]


def test_one_episode_joins_all_four_producers_with_explicit_ids():
    ep = _episode()
    for label, ev, by, _ in _joins():
        r = join_evidence(ep, ev, by)
        assert r.joined is True, f"{label}: {r.reason}"
        assert set(r.basis) == set(by), label
        assert r.reason is None

    attached = ep.metadata["lexsi_evidence"]
    assert [a["evidence"]["artifact_kind"] for a in attached] == [
        "aligntune_regression_report", "safetune_audit_event",
        "curatorkit_alpaca_export_row", "circuitkit_faithfulness_report"]
    # every attachment records WHICH id matched WHAT
    basis = {a["evidence"]["artifact_kind"]: a["join_basis"] for a in attached}
    assert basis["aligntune_regression_report"]["artifact_name"]["matched"] == "artifact"
    assert basis["safetune_audit_event"]["episode_id"] == {
        "value": EXPLICIT_EPISODE_ID, "matched": "caller_supplied",
        "artifact_value": None, "episode": f"source_id={EXPLICIT_EPISODE_ID}"}
    assert basis["safetune_audit_event"]["call_id"]["episode"] == (
        f"events.call_id={EXPLICIT_CALL_ID}")
    assert basis["curatorkit_alpaca_export_row"]["row_id"]["matched"] == "caller_supplied"
    assert basis["circuitkit_faithfulness_report"]["model_digest"]["matched"] == (
        "caller_supplied")
    # joining evidence never touches the episode's own outcome fields
    assert (ep.source_reward, ep.source_verdict, ep.source_scores) == (None, None, None)
    # and the whole joined record is strict JSON
    json.dumps(ep.to_dict(), allow_nan=False)


@pytest.mark.parametrize("case", _joins(), ids=lambda c: c[0])
def test_the_same_join_fails_by_name_when_the_explicit_id_is_withheld(case):
    label, ev, by, withheld = case
    ep = _episode()
    r = join_evidence(ep, ev, {**by, withheld: None})
    assert r.joined is False, label
    assert withheld in r.reason, r.reason
    assert r.basis == {}
    assert "lexsi_evidence" not in ep.metadata


def test_a_descriptor_alone_is_not_an_id_and_does_not_join():
    """tenant/decision/bench/task describe the evidence, they identify nothing."""
    ep = _episode()
    for ev, by in ((safetune_audit_event(SAFETUNE_AUDIT_EVENT),
                    {"tenant_id": "placeholder-tenant"}),
                   (safetune_audit_event(SAFETUNE_AUDIT_EVENT), {"decision": "block"}),
                   (safetune_bench_row(SAFETUNE_BENCH_ROW), {"bench": "placeholder_bench"}),
                   (circuitkit_faithfulness_report(CIRCUITKIT_REPORT),
                    {"task": "placeholder_task", "algorithm": "eap"})):
        r = join_evidence(ep, ev, by)
        assert r.joined is False, by
        for key in by:
            assert key in r.reason
    # a descriptor may still ACCOMPANY a real id, and is then verified
    ids = {"episode_id": EXPLICIT_EPISODE_ID, "call_id": EXPLICIT_CALL_ID}
    ok = join_evidence(ep, safetune_audit_event(SAFETUNE_AUDIT_EVENT),
                       {**ids, "decision": "block"})
    assert ok.joined is True
    assert ok.basis["decision"]["matched"] == "artifact"
    bad = join_evidence(ep, safetune_audit_event(SAFETUNE_AUDIT_EVENT),
                        {**ids, "decision": "allow"})
    assert bad.joined is False and "does not match" in bad.reason


def test_a_wrong_explicit_id_does_not_join():
    ep = _episode()
    mismatch = join_evidence(ep, safetune_audit_event(SAFETUNE_AUDIT_EVENT),
                             {"episode_id": "ep-9999"})
    assert mismatch.joined is False and "source_id" in mismatch.reason

    bad_call = join_evidence(ep, safetune_audit_event(SAFETUNE_AUDIT_EVENT),
                             {"episode_id": EXPLICIT_EPISODE_ID, "call_id": "call-9999"})
    assert bad_call.joined is False and "not a call in this episode" in bad_call.reason

    bad_variant = join_evidence(ep, aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT),
                                {"artifact_name": "not_a_variant"})
    assert bad_variant.joined is False and "does not match" in bad_variant.reason
    assert "lexsi_evidence" not in ep.metadata


# --------------------------------------------------------------------------
# join contract edges (adversarial review, 2026-09-25)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key,value", [("case_id", "case-0001"),
                                       ("episode_id", EXPLICIT_EPISODE_ID),
                                       ("call_id", EXPLICIT_CALL_ID)])
def test_aggregate_evidence_never_joins_at_case_or_call_level(key, value):
    """A real, matching case/call id still cannot pin a scorecard to one case."""
    for ev in (aligntune_audit_report(ALIGNTUNE_AUDIT_REPORT),
               aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT),
               circuitkit_faithfulness_report(CIRCUITKIT_REPORT)):
        ep = _episode()
        r = join_evidence(ep, ev, {key: value})
        assert r.joined is False, ev.artifact_kind
        assert "aggregate" in r.reason and key in r.reason
        assert "lexsi_evidence" not in ep.metadata


def test_descriptors_never_carry_a_join_alone():
    """Stage, revision, task, format, model name, source or step are shared, not ids."""
    step = CURATORKIT_DATA_SAMPLE["provenance_chain"][0]["step_name"]
    for ev, by in (
            (safetune_audit_event(SAFETUNE_AUDIT_EVENT), {"stage": "pre_check"}),
            (safetune_audit_event(SAFETUNE_AUDIT_EVENT), {"policy_revision": "v1"}),
            (curatorkit_trainer_export_row(CURATORKIT_ALPACA_ROW),
             {"task_type": "instruction_following"}),
            (circuitkit_faithfulness_report(CIRCUITKIT_REPORT), {"dataset": "placeholder"}),
            (circuitkit_faithfulness_report(CIRCUITKIT_REPORT),
             {"model_name": CIRCUITKIT_REPORT["metadata"]["model"]}),
            (aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT),
             {"artifact_format": "gguf"}),
            (curatorkit_data_sample(CURATORKIT_DATA_SAMPLE),
             {"source_uri": CURATORKIT_DATA_SAMPLE["source_uri"]}),
            (curatorkit_data_sample(CURATORKIT_DATA_SAMPLE), {"provenance_chain": step})):
        ep = _episode()
        r = join_evidence(ep, ev, by)
        assert r.joined is False, by
        assert "do not identify" in r.reason, r.reason
        assert "lexsi_evidence" not in ep.metadata


def test_list_valued_artifact_keys_must_match_one_record():
    ev = aligntune_regression_report(ALIGNTUNE_REGRESSION_REPORT)
    crossed = join_evidence(_episode(), ev, {"artifact_name": "Q4_K_M",
                                             "artifact_format": "hf"})
    assert crossed.joined is False and "same record" in crossed.reason
    crossed_path = join_evidence(_episode(), ev, {
        "artifact_name": "baseline", "artifact_path": "/models/placeholder-Q4_K_M.gguf"})
    assert crossed_path.joined is False and "same record" in crossed_path.reason
    same = join_evidence(_episode(), ev, {"artifact_name": "baseline",
                                          "artifact_format": "hf"})
    assert same.joined is True, same.reason


@pytest.mark.parametrize("value", ["   ", float("nan"), float("inf"), [], {}])
def test_blank_or_non_finite_ids_count_as_withheld(value):
    ep = _episode()
    r = join_evidence(ep, curatorkit_trainer_export_row(CURATORKIT_ALPACA_ROW),
                      {"row_id": value})
    assert r.joined is False and "row_id" in r.reason
    assert "lexsi_evidence" not in ep.metadata


@pytest.mark.parametrize("value", [["row-1"], {"id": "row-1"}, 1.5, True])
def test_non_scalar_ids_are_refused(value):
    r = join_evidence(_episode(), curatorkit_trainer_export_row(CURATORKIT_ALPACA_ROW),
                      {"row_id": value})
    assert r.joined is False and "string or integer" in r.reason


def test_a_non_finite_artifact_value_matches_nothing():
    report = {**CIRCUITKIT_REPORT,
              "metadata": {**CIRCUITKIT_REPORT["metadata"], "sparsity": float("nan")}}
    ep = _episode()
    r = join_evidence(ep, circuitkit_faithfulness_report(report),
                      {"model_digest": EXPLICIT_MODEL_DIGEST, "sparsity": "nan"})
    assert r.joined is False and "sparsity" in r.reason


@pytest.mark.parametrize("case", _joins(), ids=lambda c: c[0])
def test_the_same_join_fails_by_name_when_the_explicit_id_is_omitted(case):
    """A caller that lacks the id leaves the key out; that must fail too."""
    label, ev, by, withheld = case
    ep = _episode()
    r = join_evidence(ep, ev, {k: v for k, v in by.items() if k != withheld})
    assert r.joined is False, label
    assert withheld in r.reason, r.reason
    assert "lexsi_evidence" not in ep.metadata


def test_lineage_sidecar_refuses_one_id_with_two_lineages():
    other = {**CURATORKIT_DATA_SAMPLE, "source_uri": "placeholder_other.jsonl"}
    with pytest.raises(ValueError, match="repeats row id"):
        lineage_sidecar([CURATORKIT_DATA_SAMPLE, other])
    # an exact repeat is harmless
    assert len(lineage_sidecar([CURATORKIT_DATA_SAMPLE, CURATORKIT_DATA_SAMPLE])) == 1


# --------------------------------------------------------------------------
# real exports: files written by each library's own serializer at the pin
# --------------------------------------------------------------------------
REAL = Path(__file__).parent / "fixtures" / "lexsi_real"


def _real(rel: str) -> list[dict]:
    """A real export as dicts: one JSON document, or one dict per JSONL line.
    ``json.loads`` accepts CircuitKIT's bare NaN/Infinity tokens."""
    text = (REAL / rel).read_text(encoding="utf-8")
    try:
        return [json.loads(text)]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_real_agenttune_store_export_groups_parallel_calls_and_claims_no_pairing():
    parallel, flat = episodes_from_agenttune(str(REAL / "agenttune/trajectory_store_export.jsonl"))
    raw = _real("agenttune/trajectory_store_export.jsonl")[0]
    assert parallel.source_format == "agenttune_trace"
    assert parallel.source_id == raw["trajectory_id"]          # store uuid4
    assert parallel.source_scores == {"format": 0.1, "termination": 1.0,
                                      "search_usage": 0.3, "correctness": 1.0}
    # the two step-0 calls share one turn; the step blob is copied onto both
    calls = [e for e in parallel.events if e.type == "tool_call"]
    assert [e.turn_id for e in calls] == [0, 0]
    assert [e.payload["tool_name"] for e in calls] == ["search_corpus", "search_corpus"]
    results = {e.payload["output"] for e in parallel.events if e.type == "tool_result"}
    assert len(results) == 1
    assert parallel.coverage["parallel_grouping"] == OBSERVED
    assert parallel.coverage["call_result_pairing"] == UNAVAILABLE
    assert parallel.coverage["tool_call_arguments"] == UNAVAILABLE   # "{'query': ...}" repr
    assert isinstance(calls[0].payload["query"], str)
    assert parallel.coverage["tool_call_names"] == UNAVAILABLE and parallel.turns() == []
    # flat-action path: one call, rollout metadata still says 0 calls
    assert flat.coverage["call_result_pairing"] == INFERRED
    assert raw["n_tool_calls"] == 2 and _real(
        "agenttune/trajectory_store_export.jsonl")[1]["n_tool_calls"] == 0
    assert flat.counters["n_tool_calls"] == 1


@pytest.mark.parametrize("name,retrieved", [("trace_hotpotqa.jsonl", False),
                                            ("trace_finder.jsonl", True)])
def test_real_agenttune_trace_logger_rows(name, retrieved):
    first = episodes_from_agenttune(str(REAL / "agenttune" / name))[0]
    assert first.source_format == "agenttune_trace"
    assert first.source_id is None                     # TraceLogger writes no id
    assert first.final_answer == "<placeholder answer one>"
    assert first.coverage["tool_call_arguments"] == OBSERVED    # parsed args dict
    # both calls carry the same collapsed step blob, so no pairing can be claimed
    assert first.coverage["call_result_pairing"] == UNAVAILABLE
    assert first.coverage["parallel_grouping"] == UNAVAILABLE
    assert first.coverage["tool_call_names"] == UNAVAILABLE
    assert first.coverage["retrieved_contexts"] == (OBSERVED if retrieved else UNAVAILABLE)
    if retrieved:
        assert first.to_trace()["retrieved_contexts"] == ["doc-1::0", "doc-2::3"]
        assert first.metadata["reference_contexts"] == ["doc-1::0"]


def test_real_agenttune_report_and_dataset_rows():
    named, errored = episodes_from_agenttune(str(REAL / "agenttune/report_constructed.json"))
    assert named.source_format == "agenttune_report" and named.source_id == "0"
    assert named.coverage["tool_call_names"] == OBSERVED
    assert named.coverage["tool_call_arguments"] == UNAVAILABLE
    assert errored.errors == ["<placeholder error>"]
    assert errored.coverage["final_answer"] == UNAVAILABLE    # '' after an error is no answer
    tasks = episodes_from_agenttune(str(REAL / "agenttune/dataset_train_grpo.jsonl"))
    assert [t.source_id for t in tasks] == ["qa-0001", "qa-0003", "qa-0000"]
    assert all(t.metadata["is_task"] and t.events == [] for t in tasks)
    assert tasks[0].metadata["reference_contexts"] == ["doc-1::0"]


def test_real_aligntune_reports():
    (audit,) = _real("aligntune/AuditReport.json")
    ev = aligntune_audit_report(audit)
    assert ev.granularity == "aggregate" and list(ev.join_keys) == ["timestamp"]
    assert ev.missing_join_keys == ["episode_id", "call_id", "case_id", "artifact_name",
                                    "checkpoint_digest", "probe_set_id", "probe_id"]
    (report,) = _real("aligntune/RegressionReport.json")
    ev = aligntune_regression_report(report)
    # the runner hardcodes the baseline name
    assert ev.join_keys["artifact_name"] == ["baseline", "Q4_K_M"]
    assert ev.join_keys["artifact_format"] == ["hf", "gguf"]
    assert "checkpoint_digest" in ev.missing_join_keys
    assert all(a["artifact"]["metadata"] == {} for a in [report["baseline"]] + report["variants"])
    assert join_evidence(_episode(), ev, {"artifact_name": "Q4_K_M",
                                          "artifact_format": "gguf"}).joined is True
    assert join_evidence(_episode(), ev, {"artifact_name": "fp16_baseline"}).joined is False


@pytest.mark.parametrize("name,observed,unavailable", [
    ("safety_audit_event__post_keyword_classifier_block.json",
     ["category", "matched_keyword", "response"], []),
    ("safety_audit_event__pre_allow.json", ["category"], ["response"]),
    ("safety_audit_event__pre_tool_block.json", ["tool_name", "response"], []),
    ("safety_audit_event__post_allow_long_redacted.json", ["category", "prompt"], []),
])
def test_real_safetune_audit_events(name, observed, unavailable):
    (event,) = _real(f"safetune/{name}")
    ev = safetune_audit_event(event)
    assert ev.granularity == "per_event"
    assert sorted(ev.join_keys) == ["decision", "reason", "tenant_id", "timestamp"]
    assert ev.missing_join_keys == ["episode_id", "call_id", "case_id", "stage",
                                    "policy_revision"]
    for k in observed:
        assert ev.coverage[k] == OBSERVED, k
    for k in unavailable + ["redaction_applied", "used_default_detector"]:
        assert ev.coverage[k] == UNAVAILABLE, k
    if "long_redacted" in name:
        # default redaction keeps 16 leading + 8 trailing characters of long text
        assert "...[REDACTED]..." in event["prompt"] and event["prompt"] != "[REDACTED]"


def test_real_safetune_bench_rows():
    scored = safetune_bench_row(_real(
        "safetune/placeholder-model__base__placeholder_bench_scored.jsonl")[0])
    assert sorted(scored.join_keys) == ["bench", "prompt"]
    raw = safetune_bench_row(_real("safetune/placeholder-model__base__placeholder_bench.jsonl")[0])
    assert list(raw.join_keys) == ["prompt"]
    assert raw.missing_join_keys == ["case_id", "episode_id", "call_id", "model_name",
                                     "timestamp", "bench"]


@pytest.mark.parametrize("rel,kinds,rows,source_file,chains", [
    ("curatorkit/data_sample.json", ["curatorkit_data_sample"], [2], None,
     [["Connector", "SchemaGate", "ExactDeduplicator", "TextCleaner"]]),
    ("curatorkit/checkpoint_TextCleaner.jsonl", ["curatorkit_data_sample"] * 2, [1, 2], None,
     [["Connector", "SchemaGate", "ExactDeduplicator", "TextCleaner"]] * 2),
    # a Connector rejection (bad JSON, line 4) and a SchemaGate rejection (line 3)
    ("curatorkit/rejected.jsonl", ["curatorkit_rejected_sample"] * 2, [4, 3], None,
     [["Connector"], ["Connector", "SchemaGate"]]),
    ("curatorkit/data_sample_pdf.json", ["curatorkit_data_sample"], [None],
     "inputs/placeholder.pdf", [["PDFReader", "SchemaGate", "ExactDeduplicator", "TextCleaner"]]),
    # PDF table rejection: source_file only in the chain notes, metadata is {page}
    ("curatorkit/rejected_pdf.jsonl", ["curatorkit_rejected_sample"], [None],
     "inputs/placeholder.pdf", [["PDFReader"]]),
])
def test_real_curatorkit_rows_locate_their_source(rel, kinds, rows, source_file, chains):
    samples = _real(rel)
    assert len(samples) == len(kinds)
    for sample, kind, row, chain in zip(samples, kinds, rows, chains):
        ev = curatorkit_data_sample(sample)
        assert ev.artifact_kind == kind and ev.granularity == "per_row"
        assert ev.join_keys["row_id"] == sample["id"]
        assert ev.join_keys["provenance_chain"] == chain
        assert "row" not in sample["metadata"]             # nothing writes metadata.row
        assert ev.join_keys.get("row") == row
        assert ev.join_keys.get("source_file") == source_file
        assert ev.missing_join_keys == (["row"] if row is None else ["source_file"])
    sidecar = lineage_sidecar(samples)
    assert sorted(sidecar) == sorted(s["id"] for s in samples)


def test_real_curatorkit_manifests_and_exports():
    ev = curatorkit_manifest(_real("curatorkit/manifest.json")[0])
    assert ev.granularity == "aggregate"
    assert sorted(ev.join_keys) == ["pipeline_config_hash", "run_timestamp"]
    assert ev.missing_join_keys == ["row_id", "provenance_chain", "source_file_sha256",
                                    "source_files"]
    # a config hash is shared by every run of the config: never a join on its own
    r = join_evidence(_episode(), ev, {"pipeline_config_hash": "4e732c3bf6e9acb0"})
    assert r.joined is False and "do not identify" in r.reason
    pdf = curatorkit_manifest(_real("curatorkit/manifest_pdf.json")[0])
    assert pdf.join_keys["source_files"] == ["inputs/placeholder.pdf"]
    for rel, kind in (("curatorkit/sft_alpaca.jsonl", "curatorkit_alpaca_export_row"),
                      ("curatorkit/sft_sharegpt.jsonl", "curatorkit_sharegpt_export_row")):
        for row in _real(rel):
            ev = curatorkit_trainer_export_row(row)
            assert ev.artifact_kind == kind and ev.join_keys == {}


def test_real_circuitkit_full_report_present_keys():
    ev = circuitkit_faithfulness_report(_real("circuitkit/faithfulness_report.json")[0])
    assert ev.granularity == "aggregate" and ev.is_outcome is False
    # scope is the "unknown" sentinel on this path, so it counts as missing
    assert ev.join_keys == {"model_name": "placeholder-model", "task": "placeholder_task",
                            "algorithm": "eap", "level": "node", "sparsity": 0.0,
                            "timestamp": 1758830400.0}
    assert ev.missing_join_keys == ["model_revision", "model_digest", "circuit_artifact_id",
                                    "run_id", "dataset", "case_id", "episode_id", "scope"]
    for f in ("patching_score", "stability", "intervention_reliability"):
        assert ev.coverage[f] == OBSERVED


@pytest.mark.parametrize("name,null_scores", [
    ("faithfulness_report_ib_fastpath.json", ["stability", "robustness",
                                              "baseline_comparison", "generalization",
                                              "intervention_reliability"]),
    ("faithfulness_report_default.json", ["patching_score", "ablation_score", "stability",
                                          "intervention_reliability"]),
])
def test_real_circuitkit_null_pillars_are_not_observed(name, null_scores):
    ev = circuitkit_faithfulness_report(_real(f"circuitkit/{name}")[0])
    assert ev.join_keys == {}
    for f in null_scores:
        assert ev.coverage[f] == UNAVAILABLE, f


def test_real_circuitkit_unknown_sentinels_and_non_finite_scores():
    ev = circuitkit_faithfulness_report(
        _real("circuitkit/faithfulness_report_unknown_sentinels.json")[0])
    assert sorted(ev.join_keys) == ["level", "sparsity", "timestamp"]
    for k in ("model_name", "task", "algorithm", "scope"):
        assert k in ev.missing_join_keys
    nonfinite = circuitkit_faithfulness_report(
        _real("circuitkit/faithfulness_report_nonfinite.json")[0])
    assert math.isnan(nonfinite.payload["patching_score"])
    text = json.dumps(nonfinite.to_dict(), allow_nan=False)
    assert json.loads(text)["payload"]["ablation_score"] is None


# --------------------------------------------------------------------------
# no producer library, no torch
# --------------------------------------------------------------------------
def test_importing_lexsi_pulls_in_no_producer_library_and_no_torch():
    src = os.path.dirname(os.path.dirname(auditkit.__file__))
    code = (
        "import sys\n"
        "import auditkit.agent_eval.lexsi as lx\n"
        "producers = {'agenttune', 'aligntune', 'safetune', 'curatorkit', 'circuitkit'}\n"
        "leaked = (producers | {'torch', 'transformers', 'pydantic'}) & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "assert hasattr(lx, 'join_evidence')\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": src}
    r = subprocess.run([sys.executable, "-W", "error", "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
