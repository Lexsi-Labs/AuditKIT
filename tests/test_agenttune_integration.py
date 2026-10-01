"""End-to-end interoperability: AgentTune's real output shapes -> AuditKit metrics.

Every fixture here mirrors a *real* AgentTune writer (cited by source
file:line against AgentTune commit 36d47248b3f62704e08c490eb17eca59449957e0,
src/agenttune). The point is to prove that
what AgentTune actually emits flows through ``ak.load_agenttune`` and scores
end-to-end with the matching AuditKit agent/RAG/answer metrics -- not to
re-test the loader's field mapping (that's tests/test_loaders_agent.py).

Coverage:
  (a) TrajectoryDataset JSONL record: multi-step conversation, a parallel tool
      turn, retrieved_chunk_ids  -> ToolCallF1 / ParallelToolCalls / TrajectoryMatch
      + RetrievalMetrics.
  (b) run_eval input row: prompt messages + answer + message_ids
      -> exact_match / quasi_exact_match.
  (c) RAG-GRPO row with gold_path -> RetrievalMetrics.
  (d) TraceLogger trace.jsonl record -> answer metrics + RetrievalMetrics.
  (e) Report.save JSON -> exact_match / quasi_exact_match.

Every evaluate test sets XDG_CACHE_HOME to a tmp dir.
"""

from __future__ import annotations

import json
import uuid

import pytest

import auditkit as ak


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


# rollout_factory.py normalises every parsed tool call's "arguments" to a JSON
# *string* (rollout_factory.py:91-92) and gives each a bare 9-char hex id
# (_assign_tool_call_ids, rollout_factory.py:702-722). The assistant turn stores
# them as OpenAI-style {"id","type":"function","function":{"name","arguments"}}
# (rollout_factory.py:1587-1596 read that shape back).
def _call(name, **args):
    return {"id": uuid.uuid4().hex[:9], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


# ── (a) TrajectoryDataset JSONL ────────────────────────────────────────────────
# Modeled on rollout_factory.py's Trajectory (rollout_factory.py:1808-1822):
#   task=prompt (the [system,user] chat list fed to the rollout),
#   steps=[Step(action={"tool_calls":[...]}), ..., terminal Step(action={})]
#     (Step: step_number/state/action/observation/thought, dataset.py:7-15;
#      state="[turn=N, conv_len=M]", observation=str(dict(tool_results)),
#      thought=raw_output, rollout_factory.py:1647-1653; terminal action={},
#      rollout_factory.py:1782-1788; NO step.metadata -- Step(**s) in
#      dataset.py:100 would reject it),
#   final_response=raw_output,
#   metadata={conversation, completions_list, tool_call_count,
#             tool_failure_count, retrieved_chunk_ids} (rollout_factory.py:1809-1821).
# A tool result message is {"role":"tool","name","content","tool_call_id"}
# (rollout_factory.py:1615-1623). Written/read as JSONL by
# TrajectoryDataset.from_jsonl (dataset.py:95-111). Chunk ids carry spaces and
# "::" on purpose (the recall regex allows them, rollout_factory.py:1636) to
# prove RetrievalMetrics matches ids by exact string equality.
TASK = [
    {"role": "system", "content": "You are a financial research agent. Use the tools, then answer."},
    {"role": "user", "content": "What were Acme's FY24 revenue and EBITDA?"},
]
_TC0 = [_call("search_corpus", query="Acme FY24 revenue"),
        _call("search_corpus", query="Acme FY24 EBITDA")]
_TC1 = [_call("fetch_doc", doc_id="acme-10k")]
CHUNK_REV = "Acme Q4 2024 Earnings::66"
CHUNK_EBITDA = "Acme Q4 2024 Earnings::67"
TRAJ = {
    "task": TASK,
    "trajectory_id": "traj-acme-1",
    "reward": 1.0,
    "final_response": "<answer>Revenue $111.5M, EBITDA $32.1M.</answer>",
    "steps": [
        {"step_number": 0, "state": "[turn=0, conv_len=3]", "thought": "Two independent lookups.",
         "action": {"tool_calls": _TC0},
         "observation": "{'search_corpus': '...[chunk_id=Acme Q4 2024 Earnings::66 doc_id=acme-10k]...'}",
         "reward": None},
        {"step_number": 1, "state": "[turn=1, conv_len=7]", "thought": "Now fetch the filing.",
         "action": {"tool_calls": _TC1}, "observation": "{'fetch_doc': '...'}", "reward": None},
        {"step_number": 2, "state": "[terminal, conv_len=9]", "action": {},
         "observation": "<answer>Revenue $111.5M, EBITDA $32.1M.</answer>",
         "thought": "<answer>Revenue $111.5M, EBITDA $32.1M.</answer>", "reward": 1.0},  # terminal thought=raw_output
    ],
    "metadata": {
        "conversation": [
            TASK[0], TASK[1],
            {"role": "assistant", "content": None, "tool_calls": _TC0},
            {"role": "tool", "name": "search_corpus", "content": "[chunk_id=Acme Q4 2024 Earnings::66 ...]",
             "tool_call_id": _TC0[0]["id"]},
            {"role": "tool", "name": "search_corpus", "content": "[chunk_id=Acme Q4 2024 Earnings::67 ...]",
             "tool_call_id": _TC0[1]["id"]},
            {"role": "assistant", "content": None, "tool_calls": _TC1},
            {"role": "tool", "name": "fetch_doc", "content": "...", "tool_call_id": _TC1[0]["id"]},
            {"role": "assistant", "content": "<answer>Revenue $111.5M, EBITDA $32.1M.</answer>"},
        ],
        "completions_list": [],
        "tool_call_count": 3,
        "tool_failure_count": 0,
        "retrieved_chunk_ids": [CHUNK_REV, "Other Co Report::12", CHUNK_EBITDA],
    },
}


def test_a_trajectory_dataset_agent_and_retrieval_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (s,) = ak.load_agenttune(_write_jsonl(tmp_path / "traj.jsonl", [TRAJ]))
    assert s.kind == ak.TaskKind.AGENT and s.id == "traj-acme-1"
    # Turns come from steps: one parallel turn, then one single-call turn.
    assert [len(t) for t in ak.to_turns(s.actual_trace)] == [2, 1]

    # Attach the reference the metrics need (AgentTune trajectories don't carry
    # expected_tool_calls / gold ids -- the "attach after loading" the task expects).
    s.expected_tool_calls = [
        [{"name": "search_corpus", "arguments": {"query": "Acme FY24 revenue"}},
         {"name": "search_corpus", "arguments": {"query": "Acme FY24 EBITDA"}}],
        [{"name": "fetch_doc", "arguments": {"doc_id": "acme-10k"}}],
    ]
    s.reference_contexts = [CHUNK_REV, CHUNK_EBITDA]  # gold subset of the retrieved ids

    result = ak.evaluate(
        [s], model="precomputed",
        scorers=[ak.ToolCallF1(), ak.ParallelToolCalls(), ak.TrajectoryMatch(), ak.RetrievalMetrics()],
    )
    assert not result.errors
    h = result.headline
    # Right calls, right order, right batching.
    assert h["tool_call_f1"] == 1.0 and h["tool_call_exact"] == 1.0
    assert h["trajectory_strict"] == 1.0 and h["trajectory_in_order"] == 1.0
    assert h["parallel_recall"] == 1.0 and h["parallel_precision"] == 1.0 and h["parallel_detection"] == 1.0
    # Retrieval over ranked [rev(rel), other(not), ebitda(rel)] vs gold {rev, ebitda};
    # the space-and-"::" ids prove exact string-equality matching.
    assert h["hit_rate"] == 1.0 and h["recall"] == 1.0 and h["mrr"] == 1.0
    assert h["precision"] == pytest.approx(2 / 3)
    assert h["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)


# ── (b) run_eval input row ─────────────────────────────────────────────────────
# Modeled on eval/agent_eval.py run_eval's dataset rows (agent_eval.py:150-171):
# prompt is the [system,user] chat list, answer may be a list (Enron quirk,
# agent_eval.py:159-160), message_ids is a per-answer list-of-lists
# (agent_eval.py:8, :169-173). These are tasks to *run*, so actual_output is
# attached here to score them offline with model="precomputed".
def test_b_run_eval_row_answer_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    prompt = [{"role": "system", "content": "Answer from the mailbox. Use the tools."},
              {"role": "user", "content": "Confirmation number for my Lisbon hotel?"}]
    rows = [
        {"prompt": prompt, "answer": ["251832", "#251832"], "message_ids": [["m-101"], ["m-104", "m-105"]]},
        {"prompt": [{"role": "user", "content": "Who sent the invoice?"}], "answer": "Dana K."},
    ]
    a, b = ak.load_agenttune(_write_jsonl(tmp_path / "eval.jsonl", rows))
    assert a.input == "Confirmation number for my Lisbon hotel?" and a.target == "251832"
    assert a.reference_contexts == ["m-101", "m-104", "m-105"]  # message_ids flattened
    assert a.actual_output is None  # a task to run

    a.actual_output = "251832"     # exact hit
    b.actual_output = "dana k"     # only a normalized (quasi) hit
    result = ak.evaluate([a, b], model="precomputed", scorers=["exact_match", "quasi_exact_match"])
    assert not result.errors
    assert result.headline["exact_match"] == 0.5        # only a is exact
    assert result.headline["quasi_exact_match"] == 1.0  # both match after normalization


# ── (c) RAG-GRPO row with gold_path ────────────────────────────────────────────
# Modeled on build_dataset.py _grpo_rows (build_dataset.py:159-175):
# prompt=[system,user], gold_answer, question_id, gold_path=json.dumps(gold_chunk_ids),
# plus hop_count/difficulty_cell. gold_path is a JSON *string*, decoded to the
# gold ids by the loader. Written to dataset_grpo.jsonl (build_dataset.py:446-449).
def test_c_rag_grpo_row_retrieval_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    row = {
        "prompt": [{"role": "system", "content": "sqlite RAG system prompt"},
                   {"role": "user", "content": "What was FY24 revenue?"}],
        "gold_answer": "$111.5M", "question_id": "hp-42",
        "gold_path": json.dumps(["ch-7", "ch-9"]), "hop_count": 2, "difficulty_cell": "2-hard",
    }
    (s,) = ak.load_agenttune(_write_jsonl(tmp_path / "grpo.jsonl", [row]))
    assert s.kind == ak.TaskKind.RAG and s.id == "hp-42"
    assert s.reference_contexts == ["ch-7", "ch-9"] and "gold_path" not in s.metadata

    # Attach a simulated retrieval + answer to score offline.
    s.actual_output = "$111.5M"
    s.actual_trace = {"retrieved_contexts": ["ch-7", "ch-3", "ch-9"]}
    result = ak.evaluate([s], model="precomputed", scorers=[ak.RetrievalMetrics()])
    assert not result.errors
    h = result.headline
    # ranked [ch-7(rel), ch-3(not), ch-9(rel)] vs gold {ch-7, ch-9}
    assert h["hit_rate"] == 1.0 and h["recall"] == 1.0 and h["mrr"] == 1.0
    assert h["precision"] == pytest.approx(2 / 3)
    assert h["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)


# ── (d) TraceLogger trace.jsonl record ─────────────────────────────────────────
# Modeled on train_grpo.py TraceLogger.__call__ (train_grpo.py:140-152):
# question, tool_calls=[{query:<arguments dict>, result:<observation>}],
# n_tool_calls, final_answer (may wrap <answer>), has_answer_tag, reward,
# gold_answer, reward_components, and (FinDER/mixed only) retrieved_chunk_ids +
# gold_chunk_ids as RAW lists (train_grpo.py:150-152).
def test_d_tracelogger_trace_answer_and_retrieval(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rec = {
        "question": "What was FY24 revenue?",
        "tool_calls": [{"query": {"q": "revenue"}, "result": "[chunk_id=d::0 doc_id=x] ..."}],
        "n_tool_calls": 1, "final_answer": "<answer>$111.5 million</answer>", "has_answer_tag": True,
        "reward": 1.0, "gold_answer": "$111.5 million", "reward_components": {"answer": 1.0},
        "retrieved_chunk_ids": ["d::0", "d::1"], "gold_chunk_ids": ["d::0"],
    }
    (s,) = ak.load_agenttune(_write_jsonl(tmp_path / "trace.jsonl", [rec]))
    assert s.kind == ak.TaskKind.RAG
    assert s.actual_output == "$111.5 million"          # <answer> stripped
    assert s.reference_contexts == ["d::0"]
    assert s.actual_trace == {"retrieved_contexts": ["d::0", "d::1"]}

    result = ak.evaluate([s], model="precomputed",
                         scorers=["exact_match", "quasi_exact_match", ak.RetrievalMetrics()])
    assert not result.errors
    h = result.headline
    assert h["exact_match"] == 1.0 and h["quasi_exact_match"] == 1.0
    # ranked [d::0(rel), d::1(not)] vs gold {d::0}
    assert h["hit_rate"] == 1.0 and h["recall"] == 1.0 and h["mrr"] == 1.0
    assert h["precision"] == pytest.approx(1 / 2) and h["average_precision"] == 1.0


# ── (e) Report.save JSON ───────────────────────────────────────────────────────
# Modeled on eval/agent_eval.py Report.save (agent_eval.py:142-162):
# {n_samples, n_errors, samples:[{idx, question, gold, predicted, n_tools,
# tool_calls, scores, error}]}. predicted may wrap <answer>; gold is a string
# (agent_eval.py:152). Precomputed answers, so only answer metrics apply.
def test_e_report_save_json_answer_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    report = {
        "use_case": "mail_qa", "n_samples": 2, "n_errors": 0,
        "samples": [
            {"idx": 0, "question": "Confirmation number?", "gold": "251832",
             "predicted": "<answer>251832</answer>", "n_tools": 2,
             "tool_calls": ["search_mail", "read_mail"], "scores": {"exact": 1.0}, "error": None},
            {"idx": 1, "question": "Capital of France?", "gold": "Paris",
             "predicted": "paris.", "n_tools": 0, "tool_calls": [], "scores": {}, "error": None},
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    a, b = ak.load_agenttune(str(path))
    assert a.actual_output == "251832" and a.target == "251832"  # <answer> stripped
    assert b.actual_output == "paris." and b.target == "Paris"

    result = ak.evaluate([a, b], model="precomputed", scorers=["exact_match", "quasi_exact_match"])
    assert not result.errors
    assert result.headline["exact_match"] == 0.5        # only a is exact
    assert result.headline["quasi_exact_match"] == 1.0  # "paris." normalizes to "paris"


# ── (f) TrajectoryStore export JSONL ───────────────────────────────────────────
# Modeled on trajectory_store.py get_full_trajectory (trajectory_store.py:299-316):
# a SELECT * over the trajectories table (columns :91-104) plus parsed
# reward_components/metadata and joined steps (:106-118) and tool_calls
# (:123-131, each row carrying tool_name/query/result/step_number). One JSON line
# per trajectory, written by export_jsonl (trajectory_store.py:337-360).
# make_trajectory_callback builds the TrajectoryRecord with no metadata=
# (trajectory_store.py:432-445), so metadata_json is "{}" -- no retrieved_chunk_ids.
# It carries both "question"+"final_answer" and "steps", so it must route to the
# TRACE branch (question+final_answer wins), preserving trajectory_id as the id.
def test_f_trajectory_store_export_routes_to_trace_and_keeps_id(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    export = {
        "trajectory_id": "run7-traj-3", "run_id": "run7", "question": "What was FY24 revenue?",
        "gold_answer": "$111.5 million", "final_answer": "<answer>$111.5 million</answer>",
        "reward": 1.0, "n_tool_calls": 1, "has_answer_tag": 1, "condition": "phase1",
        "model": "qwen2.5-7b", "training_step": 40, "timestamp": 1.0,
        "reward_components": {"answer": 1.0}, "metadata": {},  # metadata_json == "{}"
        "steps": [
            {"step_id": 1, "trajectory_id": "run7-traj-3", "step_number": 0, "thought": "",
             "action_name": "search_corpus", "action_args_json": json.dumps({"q": "revenue"}),
             "observation": "[chunk_id=d::0 doc_id=x]", "reward": None, "is_tool_step": 1, "is_terminal": 0},
        ],
        "tool_calls": [
            {"tool_call_id": 1, "trajectory_id": "run7-traj-3", "step_number": 0,
             "tool_name": "search_corpus", "query": "{'q': 'revenue'}", "result": "[chunk_id=d::0 ...]"},
        ],
    }
    (s,) = ak.load_agenttune(_write_jsonl(tmp_path / "export.jsonl", [export]))
    # Trace branch, despite carrying "steps": answer is the <answer>-unwrapped
    # final_answer (not final_response), and trajectory_id is preserved as the id.
    # No gold_chunk_ids and metadata=="{}", so nothing marks it RAG -> GENERATIVE.
    assert s.kind == ak.TaskKind.GENERATIVE and s.id == "run7-traj-3"
    assert s.actual_output == "$111.5 million" and s.target == "$111.5 million"
    # An export's tool_calls carry a tool_name but still stay in metadata only.
    assert s.metadata["tool_calls"][0]["tool_name"] == "search_corpus"
    assert s.actual_trace is None  # metadata was "{}", so no retrieved ids

    result = ak.evaluate([s], model="precomputed", scorers=["exact_match"])
    assert not result.errors and result.headline["exact_match"] == 1.0
