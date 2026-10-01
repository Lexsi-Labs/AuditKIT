"""Write one Markdown report per agentic metric into metric_reports/.

    python -m tests.agentic_suite.make_metric_reports [junit.xml] [live_report.json ...]

Every edge-case row is re-scored here (expected vs actual), so the tables can't
drift from the tests. Test outcomes come from a pytest --junitxml file; live
numbers from the live_report_*.json files (default: every file in live_reports/).
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from auditkit.metrics.agent import (
    ParallelToolCalls, RedundantToolCalls, ToolCallF1, ToolCallValidity, TrajectoryMatch,
)
from auditkit.trace import to_turns

from . import test_metrics_offline as M
from .catalog import TOOLS

HERE = Path(__file__).parent
OUT = HERE / "metric_reports"


def turns_str(turns):
    try:
        tt = to_turns(turns)
    except Exception:
        return str(turns)
    if not tt:
        return "∅"
    return " → ".join("[" + ", ".join(
        f"{x.name}({', '.join(f'{k}={v!r}' for k, v in x.arguments.items())})" + (" ⚠parse" if x.parse_error else "")
        for x in t) + "]" for t in tt)


def fmt(v):
    return "–" if v is None else f"{v:.2f}"


def outcomes(junit):
    res = {}
    if not junit or not Path(junit).exists():
        return res
    for tc in ET.parse(junit).getroot().iter("testcase"):
        s = "pass"
        for ch in tc:
            if ch.tag in ("failure", "error"):
                s = "FAIL"
            elif ch.tag == "skipped":
                s = "xfail" if "xfail" in (ch.get("type", "") + ch.get("message", "")).lower() else "skip"
        res[tc.get("name")] = s
    return res


def status(res, test, case_id=None):
    key = f"{test}[{case_id}]" if case_id else test
    return {"pass": "✅ pass", "FAIL": "❌ FAIL", "xfail": "⚠️ xfail (known finding)", "skip": "skipped"}.get(
        res.get(key), "not run")


def table_rows(res, test, cases, metric_factory, keys, tools=None, pred_idx=2):
    lines = ["| case | reference | predicted | " + " | ".join(f"expected {k}" for k in keys)
             + " | " + " | ".join(f"actual {k}" for k in keys) + " | test |",
             "|---" * (3 + 2 * len(keys) + 1) + "|"]
    ok = bad = 0
    for row in cases:
        cid, ref, pred, want = row[0], row[1], row[pred_idx], row[-1]
        if not isinstance(want, dict):
            want = {keys[0]: want}
        got = M.scores(metric_factory(), ref, pred, tools=tools)
        same = set(got) == set(want) and all(abs(got[k] - want[k]) < 1e-9 for k in want)
        ok += same
        bad += not same
        lines.append(f"| `{cid}` | {turns_str(ref) if ref is not None else '—'} | {turns_str(pred)} | "
                     + " | ".join(fmt(want.get(k)) for k in keys) + " | "
                     + " | ".join(fmt(got.get(k)) for k in keys) + f" | {status(res, test, cid)} |")
    return lines, ok, bad


def live_section(lives, keys, agent_keys=None):
    if not lives:
        return ["_No live report found. Run `AK_LIVE=1 pytest tests/agentic_suite/test_live.py`._", ""]
    out = []
    for name, rep in lives.items():
        for cond, d in rep.get("conditions", {}).items():
            h = d["headline"]
            out += [f"**{name} · {cond}** — headline: " + ", ".join(f"`{k}`={fmt(h.get(k))}" for k in keys), "",
                    "| case | probe | predicted | " + " | ".join(keys) + " |", "|---" * (3 + len(keys)) + "|"]
            for r in d["rows"]:
                out.append(f"| {r['id']} | {r['probe']} | `{r['predicted'][:90]}` | "
                           + " | ".join(fmt(r["scores"].get(k)) for k in keys) + " |")
            out.append("")
        for model, d in rep.get("agent", {}).items():
            ks = agent_keys or keys
            out += [f"**{name} · agent loop ({model})** — headline: "
                    + ", ".join(f"`{k}`={fmt(d['headline'].get(k))}" for k in ks), "",
                    "| case | turns | " + " | ".join(ks) + " |", "|---" * (2 + len(ks)) + "|"]
            for r in d["rows"]:
                out.append(f"| {r['id']} | `{r['predicted'][:110]}` | " + " | ".join(fmt(r["scores"].get(k)) for k in ks) + " |")
            out.append("")
    return out


def _rag_live(lives, metric):
    out = []
    for name, rep in lives.items():
        for judge, d in rep.get("rag", {}).items():
            cases = d["scores"].get(metric, {})
            out += [f"**{name} · judge {judge}** — parse errors {d['n_errors']}/{d['n_calls']}: "
                    + ", ".join(f"{k}={'error/skip' if v is None else f'{v:.2f}'}" for k, v in cases.items()), ""]
    return out or ["_No live RAG report found (AK_LIVE=1 pytest tests/agentic_suite/test_live_rag.py)._"]


def write(name, title, summary, fields, emits, sections):
    OUT.mkdir(exist_ok=True)
    body = [f"# `{name}` — {title}", "", summary, "", "## What it needs from the sample", "", fields, "",
            "## What it emits", "", emits, ""]
    for heading, lines in sections:
        body += [f"## {heading}", ""] + lines + [""]
    (OUT / f"{name}.md").write_text("\n".join(body))


def main(argv):
    junit = next((a for a in argv if a.endswith(".xml")), None)
    live_files = [Path(a) for a in argv if a.endswith(".json")] or sorted((HERE / "live_reports").glob("live_report_*.json"))
    lives = {f.stem.replace("live_report_", ""): json.loads(f.read_text()) for f in live_files}
    for f in sorted((HERE / "live_reports").glob("live_rag_report_*.json")):
        lives.setdefault(f.stem.replace("live_rag_report_", ""), {})["rag"] = json.loads(f.read_text())
    res = outcomes(junit)
    index = []

    # ---- tool_call_f1
    keys = ["tool_call_precision", "tool_call_recall", "tool_call_f1", "tool_call_exact"]
    rows, ok, bad = table_rows(res, "test_tool_call_f1", M.F1_CASES, ToolCallF1, keys)
    extra = [f"- subset mode: {status(res, 'test_f1_subset_mode')}",
             f"- name mode ignores args, not names: {status(res, 'test_f1_name_mode_ignores_arguments_but_not_names')}",
             f"- custom `arg_match` (casefold + numeric tolerance): {status(res, 'test_f1_custom_matcher_casefold_and_tolerance')}",
             f"- exact bipartite beats greedy first-fit: {status(res, 'test_f1_bipartite_beats_greedy')}",
             f"- metadata counts (n_expected / n_predicted / n_matched): {status(res, 'test_f1_metadata_counts')}",
             f"- not applicable without a reference: {status(res, 'test_f1_not_applicable_without_reference')}",
             f"- invalid `arg_mode` rejected: {status(res, 'test_invalid_arg_mode_rejected')}",
             f"- same behaviour in 4 wire formats scores identically: {status(res, 'test_same_behaviour_scored_identically_across_formats')}",
             f"- parse-error calls never match, even in name mode: {status(res, 'test_parse_errors_never_match_even_in_name_mode')}"]
    write("tool_call_f1", "did it call the right tools with the right arguments?",
          "Multiset precision/recall/F1 over all calls, ignoring turn boundaries and order. Duplicates count. "
          "Calls are paired by an exact maximum bipartite matching. An empty reference (`[]`) is an "
          "irrelevance case: 1.0 only if no call was made.",
          "`expected_tool_calls` (required; `[]` allowed). Predicted calls come from the trace "
          "(`tool_calls` > `messages`) or, failing that, parsed from the text output.",
          "`tool_call_precision`, `tool_call_recall`, `tool_call_f1`, `tool_call_exact` "
          "(suffix `_subset` / `_name` / `_custom_<hash>` for non-default matching). Higher is better.",
          [(f"Edge cases ({ok} agree with expectation, {bad} disagree)", rows),
           ("Other checks", extra),
           ("Findings", ["- **LIVE-1 — fixed on this branch.** Live, qwen3:8b and gemma3:12b add the optional `unit=\"celsius\"` argument unasked; exact matching used to fail those correct calls (and correct parallel batches). `exact` is now schema-aware: an optional argument the schema declares and the reference omits is tolerated; undeclared arguments still fail.",
                         "- An `agent:` reply with no transcript and no `tool_calls` is marked `tool_calls_unavailable` "
                         "and the tool metrics skip it (no score). A *recorded text output* with no trace is still "
                         f"parsed as real evidence (recall 0 if it holds no calls). Test: {status(res, 'test_missing_trace_is_scored_as_no_calls')}.",
                         "- String arguments compare exactly (`\"paris\"` ≠ `\"Paris\"`, trailing spaces matter). "
                         "By design; use `arg_match` for normalisation.",
                         "- A reference may name several valid routes (`{\"any_of\": [...]}`): the run is scored "
                         "against the route it took, and `RunResult.route_distribution()` / `summary()` report "
                         "which route each sample took and how often listed order broke a tie."]),
           ("Live results", live_section(lives, ["tool_call_f1", "tool_call_exact", "tool_call_f1_name", "tool_call_f1_subset"]))])
    index.append(("tool_call_f1", ok, bad))

    # ---- trajectory_match
    keys = ["trajectory_strict", "trajectory_in_order"]
    rows, ok, bad = table_rows(res, "test_trajectory_match", M.TRAJ_CASES, TrajectoryMatch, keys)
    write("trajectory_match", "were the calls made in the right turns and order?",
          "`trajectory_strict`: same number of turns and turn *i* holds exactly reference turn *i* (any order "
          "inside a turn). `trajectory_in_order`: every reference turn's calls appear in order in the flattened "
          "run; extra calls allowed, turn boundaries ignored.",
          "`expected_tool_calls` as a list of turns.",
          "`trajectory_strict`, `trajectory_in_order` (binary). Higher is better.",
          [(f"Edge cases ({ok} agree, {bad} disagree)", rows),
           ("Other checks", [f"- name mode suffix: {status(res, 'test_trajectory_name_mode_suffix')}",
                             f"- transcript (`messages`) input end to end: {status(res, 'test_precomputed_transcript_messages')}",
                             f"- agent: backend multi-turn: {status(res, 'test_agent_backend_multi_turn_end_to_end')}"]),
           ("Findings", ["- **LIVE-1 — fixed on this branch.** Live, qwen3:8b and gemma3:12b add the optional `unit=\"celsius\"` argument unasked; exact matching used to fail those correct calls (and correct parallel batches). `exact` is now schema-aware: an optional argument the schema declares and the reference omits is tolerated; undeclared arguments still fail.",
                         "- **LIVE-2** — agent loop, case `agent-par-then-dep`: qwen3:8b batched both weather calls "
                         "correctly, then answered \"local time in Rome is 15:30\" **without calling `get_time`** "
                         "(the stub returns 12:00). `trajectory_strict`=0 and `tool_call_recall_name`=0.67 catch the "
                         "fabrication; the answer alone looks plausible.",
                         "- **DESIGN-1 — FIXED.** `trajectory_in_order` used to be 1.0 on an irrelevance case even "
                         "when a call was made. It is now 1.0 only if no call was made (row `irrelevance-call-made`).",
                         "- `trajectory_in_order` also rates \"everything batched into one turn\" as 1.0 because it "
                         "ignores turn boundaries; pair it with `parallel_precision`."]),
           ("Live results", live_section(lives, keys + ["trajectory_strict_subset"]))])
    index.append(("trajectory_match", ok, bad))

    # ---- parallel_tool_calls
    keys = ["parallel_recall", "parallel_precision", "parallel_detection"]
    rows, ok, bad = table_rows(res, "test_parallel_tool_calls", M.PAR_CASES, ParallelToolCalls, keys)
    write("parallel_tool_calls", "batched the independent calls, and only those?",
          "A reference turn with 2+ calls is a parallel group. `parallel_recall` = groups made together in one "
          "predicted turn; `parallel_precision` = predicted multi-call turns whose matched calls all come from one "
          "reference turn; `parallel_detection` = made a multi-call turn exactly when the reference has one. "
          "Recall/precision are omitted (–) when they have no denominator.",
          "`expected_tool_calls` as turns; a turn with 2+ calls marks the parallel group. For single-response "
          "models put only the NEXT turn in the reference.",
          "`parallel_recall`, `parallel_precision`, `parallel_detection`. Higher is better.",
          [(f"Edge cases ({ok} agree, {bad} disagree)", rows),
           ("Other checks", [f"- partial recall across two groups: {status(res, 'test_parallel_partial_recall_two_groups')}",
                             f"- recall metadata (n_groups, recall_exact): {status(res, 'test_parallel_recall_metadata')}",
                             f"- native parallel calls via mock api: server: {status(res, 'test_api_backend_native_tools_end_to_end')}",
                             f"- Hermes multi-block and concatenated bare JSON = one parallel turn: "
                             f"{status(res, 'test_text_output_parsing', 'hermes-two-blocks-one-turn')} / "
                             f"{status(res, 'test_text_output_parsing', 'concatenated-bare-json')}"]),
           ("Findings", ["- **LIVE-1 — fixed on this branch.** Live, qwen3:8b and gemma3:12b add the optional `unit=\"celsius\"` argument unasked; exact matching used to fail those correct calls (and correct parallel batches). `exact` is now schema-aware: an optional argument the schema declares and the reference omits is tolerated; undeclared arguments still fail.",
                         "- Scores are averaged per sample; no dataset-level micro-average yet.",
                         "- Only the next turn is observable from a single `api:` call; multi-turn parallelism needs "
                         "an `agent:` endpoint or a recorded trace (no harness loop yet)."]),
           ("Live results", live_section(lives, keys + ["parallel_recall_subset"]))])
    index.append(("parallel_tool_calls", ok, bad))

    # ---- tool_call_validity
    keys = ["tool_call_validity"]
    rows, ok, bad = table_rows(res, "test_tool_call_validity",
                               [(x[0], None, x[1], x[2]) for x in M.VALID_CASES], ToolCallValidity, keys, tools=TOOLS)
    write("tool_call_validity", "are the calls valid against the tool schemas?",
          "Fraction of calls that are valid: tool exists, required args present, no unknown args, top-level JSON "
          "types and `enum` hold, arguments decoded as JSON. No reference needed. Samples with no calls are skipped.",
          "`tools` (OpenAI `parameters` or Anthropic `input_schema`).",
          "`tool_call_validity` in [0,1], with a `reason` listing each problem. Higher is better.",
          [(f"Edge cases ({ok} agree, {bad} disagree)", rows),
           ("Other checks", [f"- reason names the problems: {status(res, 'test_validity_reason_names_the_problem')}",
                             f"- skipped with no calls / not applicable without tools: {status(res, 'test_validity_skips_when_no_calls_and_needs_tools')}",
                             f"- Anthropic schema + nullable type list: {status(res, 'test_validity_anthropic_schema_and_nullable_type')}",
                             f"- hallucinated tool via api: backend: {status(res, 'test_api_backend_native_tools_end_to_end')}"]),
           ("Findings", [f"- **LIMIT-1 — FIXED.** Only top-level types used to be checked. Validation now recurses "
                         f"into array `items` and object `properties` (nested `required`, `additionalProperties: false`). "
                         f"Regression tests: {status(res, 'test_validity_checks_nested_types')}, "
                         f"`test_validity_nested_cases`, `test_validity_nested_required_and_additional_properties`."]),
           ("Live results", live_section(lives, keys))])
    index.append(("tool_call_validity", ok, bad))

    # ---- redundant_tool_calls
    keys = ["redundant_tool_calls"]
    rows, ok, bad = table_rows(res, "test_redundant_tool_calls",
                               [(x[0], None, x[1], x[2]) for x in M.REDUNDANT_CASES], RedundantToolCalls, keys)
    write("redundant_tool_calls", "did it repeat identical calls?",
          "Fraction of calls that exactly repeat an earlier call (same name and arguments, key order ignored). "
          "Malformed calls are not counted as repeats but stay in the denominator. No reference needed.",
          "Nothing beyond the trace.",
          "`redundant_tool_calls` in [0,1]. **Lower is better.**",
          [(f"Edge cases ({ok} agree, {bad} disagree)", rows),
           ("Other checks", [f"- skipped with no calls: {status(res, 'test_redundant_skips_no_calls')}",
                             f"- looping agent via agent: backend: {status(res, 'test_agent_backend_multi_turn_end_to_end')}"]),
           ("Findings", [f"- **FINDING-2 — FIXED.** Repeats are now keyed on the same JSON-value equality as "
                         f"matching (`1` = `1.0`, `True` ≠ `1`, recursively). "
                         f"Regression test: {status(res, 'test_redundant_uses_same_equality_as_matching')}.",
                         "- A legitimate retry after an error counts as redundant (it can't see tool results)."]),
           ("Live results", live_section(lives, keys))])
    index.append(("redundant_tool_calls", ok, bad))

    # ---- task_completion
    judge_lines = []
    for name, rep in lives.items():
        if rep.get("judge"):
            judge_lines += [f"**{name}** — judge `{rep['judge'].get('judge')}`: "
                            + ", ".join(f"{k}={v['value']}" + (" (unknown)" if (v.get('meta') or {}).get('unknown') else "")
                                        for k, v in rep["judge"].items() if k != "judge"), ""]
    write("task_completion", "did the agent accomplish the task? (LLM judge)",
          "An LLM judge sees the task, the optional expected outcome (`target`), the rendered tool transcript "
          "(with tool results when the trace has `messages`) and the final answer, and picks complete (1.0), "
          "partial (0.5) or failed (0.0).",
          "A `judge_model`. `target` optional.",
          "`task_completion` in {0, 0.5, 1}. A reply with no readable verdict is a recorded error, not a score.",
          [("Scripted-judge checks", [
              "- complete→1.0 / partial→0.5 / failed→0.0 / think-draft ignored (last verdict wins): "
              + ", ".join(f"{k.split('[', 1)[1].rstrip(']')[:40]!r}: {v}" for k, v in sorted(res.items())
                          if k.startswith("test_task_completion_verdicts[")),
              "- no verdict / empty reply raises (recorded error): "
              + ", ".join(v for k, v in sorted(res.items()) if k.startswith("test_task_completion_unreadable_verdict_raises[")),
              f"- prompt carries task, transcript, tool results, answer: {status(res, 'test_task_completion_prompt_carries_everything')}",
              f"- '(no tool calls)' visible to the judge: {status(res, 'test_task_completion_no_calls_is_visible_to_judge')}"]),
           ("Findings", [f"- **FINDING-3 — FIXED.** An unreadable verdict used to score 0.0 and be averaged in. It "
                         f"now raises, the Runner records it in `errors`, and the sample gets no score — the same "
                         f"rule as the RAG judges. Regression: "
                         f"{status(res, 'test_task_completion_parse_failure_is_not_averaged_as_zero')}.",
                         "- Only as good as the judge; the live run self-judges on one model.",
                         "- **LIVE-2** — on the fabricated-time agent run the real judge (qwen3:8b) said *partial* (0.5), "
                         "not *failed*: it partly trusted an answer no tool produced. The deterministic trajectory "
                         "metrics were stricter than the judge here.",
                         "- Real-judge sanity passed: a genuine booking scored 1.0, a claimed-but-not-done booking 0.0."]),
           ("Live results: real judge sanity (true completion vs claimed-but-not-done)", judge_lines or
            ["_No live judge result._"]),
           ("Live results: agent loop", live_section({k: {"agent": v.get("agent", {})} for k, v in lives.items()},
                                                     ["task_completion"], ["task_completion", "tool_call_f1",
                                                                           "trajectory_strict"]))])
    index.append(("task_completion", None, None))


    # ================= RAG metrics =================
    from . import test_rag_offline as RG

    keys = ["hit_rate", "precision", "recall", "mrr", "average_precision", "ndcg"]
    lines = ["| case | retrieved (ranked) | gold | k | " + " | ".join(f"exp {k}" for k in keys)
             + " | " + " | ".join(f"act {k}" for k in keys) + " | test |", "|---" * (4 + 2 * len(keys) + 1) + "|"]
    ok = bad = 0
    for cid, ranked, ref, k, want in RG.RET_CASES:
        got = RG.ret(ranked, ref, k)
        at = f"@{k}" if k else ""
        same = got.keys() == want.keys() and all(abs(got[x] - want[x]) < 1e-9 for x in want)
        ok += same
        bad += not same
        lines.append(f"| `{cid}` | {ranked} | {ref} | {k or '–'} | " + " | ".join(fmt(want[x + at]) for x in keys)
                     + " | " + " | ".join(fmt(got.get(x + at)) for x in keys) + f" | {status(res, 'test_retrieval_metrics', cid)} |")
    write("retrieval", "deterministic ranking quality against gold chunk ids",
          "Compares the ranked list a RAG pipeline actually retrieved (trace `retrieved_contexts`, else "
          "`Sample.retrieval_context`) with gold `reference_contexts` (a list, or `{id: grade}` for graded nDCG). "
          "No model calls.",
          "`reference_contexts` (required). Retrieved list from the trace or `retrieval_context`.",
          "`hit_rate`, `precision`, `recall`, `mrr`, `average_precision` (÷ all relevant), `ndcg` (ideal from gold "
          "grades); `@k` suffix with `k`. Higher is better.",
          [(f"Edge cases ({ok} agree, {bad} disagree)", lines),
           ("Other checks", [f"- trace wins over dataset contexts: {status(res, 'test_retrieval_trace_wins_over_dataset_contexts')}",
                             f"- falls back to `retrieval_context`: {status(res, 'test_retrieval_falls_back_to_dataset_contexts')}",
                             f"- skipped without gold; k=0 rejected: {status(res, 'test_retrieval_skipped_without_gold_and_rejects_bad_k')}"]),
           ("Findings", ["- **RAG-1 — FIXED.** Duplicates were removed *before* ranking, so every item after a "
                         "duplicate moved up a rank (`[x, x, a]` put `a` at rank 2): `mrr`, `average_precision` and "
                         "`ndcg` were inflated. A duplicate now keeps its rank slot (rows `duplicate-*`).",
                         "- Ids compare as exact strings after trimming whitespace; `Doc-1` ≠ `doc-1` by design."])])
    index.append(("retrieval", ok, bad))

    def judge_checks(prefix_list):
        return [f"- `{t}`: " + ", ".join(sorted({v for k, v in res.items() if k == t or k.startswith(t + "[")}) or {"not run"})
                for t in prefix_list]

    write("faithfulness", "is every claim in the answer supported by the retrieved context? (LLM judge)",
          "Two judge calls: split the answer into claims, then one numbered verdict per claim "
          "(SUPPORTED / UNSUPPORTED / CONTRADICTED). Score = supported / claims, so an unsupported claim fails too.",
          "Retrieved contexts (trace or `retrieval_context`) and a non-empty answer; a `judge_model`.",
          "`faithfulness` in [0,1]; `metadata` has claims, verdicts, `n_contradicted`. Higher is better.",
          [("Checks (scripted judge, every verdict known)", judge_checks([
              "test_faithfulness_supported_unsupported_contradicted", "test_faithfulness_tolerates_decorated_verdicts",
              "test_unsupported_is_not_read_as_supported", "test_faithfulness_wrong_verdict_count_is_an_error_not_a_zero",
              "test_faithfulness_conflicting_duplicate_verdict_is_an_error", "test_faithfulness_no_claims_is_skipped",
              "test_faithfulness_skips_without_contexts_or_answer", "test_faithfulness_strips_judge_reasoning",
              "test_faithfulness_does_not_judge_the_answers_own_reasoning",
              "test_faithfulness_uses_retrieved_contexts_and_truncates", "test_rag_pipeline_end_to_end"])),
           ("Findings", ["- **RAG-2 — FIXED.** A reasoning model's answer was judged *including* its "
                         "`<think>...</think>` drafts, so discarded guesses became unsupported claims. The answer's "
                         "reasoning is now stripped before claim extraction.",
                         "- **RAG-3 — FIXED.** A judge that gave one item two different verdicts (`1: SUPPORTED` … "
                         "`1: CONTRADICTED`) was silently read as the last one. It is now a recorded error.",
                         "- Claim extraction is non-deterministic across runs (inherent to the method)."]),
           ("Live results", _rag_live(lives, "faithfulness"))])
    index.append(("faithfulness", None, None))

    write("context_precision", "are the retrieved chunks relevant, and ranked relevant-first? (LLM judge)",
          "One judge call labels each chunk RELEVANT / IRRELEVANT for the question (and the reference answer when "
          "`target` is set).",
          "Retrieved contexts; a `judge_model`; `target` optional.",
          "`context_precision` (RAGAS rank-weighted AP over relevant chunks) and `context_relevance` (fraction relevant).",
          [("Checks", judge_checks(["test_context_precision", "test_context_precision_includes_reference_only_when_set",
                                    "test_context_precision_count_mismatch_is_error"])),
           ("Notes", ["- Its AP divides by *retrieved*-relevant chunks (RAGAS formula), so a missed document is "
                      "invisible here; `retrieval` (gold ids) and `context_recall` cover misses."]),
           ("Live results", _rag_live(lives, "context_precision"))])
    index.append(("context_precision", None, None))

    write("context_recall", "does the retrieved context cover the reference answer? (LLM judge)",
          "`target` is split into sentences locally; one judge call labels each ATTRIBUTED / NOT_ATTRIBUTED.",
          "`target` (required), retrieved contexts, a `judge_model`.",
          "`context_recall` = attributed / sentences; `reason` lists the missing sentences.",
          [("Checks", judge_checks(["test_context_recall_per_sentence", "test_context_recall_sentence_split_keeps_decimals",
                                    "test_context_recall_needs_target", "test_unsupported_is_not_read_as_supported"])),
           ("Notes", ["- Sentence split is on `.`/`!`/`?` + space: decimals survive, abbreviations like `e.g. x` split."]),
           ("Live results", _rag_live(lives, "context_recall"))])
    index.append(("context_recall", None, None))

    lines = ["| case | metric | output | contexts | target | expected | actual | test |", "|---" * 8 + "|"]
    ok = bad = 0
    for cid, metric, output, contexts, target, want in RG.LEX_CASES:
        got = RG.lex(metric, output, contexts, target).get(metric.name)
        same = got is not None and abs(got - want) < 1e-9
        ok += same
        bad += not same
        lines.append(f"| `{cid}` | `{metric.name}` | {output!r} | {contexts} | {target!r} | {fmt(want)} | {fmt(got)} | "
                     f"{status(res, 'test_lexical_metrics', cid)} |")
    write("lexical_rag_metrics", "lexical_groundedness · context_coverage · context_overlap · answer_overlap",
          "Cheap word-overlap heuristics (no model): answer words found in the contexts; target words found in the "
          "contexts; contexts sharing a word with the answer; target words found in the answer.",
          "Retrieved contexts (trace or `retrieval_context`); `target` for coverage and answer_overlap.",
          "Each in [0,1]. Higher is better. Skipped (no score) when there are no contexts at all.",
          [(f"Edge cases ({ok} agree, {bad} disagree)", lines),
           ("Other checks", judge_checks(["test_lexical_metrics_read_live_retrieved_contexts",
                                          "test_lexical_trace_wins_over_dataset_contexts", "test_coverage_reads_live_contexts"])),
           ("Findings", ["- **RAG-4 — FIXED.** The four lexical metrics only read `Sample.retrieval_context`, so a live "
                         "RAG endpoint's actual retrieval was ignored (and the metric skipped when the dataset had none). "
                         "They now use the same source as `retrieval` and the judges.",
                         "- **RAG-5 — FIXED.** Tokenisation disagreed between them and none ignored case, so "
                         "`\"Paris.\"` ≠ `\"Paris\"` and `\"paris\"` ≠ `\"Paris\"`. All four now lowercase and strip punctuation.",
                         "- Still coarse by design: word-prefix matching (`in` matches `information`), and one shared "
                         "word (even `the`) makes a whole chunk count for `context_overlap`. Prefer the judge metrics."])])
    index.append(("lexical_rag_metrics", ok, bad))

    # ---- index
    lines = ["# Agentic + RAG metric reports", "",
             "Generated by `make_metric_reports.py` from the case tables in the tests, a pytest junit file"
             + (f" (`{Path(junit).name}`)" if junit else "") + " and live reports: "
             + (", ".join(f"`{k}`" for k in lives) or "none") + ".", "",
             "| metric | offline edge cases (agree / disagree) | report |", "|---|---|---|"]
    for n, ok, bad in index:
        lines.append(f"| `{n}` | {'judge; see report' if ok is None else f'{ok} / {bad}'} | [{n}.md]({n}.md) |")
    counts = {}
    for v in res.values():
        counts[v] = counts.get(v, 0) + 1
    lines += ["", "Offline test outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "",
              "See FINDINGS.md for every issue found, what was fixed, and what is still open."]
    (OUT / "README.md").write_text("\n".join(lines))
    print("wrote", sorted(p.name for p in OUT.glob("*.md")))


if __name__ == "__main__":
    main(sys.argv[1:])
