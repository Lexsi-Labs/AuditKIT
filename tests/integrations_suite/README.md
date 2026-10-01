# Integrations suite: AgentTune, Cohere, `hf:`

These are tests for the agentic integrations added by #1 and #9/#10: AgentTune import, Cohere tool calling, and the `hf:` backend's native tool path. The suite is based on `HK-AK/interop` (#10), where these features live.

| File | Covers | Needs |
|---|---|---|
| `test_agenttune.py` | Every AgentTune output shape in `tests/fixtures/lexsi_real/agenttune`: GRPO / run_eval dataset rows, TraceLogger records, TrajectoryStore exports, TrajectoryDataset runs, run_eval reports and EventLogs. Also: answer-tag stripping, run folders (name order, sub-folders ignored, `lexsi_provenance.json`), `inspect_agenttune`, `rescore` (unrun tasks skipped, source reward is not the outcome), a deployed agent behind `agent:` (Cohere-shaped calls, OpenAI transcripts with contexts, per-sample errors), and checkpoint provenance reaching `RunResult.metadata` and the saved results. | offline; the provenance test needs the Qwen3 tokenizer cached |
| `test_cohere.py` | Every Cohere surface form: Command R7B action blocks, the special-token-stripped decode, Command-R `Action:` JSON, parallel calls, string parameters, no parameters, unicode, and plans containing brackets. Also: plain answers, scoring through `ToolCallF1` / `ParallelToolCalls` / `ToolCallValidity`, Cohere calls in AgentTune event logs, and the **real** Command R7B / Aya Expanse / Tiny Aya chat templates. | offline; the template tests need the gpt2 tokenizer cached |
| `test_hf.py` | The `hf:` backend with real tokenizers and templates and a stub pipeline. Covers: native tool schemas in the rendered prompt (Qwen3 and Command R7B), the `chat_template_kwargs` precedence, refusal with advice for tool-less templates (Aya Expanse, Tiny Aya, no template), a single BOS, images (refused on text models, batch order kept), whole `evaluate()` runs in native and prompt mode, and the `agent_eval` harness driving an `hf:` policy. | offline; tokenizers for gpt2, Qwen3-1.7B and Llama-2 cached |
| `test_live.py` | Live runs, **opt-in**: `AK_LIVE_HF=1` (Qwen3-1.7B), `AK_LIVE_COHERE=1` (Tiny Aya Global, gated), `AK_LIVE_R7B=1` (Command R7B, gated, 24 GB+ GPU). | the models, plus an `HF_TOKEN` for Cohere |
| `colab_integrations.ipynb` | **One notebook for all of the above**, on a Colab GPU. | Colab secrets `GITHUB_TOKEN` and `HF_TOKEN` |

```bash
python -m pytest -o addopts= tests/integrations_suite -q -rx            # offline (+ live tests skip)
AK_LIVE_HF=1 python -m pytest -o addopts= tests/integrations_suite/test_live.py -s
```

Local results (M3 Pro, MPS):
- offline: **76 passed, 13 xfailed**;
- live Qwen3-1.7B through `hf:`: `tool_call_f1` = **1.0** in native mode and 1.0 in prompt mode, covering a single call, a parallel pair, a no-tool answer and a different tool.

The Cohere live tests need a Hub token with the licences accepted, so run them from the notebook.

## Findings (strict `xfail`)

| Area | Test | Finding |
|---|---|---|
| Cohere | `test_cohere.py::test_the_cohere_call_id_is_kept` | `normalize_call` reads the id from `id` only, so Cohere's `tool_call_id` is dropped. |
| Cohere | `test_cohere.py::test_cohere_call_result_pairing_is_observed` | As a consequence, call/result pairing is "unavailable" for Cohere event logs, even when the ids match. |
| Cohere | `test_cohere.py::test_a_parameterless_cohere_call_reaches_the_trace` | `{"tool_name": "get_time"}` with no `parameters` is counted in `n_tool_calls` but left out of `to_trace()`, so the metrics see no call. |
| Cohere | `test_cohere.py::test_command_r_directly_answer_is_not_a_tool_call` | Command-R's `directly-answer` pseudo-tool is parsed as a real call. |
| Cohere | `test_cohere.py::test_a_truncated_action_list_is_reported_not_dropped` | An action list cut off by `max_tokens` parses to no calls; the Hermes path reports a `parse_error`. |
| `hf:` + Cohere | `test_cohere.py::test_openai_string_arguments_render_as_an_object_for_command_r7b` | OpenAI-format history (`arguments` as a JSON string) reaches the Command R7B template unchanged, so the model sees `"parameters": "{\"a\": 2}"` in its own history. |
| `hf:` | `test_hf.py::test_a_mixed_batch_gives_every_prompt_one_bos` | `add_special_tokens` is decided by the first request of a batch: [raw, templated] gives the templated prompt two BOS tokens. |
| AgentTune | `test_agenttune.py::test_nameless_trace_calls_are_not_scored_as_no_calls` | TraceLogger calls carry arguments but no tool name. They are counted, but `to_trace()` emits `tool_calls=[]` with no "unavailable" marker, so `ToolCallF1` = **0.0**, as if the agent never called a tool. |
| AgentTune | `test_agenttune.py::test_inspect_does_not_offer_name_based_metrics_without_names` | `inspect_agenttune` lists `tool_call_f1` / `trajectory_match` / `parallel_tool_calls` for those files. |
| AgentTune | `test_agenttune.py::test_rescore_says_why_a_metric_was_not_scored` | `rescore` gives empty diagnostics **and** an empty `ineligible` map for them, so the metric silently disappears. |
| `agent:` | `test_agenttune.py::test_an_answer_only_reply_from_a_server_side_tool_agent_is_unscored` | The "tool calls unavailable" marker needs the request to offer tools. A deployed agent with server-side tools that answers without a transcript is scored as observed zero calls. |
| `agent_eval` harness | `test_hf.py::test_the_harness_executes_text_tool_calls_from_an_hf_model` | The harness reads calls only from a structured trace. An `hf:`/`vllm:`/Cohere model's text calls are taken as the final answer, the tool never runs, and the state oracle **blames the agent**. Verified live: Qwen3-1.7B called `transfer`, the state stayed unchanged, outcome = failure. |
| `agent_eval` harness | `test_hf.py::test_the_harness_passes_a_generation_budget` | Harness requests carry no generation settings, so an `hf:` policy runs at 128 tokens, and a thinking model is cut off before it calls. |

## Using these integrations

`examples/16_agentic_endpoints_live.ipynb` is the user-facing walkthrough with real models:
- `hf:` native and prompt-mode tools;
- multi-turn;
- Cohere through `hf:` and the hosted API;
- a deployed `agent:`;
- the harness with a verified final state.
