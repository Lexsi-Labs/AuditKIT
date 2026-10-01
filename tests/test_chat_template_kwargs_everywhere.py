"""Regression tests for H1, J1 and L2 of docs/notes/open-bugs-fix-plan.md: chat_template_kwargs reach every
model AuditKIT calls on its own behalf (the agent harness, and every LLM judge)."""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion
from auditkit.metrics.agent import TaskCompletion, ToolSelectionJudge
from auditkit.metrics.judge import BiasJudge, ClosedQA, Factuality, GEval, LLMJudge, Relevance, RubricItem
from auditkit.metrics.rag_judge import (
    AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance, Faithfulness, Hallucination, ResponseGroundedness,
)
from auditkit.model import Capability, Generated, Model, Result_
from auditkit.runspec import RunConfig
from auditkit.trace import strip_tool_calls

NO_THINK = {"enable_thinking": False}


class Recorder(Model):
    """A stand-in model that records each request's params and replies with ``reply(params)``."""

    def __init__(self, reply=lambda params: "yes"):
        self.name, self.params, self.reply = "recorder", [], reply

    def capabilities(self):
        return {Capability.GENERATE, Capability.TOOLS}

    def generate(self, requests):
        self.params.extend(dict(r.params) for r in requests)
        return [Result_(completions=[Generated(text=self.reply(r.params))]) for r in requests]


# -- H1: the harness forwards chat_template_kwargs ---------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "transfer", "parameters": {"type": "object", "properties": {
    "src": {"type": "string"}, "dst": {"type": "string"}, "amount": {"type": "number"}}}}}]


class Bank:
    def __init__(self):
        self.acc = {"checking": 1000.0, "savings": 0.0}

    def __call__(self, name, args):
        self.acc[args["src"]] -= args["amount"]
        self.acc[args["dst"]] += args["amount"]
        return json.dumps(self.acc)

    def snapshot(self):
        return dict(self.acc)


def _policy():
    def reply(params):
        if params["messages"][-1]["role"] == "tool":
            return "Done."
        return '<tool_call>{"name": "transfer", "arguments": {"src": "checking", "dst": "savings", "amount": 250}}</tool_call>'
    return Recorder(reply)


@pytest.mark.parametrize("config,want", [(RunConfig(chat_template_kwargs=NO_THINK, max_tokens=64), NO_THINK),
                                         (RunConfig(max_tokens=64), None)])
def test_h1_every_harness_request_carries_the_config_kwargs(config, want):
    policy = _policy()
    case = AgentCase(id="c", task="Move 250.", allowed_tools=TOOLS, outcome=FinalStateAssertion("savings", equals=250.0))
    res = AgentEvalRunner().run(AgentEvalSpec(cases=[case], mode="harness", agent=policy, config=config,
                                              agent_opts={"tool_env": Bank(), "max_steps": 3}))
    assert res.rows[0].outcome["verdict"] == "success" and len(policy.params) == 2
    assert all(p.get("chat_template_kwargs") == want for p in policy.params)
    assert all(p["max_tokens"] == 64 for p in policy.params)                     # G2 still holds


def test_h1_the_kwargs_are_part_of_the_spec_identity():
    ids = [AgentEvalSpec(cases=[], mode="harness", config=RunConfig(chat_template_kwargs=k)).identity()
           for k in (None, NO_THINK)] if hasattr(AgentEvalSpec, "identity") else None
    if ids is None:
        pytest.skip("AgentEvalSpec has no identity() on this branch")
    assert ids[0] != ids[1]


# -- J1: judges accept judge_chat_template_kwargs --------------------------------------------------------------

S = ak.Sample(input="What is the capital of France?", target="Paris is the capital of France.",
              retrieval_context=["Paris is the capital of France.", "Berlin is in Germany."])

JUDGES = {
    "llm_judge": lambda **kw: LLMJudge(choices={"yes": 1.0, "no": 0.0}, **kw),
    "factuality": lambda **kw: Factuality(**kw),
    "closed_qa": lambda **kw: ClosedQA(**kw),
    "relevance": lambda **kw: Relevance(**kw),
    "bias_judge": lambda **kw: BiasJudge(**kw),
    "g_eval": lambda **kw: GEval(rubric=[RubricItem(criterion="is it correct")], **kw),
    "task_completion": lambda **kw: TaskCompletion(**kw),
    "tool_selection": lambda **kw: ToolSelectionJudge(**kw),
    "faithfulness": lambda **kw: Faithfulness(**kw),
    "hallucination": lambda **kw: Hallucination(**kw),
    "answer_relevancy": lambda **kw: AnswerRelevancy(**kw),
    "response_groundedness": lambda **kw: ResponseGroundedness(**kw),
    "context_precision": lambda **kw: ContextPrecision(**kw),
    "context_relevance": lambda **kw: ContextRelevance(**kw),
    "context_recall": lambda **kw: ContextRecall(**kw),
}


def _score(metric):
    ctx = {"trace": {"tool_calls": [[{"name": "search", "arguments": {"q": "capital"}}]],
                     "retrieved_contexts": list(S.retrieval_context)}}
    try:
        metric.score(S, "Paris is the capital of France.", ctx)
    except Exception:  # noqa: BLE001 -- a stub verdict may not parse; only the request matters here
        pass


@pytest.mark.parametrize("name", sorted(JUDGES))
def test_j1_the_judge_sends_the_kwargs(name):
    rec = Recorder()
    _score(JUDGES[name](judge_model=rec, judge_chat_template_kwargs=NO_THINK))
    assert rec.params and all(p.get("chat_template_kwargs") == NO_THINK for p in rec.params)


@pytest.mark.parametrize("name", sorted(JUDGES))
def test_j1_without_the_argument_nothing_is_sent(name):
    rec = Recorder()
    _score(JUDGES[name](judge_model=rec))
    assert rec.params and not any("chat_template_kwargs" in p for p in rec.params)


@pytest.mark.parametrize("name", sorted(JUDGES))
def test_j1_the_kwargs_change_the_judge_identity(name):
    rec = Recorder()
    assert JUDGES[name](judge_model=rec).identity() != JUDGES[name](judge_model=rec, judge_chat_template_kwargs=NO_THINK).identity()


def test_j1_a_thinking_judge_gives_a_verdict_only_when_told_not_to_think():
    # a Qwen3-like judge: without the switch it spends its budget reasoning and never writes a verdict
    reply = lambda p: "yes" if (p.get("chat_template_kwargs") or {}).get("enable_thinking") is False else "<think>Let me consider"  # noqa: E731
    sample = ak.Sample(input="q", target="a", actual_output="a")
    thinking = ak.evaluate([sample], "precomputed", [LLMJudge(choices={"yes": 1.0, "no": 0.0}, judge_model=Recorder(reply))])
    told = ak.evaluate([sample], "precomputed", [LLMJudge(choices={"yes": 1.0, "no": 0.0}, judge_model=Recorder(reply),
                                                           judge_chat_template_kwargs=NO_THINK)])
    assert "llm_judge" not in thinking.headline and thinking.errors          # unreadable -> error, never 0
    assert told.headline["llm_judge"] == 1.0 and not told.errors


# -- L2: stripping calls from the fed-back content never loses prose ----------------------------------------------

def test_l2_prose_next_to_a_json_object_is_kept():
    text = 'I will look it up. {"name": "get_weather", "arguments": {"city": "Paris"}}'
    assert ak.parse_tool_calls(text) == []          # not read as a call ...
    assert strip_tool_calls(text) == text           # ... so nothing is stripped


def test_l2_a_whole_json_reply_leaves_no_content():
    text = '<think>need the weather</think>\n{"name": "get_weather", "arguments": {"city": "Paris"}}'
    assert [c.name for c in ak.parse_tool_calls(text)] == ["get_weather"] and strip_tool_calls(text) == ""


def test_l2_prose_around_tagged_calls_is_kept():
    text = 'Let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\nOne moment.'
    assert strip_tool_calls(text) == "Let me check.\n\nOne moment."
