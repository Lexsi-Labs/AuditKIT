"""Evaluate an externally deployed agent over HTTP (stdlib only).

The agent runs its own tool loop (LangGraph, AgentTune, a custom service...);
AuditKit only POSTs the task and reads back the final answer plus whatever
trace the service exposes, so the agent and RAG metrics can score it.

Default contract. Request body::

    {"input": "<last user message>", "messages": [...chat turns...],
     "tools": [...]}            # "tools" only when the sample offers tools

Response (any subset)::

    {"output": "final answer",
     "messages": [...OpenAI-format transcript incl. tool_calls / role: tool...],
     "tool_calls": [[call, call], [call]],    # turns; optional if messages given
     "contexts": ["chunk 1", "chunk 2"]}      # ranked retrieved contexts

An OpenAI chat-completion response (``{"choices": [...]}``) is detected and
read as such. Other shapes: point the ``*_path`` settings at them with dotted
paths (``output_path="data.answer"``, ``"result.0.text"``), or pass
``request_fn``/``response_fn`` for full control.

Tool-call coverage. When tools were offered but the reply exposes neither a
transcript (``messages``) nor an explicit ``tool_calls`` field (a plain-answer
or post-loop chat-completion reply included), the agent's tool use is
unobservable: the trace is marked ``tool_calls_unavailable`` so the reference
tool metrics stay ineligible rather than scoring it as zero calls. An explicit
``"tool_calls": []`` or a transcript is what makes it observed-zero.

Example::

    ak.evaluate(samples, model="agent:http://localhost:8000/invoke",
                scorers=[ak.ToolCallF1(), ak.RetrievalMetrics(k=5)],
                adapter=ak.ToolCallAdapter())
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from . import (Generated, Model, Request, Result_, generate_each, reject_generation_kwargs,
               resolve_messages)
from ..errors import AuditKitError, ModelError
from ..types import Capability

_MISSING = object()
_log = logging.getLogger(__name__)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        # ponytail: agent endpoints must not auto-redirect. urllib copies every
        # header (incl. Authorization) to the redirect target, even on an
        # https->http downgrade, so a token would leak to any host a Location
        # points at. Returning None makes a 3xx raise HTTPError instead.
        return None


# One shared opener that never follows redirects (urlopen's default one does).
_OPENER = urllib.request.build_opener(_NoRedirect)


def _dig(obj: Any, path: str) -> Any:
    """``obj`` at dotted *path* (integer segments index lists), or ``_MISSING``."""
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.lstrip("-").isdigit() and -len(obj) <= int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return _MISSING
    return obj


def _qualname(fn: Optional[Callable]) -> Optional[str]:
    # __qualname__ alone is "<lambda>" for every lambda (and unchanged when a
    # named fn is redefined), so two different response_fns collide and the disk
    # cache replays a stale result. _default_callable_name folds in bytecode +
    # captured config, the same fix metrics/agent.py uses for arg_match.
    if not fn:
        return None
    from . import _default_callable_name
    return _default_callable_name(fn)


def _contexts(ctx: Any) -> Any:
    """Retrieved contexts as text: LangChain/LlamaIndex document dicts
    (``{"page_content": ...}``, ``{"text": ...}``, ``{"content": ...}``) map
    to their text instead of being compared as ``str(dict)``."""
    if isinstance(ctx, str):
        return ctx
    if not isinstance(ctx, list):
        raise ValueError(f"contexts must be a list, got {ctx!r}"[:300])
    out = []
    for c in ctx:
        if isinstance(c, dict):
            text = next((c[k] for k in ("page_content", "text", "content") if isinstance(c.get(k), str)), None)
            if text is None:
                raise ValueError(f"cannot read a context's text from {c!r}"[:300])
            c = text
        out.append(c)
    return out


class AgentEndpointModel(Model):
    """An agent behind an HTTP endpoint, one POST per sample.

    Parameters
    ----------
    model, url
        The endpoint URL (``agent:<url>`` puts it in ``model``).
    headers
        Extra request headers (override the defaults).
    api_key
        Sent as ``Authorization: Bearer``. Falls back to ``AGENT_API_KEY`` only:
        a self-hosted agent must never receive the user's OpenAI key.
    input_key
        Body key for the last user message (default ``"input"``).
    output_path, messages_path, tool_calls_path, contexts_path
        Dotted paths into the JSON response. Missing paths are skipped. With a
        transcript but no output, the last assistant message is the answer.
    extra_body
        Merged into every request body (e.g. ``{"thread_id": ...}``).
    request_fn
        ``Request -> dict``: builds the whole body instead of the default.
    response_fn
        ``decoded JSON -> dict`` with optional keys ``output``, ``messages``,
        ``tool_calls``, ``retrieved_contexts``.

    Generation settings (temperature, ...) are not sent: the agent owns its
    sampling. A LangGraph/AgentTune service returning
    ``{"output": ..., "messages": [...], "contexts": [...]}`` works as is::

        model = AgentEndpointModel("http://localhost:8000/invoke")
        # or ak.evaluate(..., model="agent:http://localhost:8000/invoke")
    """

    name = "agent"
    threadsafe = True  # one independent urllib call per request
    # generate() never raises and applies timeout/retries per request; the
    # Runner sets these three from RunConfig instead of retrying the batch.
    per_request = True
    max_retries = 0
    retry_delay = 1.0

    def __init__(
        self,
        model: str | None = None,
        *,
        url: str | None = None,
        name: str = "agent",
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
        timeout: float = 120,
        input_key: str = "input",
        output_path: str = "output",
        messages_path: str = "messages",
        tool_calls_path: str = "tool_calls",
        contexts_path: str = "contexts",
        state_path: str | None = "final_state",
        artifacts_path: str | None = "artifacts",
        extra_body: dict[str, Any] | None = None,
        request_fn: Optional[Callable[[Request], dict]] = None,
        response_fn: Optional[Callable[[Any], dict]] = None,
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "AgentEndpointModel")
        if kwargs:
            raise AuditKitError(f"AgentEndpointModel got unknown arguments {sorted(kwargs)}")
        url = url or model
        if not url:
            raise AuditKitError(
                "AgentEndpointModel needs the agent's URL: AgentEndpointModel(url=...) "
                "or model='agent:https://host/path'."
            )
        if urllib.parse.urlsplit(url).scheme.lower() not in ("http", "https"):
            # file://, ftp://, data: would read local files / arbitrary data into
            # the error message and results. The agent contract is HTTP only.
            raise AuditKitError(f"AgentEndpointModel needs an http(s) URL, got {url!r}")
        self._model_name = url
        self.url = url
        self.name = name
        api_key = api_key or os.environ.get("AGENT_API_KEY")
        self._headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._headers.update(headers or {})
        self.timeout = timeout
        self.input_key = input_key
        self.output_path = output_path
        self.messages_path = messages_path
        self.tool_calls_path = tool_calls_path
        self.contexts_path = contexts_path
        self.state_path = state_path
        self.artifacts_path = artifacts_path
        self.extra_body = dict(extra_body or {})
        self.request_fn = request_fn
        self.response_fn = response_fn

    def capabilities(self) -> set[Capability]:
        return {Capability.GENERATE, Capability.CHAT, Capability.TOOLS}

    def identity(self) -> dict:
        return {
            "name": self.name, "url": self.url, "input_key": self.input_key,
            "output_path": self.output_path, "messages_path": self.messages_path,
            "tool_calls_path": self.tool_calls_path, "contexts_path": self.contexts_path,
            "extra_body": self.extra_body,
            "request_fn": _qualname(self.request_fn), "response_fn": _qualname(self.response_fn),
            # A non-reversible digest of the auth/headers so two runs against one
            # URL with different credentials (or X-Tenant routing) don't collide
            # on one cached result. No raw secret enters the identity dict.
            "headers_digest": hashlib.sha256(
                json.dumps(sorted(self._headers.items())).encode()).hexdigest()[:16],
        }

    def _body(self, r: Request) -> dict:
        if self.request_fn is not None:
            return self.request_fn(r)
        messages = resolve_messages(r)
        last_user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), r.prompt)
        body: dict[str, Any] = {self.input_key: last_user, "messages": messages}
        if r.params.get("tools"):
            body["tools"] = r.params["tools"]
        body.update(self.extra_body)
        return body

    def _post(self, body: dict) -> Any:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers=self._headers, method="POST")
        try:
            with _OPENER.open(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            err = ModelError(f"agent endpoint {self.url} returned HTTP {e.code} {e.reason} -- response body: {text[:500]}")
            err.status = e.code  # is_transient() retries 429/5xx
            raise err from None
        try:
            return json.loads(raw)
        except ValueError:
            raise ModelError(f"agent endpoint {self.url} did not return JSON: {raw[:300]!r}") from None

    def _parse(self, data: Any) -> tuple[dict[str, Any], Optional[str]]:
        """``(fields, finish_reason)``; fields has output/messages/tool_calls/retrieved_contexts."""
        if self.response_fn is not None:
            return dict(self.response_fn(data) or {}), None
        if isinstance(data, dict) and "choices" in data:
            from .api_gen import message_trace
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            return {"output": message.get("content") or "", **(message_trace(message) or {})}, choice.get("finish_reason")
        fields = {}
        for key, path in (("output", self.output_path), ("messages", self.messages_path),
                          ("tool_calls", self.tool_calls_path), ("retrieved_contexts", self.contexts_path),
                          ("final_state", self.state_path), ("artifacts", self.artifacts_path)):
            if not path:
                continue
            val = _dig(data, path)
            if val is not _MISSING and val is not None:
                fields[key] = val
        return fields, None

    def generate(self, requests: list[Request]) -> list[Result_]:
        # Any failure (HTTP error, bad JSON, a malformed reply, IncompleteRead,
        # RecursionError on deep JSON) fails only its own request: a raise here
        # would make the runner retry the WHOLE batch, re-POSTing every earlier
        # agent request and re-running its side-effecting tools.
        return generate_each(requests, self._one, max_retries=self.max_retries,
                             retry_delay=self.retry_delay, where=f"agent {self.url}", log=_log)

    def _one(self, r: Request) -> Result_:
        t0 = time.monotonic()
        data = self._post(self._body(r))
        latency_ms = (time.monotonic() - t0) * 1000
        fields, finish_reason = self._parse(data)
        messages = fields.get("messages")
        output = fields.get("output")
        if output is None and messages:
            last = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), {})
            output = last.get("content") or ""
        structured = output is not None and not isinstance(output, str)
        if output is None:
            output = ""
        elif structured:
            output = json.dumps(output)
        trace = {k: fields[k] for k in ("messages", "tool_calls", "retrieved_contexts") if fields.get(k) is not None}
        # endpoint-supplied state/artifacts ride along for state and artifact oracles
        trace.update({k: fields[k] for k in ("final_state", "artifacts") if isinstance(fields.get(k), dict)})
        if "tool_calls" in trace and not isinstance(trace["tool_calls"], (list, dict, str)):
            # 5 or false is not "no calls" (that's [] or absent): fail the sample.
            raise ValueError(f"tool_calls must be a list, got {trace['tool_calls']!r}")
        if "retrieved_contexts" in trace:
            trace["retrieved_contexts"] = _contexts(trace["retrieved_contexts"])
        # Whether the agent's tool use is OBSERVABLE: it returned a transcript or
        # an explicit tool_calls field (even []). Computed before the guards below
        # add a placeholder tool_calls of their own.
        observed_calls = ("tool_calls" in trace) or bool(trace.get("messages"))
        if structured and not trace.get("messages"):
            # A structured (dict) answer with no transcript is an answer, not
            # text to parse for tool calls. Mark an explicit empty turn list
            # so the metrics' text fallback (predicted_turns) is skipped and
            # it isn't scored as a phantom call.
            trace.setdefault("tool_calls", [])
        if not observed_calls:
            # The reply exposed neither a transcript nor a tool_calls field, so whether
            # the agent used any tool is UNOBSERVABLE -- whether or not the request
            # offered tools: a deployed agent (e.g. AgentTune-served) usually keeps its
            # tools server-side. An agent that really made no calls says "tool_calls": [].
            # Mark coverage unavailable (finding 5) so the reference tool metrics
            # stay ineligible instead of scoring it as observed-zero-calls. Only a
            # marker (never a placeholder tool_calls=[]) so an offline agent_eval
            # importer still reads the absent tool_calls as "no trace", not "zero
            # calls observed".
            trace["tool_calls_unavailable"] = True
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        usage = {}
        if isinstance(raw_usage, dict):
            usage = {k: raw_usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return Result_(
            completions=[Generated(text=output, finish_reason=finish_reason, trace=trace or None)],
            usage=usage, latency_ms=latency_ms,
        )
