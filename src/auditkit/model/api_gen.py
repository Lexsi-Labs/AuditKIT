"""Generic OpenAI-compatible HTTP API backend (optional extra: auditkit[requests])."""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
from typing import Any
from urllib.parse import urlsplit

from . import (
    Model, Request, Result_, Generated, generate_each, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs, split_session_kwargs,
)
from ..errors import ExtraNotInstalled, ModelError
from ..types import Capability

_log = logging.getLogger(__name__)

# The default (real OpenAI) endpoint. Only when the target host is actually
# OpenAI do we fall back to OPENAI_API_KEY -- pointing api: at a self-hosted
# vLLM/SGLang/Ollama server must never leak the user's real OpenAI key to it.
DEFAULT_API_BASE = "https://api.openai.com/v1"


def _is_openai_host(api_base: str) -> bool:
    """True only for OpenAI's own API host (``api.openai.com`` / ``*.openai.com``)."""
    host = (urlsplit(api_base).hostname or "").lower()
    return host == "openai.com" or host.endswith(".openai.com")


# OpenAI-compatible endpoints (the whole point of this backend) use
# OpenAI's own kwarg names.
_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "stop_sequences": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "num_completions": "n",
    "seed": "seed",
}

# Tool-use fields an adapter (ToolCallAdapter) puts on Request.params; sent
# verbatim in chat mode. parallel_tool_calls=False asks the server for at most
# one call per response.
_TOOL_KEYS = ("tools", "tool_choice", "parallel_tool_calls")

# Request fields a server may drop without a word (#43): an OpenAI-style reply
# never acknowledges them, so every run that sends one records it as unverified
# unless the server kind was declared with APIModel(server=...) and is known to
# honour it. A probed kind (GET /models `owned_by`) only makes the note specific:
# `owned_by` belongs to the deployment, and a proxy in front changes it, so an
# inference never silences the note on its own.
_UNVERIFIED_FIELDS = ("chat_template_kwargs", "parallel_tool_calls")
_KNOWN_SERVERS = ("sglang", "vllm")


def _honours(server: str | None, field: str, body: dict[str, Any]) -> bool:
    """Whether a server of kind *server* is known to apply *field* in *body*."""
    if server == "vllm":
        return True
    if server == "sglang":
        if field == "chat_template_kwargs":
            return True
        # docs/SGLANG.md: true is the default; false caps a response at one call
        # only under grammar-constrained output (tool_choice "required" or a
        # named function)
        choice = body.get("tool_choice")
        return body.get(field) is True or choice == "required" or isinstance(choice, dict)
    return False


def _image_part(image: Any) -> dict[str, Any]:
    """One OpenAI ``image_url`` content part: a URL as-is, anything else as a data URI.

    A remote URL is passed straight through rather than inlined, so a hosted
    provider fetches it itself and the request body stays small. PIL images, paths
    and bytes are encoded here. OpenAI's chat API has no content-part type for a
    local file, so a data URI is the only portable form.
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": image}}
        image = _open_image(image)  # a path
    buffer = io.BytesIO()
    image.save(buffer, format=(getattr(image, "format", None) or "PNG"))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _open_image(image: Any) -> Any:
    """Open a path as a PIL image, naming the path if PIL or the file is the problem."""
    try:
        from PIL import Image as PILImage
    except ImportError as e:  # images are the only reason PIL is needed here
        raise ExtraNotInstalled(
            "vision", "reading Sample.images from a path needs pillow: "
            "pip install 'auditkit[vision]'") from e
    try:
        opened = PILImage.open(image)
        opened.load()  # decode now: a closed file handle would fail later, mid-request
        return opened
    except Exception as e:
        raise ValueError(f"could not read an image for the request: {e}") from e


def messages_with_images(request: "Request") -> list[dict[str, Any]]:
    """The request's messages, with ``params["images"]`` attached for a vision model.

    Images go at the **start of the last user turn**, matching ``hf:``, so the two
    backends build the same prompt from the same sample. An already-structured
    content list keeps its order and gains the images ahead of the text; a plain
    string becomes a text part.

    An ``api:`` request used to drop ``Sample.images`` entirely, which is worse
    than an error: a vision eval scored a model that had never seen the image and
    reported the result as real. It now either sends the image or says why it
    cannot.
    """
    messages = [dict(m) for m in resolve_messages(request)]
    images = request.params.get("images")
    if not images:
        return messages
    if isinstance(images, (str, bytes)) or not hasattr(images, "__iter__"):
        images = [images]
    user_turn = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if user_turn is None:
        raise ValueError(
            "this request carries images but its messages have no user turn to attach them "
            "to. Images are only meaningful on a user turn.")
    content = user_turn.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    user_turn["content"] = [_image_part(img) for img in images] + list(content)
    return messages


def message_trace(message: dict[str, Any]) -> dict[str, Any] | None:
    """Trace for one OpenAI chat response message that made tool calls.

    All of one message's ``tool_calls`` are one turn -- several of them is a
    parallel call, which is exactly what the parallel metrics need to see.
    """
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        # 5 or false is not "no calls" (that's [] or null): fail the sample.
        raise ValueError(f"tool_calls must be a list, got {calls!r}"[:300])
    if not calls:
        return None
    return {"tool_calls": [calls], "messages": [message]}


class APIModel(Model):
    """Call any OpenAI-compatible HTTP API.

    Parameters
    ----------
    model
        Model name sent in the API request body.
    api_base
        Base URL of the API endpoint (default: ``https://api.openai.com/v1``).
    api_key
        API key. Falls back to the ``API_KEY`` env var; ``OPENAI_API_KEY`` is
        used only when ``api_base`` points at OpenAI itself, so a custom
        ``api_base`` (a self-hosted vLLM/SGLang/Ollama server) never receives
        your real OpenAI key -- pass ``api_key`` (or set ``API_KEY``) for those.
    chat_template
        If True (default), uses ``/chat/completions`` with messages format.
        If False, uses ``/completions`` with raw prompt.
    server
        The server kind, declared: ``"sglang"`` or ``"vllm"``. Only a declared
        kind known to apply ``chat_template_kwargs`` / ``parallel_tool_calls``
        clears them from ``RunResult.metadata["unverified_request_fields"]``.
        Undeclared, the kind is probed once from ``GET /models`` (``owned_by``)
        and recorded, but the fields stay unverified.
    """

    name = "api"
    threadsafe = False
    # generate() never raises and applies timeout/retries per request; the
    # Runner sets max_retries/retry_delay/timeout from RunConfig.
    per_request = True
    max_retries = 0
    retry_delay = 1.0

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str = DEFAULT_API_BASE,
        api_key: str | None = None,
        chat_template: bool = True,
        name: str = "api",
        server: str | None = None,
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "APIModel")
        # (kind, how it was established); "declared" skips the probe
        self._server = (server.lower(), "declared") if server else None
        self._server_owned_by: str | None = None
        self._unverified: dict[str, str] = {}
        self._notes_lock = threading.Lock()
        self._model_name = model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or os.environ.get("API_KEY")
        # OPENAI_API_KEY is a fallback ONLY for OpenAI's own host. A custom
        # api_base (self-hosted vLLM/SGLang/Ollama, Azure, ...) must never be
        # handed the user's real OpenAI key; it requires an explicit api_key=
        # or API_KEY instead. (agent: is already protected this way.)
        if not self._api_key and _is_openai_host(self._api_base):
            self._api_key = os.environ.get("OPENAI_API_KEY")
        self._chat_template = chat_template
        # verify=/proxies=/cert=/timeout= are real requests.Session concerns,
        # not API request-body fields -- split them out so e.g.
        # APIModel(verify=False) actually disables TLS verification (useful
        # for a self-signed internal endpoint) instead of silently becoming
        # an extra, ignored JSON field in the chat-completions payload.
        self._session_kwargs, self._extra_kwargs = split_session_kwargs(kwargs)
        self._session = None
        self.timeout = self._session_kwargs.get("timeout", 120)
        self.name = name

    def capabilities(self) -> set[Capability]:
        # No vision capability is advertised, and none is: `Capability` has no such
        # member and hf_gen -- the backend that has always handled images -- does not
        # claim one either. Images work in chat mode and are refused in raw mode;
        # that is enforced where the request is built, not advertised here.
        if self._chat_template:
            return {Capability.GENERATE, Capability.CHAT, Capability.TOOLS}
        return {Capability.GENERATE}

    def identity(self) -> dict:
        # api_base and chat_template change results (which server answers, tool
        # support, and whether the trace is parsed), so they must enter the run
        # fingerprint or the same api:<model> spec against a different server
        # returns a stale cached result. api_key stays out; _extra_kwargs are
        # non-secret body fields that only feed the sha256 fingerprint.
        # A declared server changes the run's metadata (which fields are
        # unverified), so it keys the cache; left out when undeclared, so
        # existing api: fingerprints are unchanged.
        declared = {"server": self._server[0]} if self._server and self._server[1] == "declared" else {}
        return {**super().identity(), "api_base": self._api_base,
                "chat_template": self._chat_template, "extra": self._extra_kwargs, **declared}

    def _server_kind(self) -> tuple[str | None, str]:
        """(kind, source): declared, or probed once from ``GET /models``.
        Anything but a recognised ``owned_by`` is (None, "unknown")."""
        if self._server is None:
            owned_by = None
            try:
                resp = self._session.get(f"{self._api_base}/models", timeout=min(float(self.timeout), 10.0))
                resp.raise_for_status()
                entries = [e for e in (resp.json().get("data") or []) if isinstance(e, dict)]
                entry = next((e for e in entries if e.get("id") == self._model_name), entries[0] if entries else {})
                owned_by = entry.get("owned_by")
            except Exception as e:  # noqa: BLE001 -- no /models, a proxy, a non-JSON reply: all "unknown"
                _log.debug("GET %s/models failed: %s", self._api_base, e)
            self._server_owned_by = owned_by if isinstance(owned_by, str) else None
            kind = (self._server_owned_by or "").lower()
            self._server = (kind, "probed") if kind in _KNOWN_SERVERS else (None, "unknown")
        return self._server

    def _note_unverified(self, body: dict[str, Any]) -> None:
        """Record (and warn once per run about) each field in *body* the server
        is not known to honour. The field is still sent: harmless where ignored."""
        fields = [f for f in _UNVERIFIED_FIELDS if body.get(f) is not None]
        if not fields:
            return
        with self._notes_lock:
            kind, source = self._server_kind()
            for field in fields:
                if field in self._unverified or (source == "declared" and _honours(kind, field, body)):
                    continue
                if source == "declared":
                    reason = f"not known to be honoured by the declared server {kind!r}"
                elif source == "probed" and _honours(kind, field, body):
                    reason = (f"server reports owned_by={self._server_owned_by!r}, which honours it; "
                              f"declare APIModel(server={kind!r}) to accept that")
                elif source == "probed":
                    reason = f"not honoured by {kind} (owned_by={self._server_owned_by!r}) for this request"
                else:
                    seen = f"owned_by={self._server_owned_by!r}" if self._server_owned_by else "no GET /models answer"
                    reason = (f"unknown server ({seen}); declare APIModel(server=...) if it honours "
                              f"this field")
                self._unverified[field] = reason
                _log.warning("api: %s was sent to %s but may have been ignored: %s",
                             field, self._api_base, reason)

    def run_notes(self) -> dict[str, Any]:
        with self._notes_lock:
            notes: dict[str, Any] = {}
            if self._server is not None:
                kind, source = self._server
                notes["api_server"] = {"kind": kind, "source": source,
                                       **({"owned_by": self._server_owned_by} if source != "declared" else {})}
            if self._unverified:
                notes["unverified_request_fields"] = dict(self._unverified)
            self._unverified.clear()
            return notes

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import requests
        except ImportError:
            raise ExtraNotInstalled("requests", "pip install auditkit[requests]")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
            "Content-Type": "application/json",
        })
        for attr in ("verify", "proxies", "cert"):
            if attr in self._session_kwargs:
                setattr(self._session, attr, self._session_kwargs[attr])

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_session()
        # Any per-request failure (HTTP 4xx/5xx, connection reset, a redirect
        # to file:/ftp:, a malformed 200) fails only that request: a raise here
        # would make the runner retry the WHOLE batch, re-POSTing every earlier
        # (billed) request. 429/5xx/connection errors retry per request.
        return generate_each(requests, self._one, max_retries=self.max_retries,
                             retry_delay=self.retry_delay, where=f"API {self._api_base}", log=_log)

    def _one(self, r: Request) -> Result_:
        import time

        import requests
        prompt = r.prompt if isinstance(r.prompt, str) else str(r.prompt)
        defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
        gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
        if self._chat_template:
            endpoint = f"{self._api_base}/chat/completions"
            body = {
                "model": self._model_name,
                "messages": messages_with_images(r),
                **gen_kwargs,
            }
            body.update({k: r.params[k] for k in _TOOL_KEYS if r.params.get(k) is not None})
            if r.params.get("chat_template_kwargs"):
                # vLLM / SGLang OpenAI servers accept this field; sent only when set explicitly
                body["chat_template_kwargs"] = dict(r.params["chat_template_kwargs"])
        else:
            if r.params.get("images"):
                # /v1/completions takes a flat prompt string: there is no message
                # to hang a content part on, so the image cannot be represented.
                # Say so rather than sending the text and scoring a model that
                # never saw the picture.
                raise ValueError(
                    f"{self._model_name!r} was given Sample.images but this backend is in "
                    "raw-prompt mode (chat_template=False), where /v1/completions takes a "
                    "string and cannot carry an image. Use chat mode, or the hf: backend.")
            endpoint = f"{self._api_base}/completions"
            body = {
                "model": self._model_name,
                "prompt": prompt,
                **gen_kwargs,
            }
        body.update(self._extra_kwargs)
        if self._chat_template:
            self._note_unverified(body)
        t0 = time.monotonic()
        try:
            resp = self._session.post(endpoint, json=body, timeout=self.timeout)
        except requests.ConnectionError as e:
            if isinstance(e, requests.Timeout):
                raise
            raise ConnectionError(str(e)) from e  # is_transient() retries it
        try:
            resp.raise_for_status()
        except Exception as e:
            # requests' default HTTPError message drops the response
            # body, which is where the real API actually explains what's
            # wrong. Surface it instead of a bare "400 Client Error".
            err = ModelError(f"{e} -- response body: {str(resp.text)[:500]}")
            err.status = getattr(resp, "status_code", None)  # is_transient() retries 429/5xx
            raise err from None
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise ValueError(f"no choices in response: {getattr(resp, 'text', '')[:200]!r}")
        choice = choices[0]
        trace = None
        if self._chat_template:
            message = choice.get("message")
            if message is None:
                raise ValueError(f"no message in first choice: {choice!r}")
            # content is null when the model only calls tools; some
            # vLLM/proxy servers return it as a list of {type,text} parts.
            text = message.get("content")
            trace = message_trace(message)
        else:
            text = choice.get("text")
        if isinstance(text, list):
            text = "".join(p.get("text") or "" for p in text if isinstance(p, dict))
        text = text or ""  # present-null "text"/"content" -> "" (not None)
        raw_usage = data.get("usage") or {}
        usage = {}
        if raw_usage:
            usage = {
                "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                "completion_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }
        completion = Generated(text=text, finish_reason=choice.get("finish_reason"), trace=trace)
        return Result_(completions=[completion], usage=usage, latency_ms=latency_ms)
