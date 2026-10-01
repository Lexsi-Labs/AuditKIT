"""A real model behind OpenAI-compatible servers that behave like the ones users deploy.

One process loads one model through AuditKIT's hf: backend and serves it on several
ports, one per persona. A persona reports its own ``owned_by`` on ``GET /v1/models``
and applies (or drops) ``chat_template_kwargs`` / ``parallel_tool_calls`` as that
server does:

- ``sglang``: applies chat_template_kwargs; parallel_tool_calls=false only under
  tool_choice "required" or a named function (docs/SGLANG.md)
- ``vllm``:   applies both
- ``proxy``:  owned_by "openai" (a LiteLLM-style gateway); drops both
- ``bare``:   no /v1/models route; drops both

Usage: python serve_model.py MODEL PERSONA:PORT [PERSONA:PORT ...]
``GET /_stats`` returns the number of chat requests each persona answered.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from auditkit.model import Request
from auditkit.model.hf_gen import HFGenModel
from auditkit.trace import parse_tool_calls, strip_tool_calls

MODEL = sys.argv[1]
PERSONAS = {
    "sglang": {"owned_by": "sglang", "template_kwargs": True, "parallel": "forced_only"},
    "vllm": {"owned_by": "vllm", "template_kwargs": True, "parallel": True},
    "proxy": {"owned_by": "openai", "template_kwargs": False, "parallel": False},
    "bare": {"owned_by": None, "template_kwargs": False, "parallel": False},
}
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
LLM = HFGenModel(MODEL, device=DEVICE, name=f"hf:{MODEL}")
LOCK = threading.Lock()
STATS: dict[str, int] = {}


def clean(messages):
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def handler(persona: str):
    cfg = PERSONAS[persona]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/").endswith("/_stats"):
                return self._send(200, STATS)
            if cfg["owned_by"] is None:
                return self._send(404, {"error": "not found"})
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": cfg["owned_by"]}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            STATS[persona] = STATS.get(persona, 0) + 1
            params = {"messages": clean(body["messages"]), "max_tokens": body.get("max_tokens") or 512,
                      "temperature": body.get("temperature") or 0.0}
            if cfg["template_kwargs"] and body.get("chat_template_kwargs"):
                params["chat_template_kwargs"] = body["chat_template_kwargs"]
            if body.get("tools"):
                params["tools"] = body["tools"]
            with LOCK:
                g = LLM.generate([Request(prompt="", request_type="chat", params=params)])[0].completions[0]
            raw = g.text or ""
            thought, _, text = raw.rpartition("</think>") if "</think>" in raw else ("", "", raw)
            if "<think>" in text and "</think>" not in raw:        # cut off while thinking
                thought, text = text, ""
            text = text.strip()
            calls = [c for c in parse_tool_calls(text) if not c.parse_error] if body.get("tools") else []
            choice = body.get("tool_choice")
            forced = choice == "required" or isinstance(choice, dict)
            cap = body.get("parallel_tool_calls") is False and (
                cfg["parallel"] is True or (cfg["parallel"] == "forced_only" and forced))
            if cap:
                calls = calls[:1]
            msg = {"role": "assistant", "content": strip_tool_calls(text).strip() if calls else text}
            if thought.strip():
                msg["reasoning_content"] = thought.replace("<think>", "").strip()
            if calls:
                msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                      "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                     for c in calls]
            finish = "tool_calls" if calls else ("length" if not text else "stop")
            self._send(200, {"id": "chatcmpl-local", "object": "chat.completion", "created": int(time.time()),
                             "model": MODEL, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return H


for spec in sys.argv[2:]:
    persona, port = spec.split(":")
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), handler(persona))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"{persona} on :{port}", flush=True)
print(f"ready {MODEL} ({DEVICE})", flush=True)
threading.Event().wait()
