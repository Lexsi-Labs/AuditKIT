"""Fixtures for the integrations suite: AgentTune, Cohere tool calls, and the hf: backend.

Offline by default. Real tokenizers and chat templates are loaded from the local
Hugging Face cache (a test skips when its tokenizer is not cached); model weights are
replaced by a stub pipeline. Live runs are opt-in (see test_live.py).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# The run cache is isolated per test through XDG_CACHE_HOME, which Hugging Face also
# uses for its hub cache. Pin HF_HOME to the real cache first so cached tokenizers
# and models stay visible.
_XDG = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
os.environ.setdefault("HF_HOME", os.path.join(_XDG, "huggingface"))

TEMPLATES = Path(__file__).resolve().parents[1] / "fixtures" / "chat_templates"
AGENTTUNE = Path(__file__).resolve().parents[1] / "fixtures" / "lexsi_real" / "agenttune"


def pytest_configure(config):
    config.addinivalue_line("markers", "live: runs a real model (opt-in, see test_live.py)")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def cached_tokenizer(name: str, template: str | None = None):
    """A real tokenizer from the local HF cache (skip when absent), optionally with a
    chat template from tests/fixtures/chat_templates."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # not cached here
        pytest.skip(f"tokenizer {name!r} is not in the local HF cache ({type(e).__name__})")
    if template is not None:
        tok.chat_template = (TEMPLATES / template).read_text()
    return tok


class StubPipeline:
    """Stands in for transformers' text-generation pipeline: a real tokenizer, a
    scripted reply, and a record of every call (prompts and kwargs)."""

    def __init__(self, tokenizer, reply="ok"):
        self.tokenizer, self.reply, self.calls = tokenizer, reply, []

    def __call__(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        return [[{"generated_text": self.reply(p) if callable(self.reply) else self.reply}] for p in prompts]


def stub_hf(tokenizer, reply="ok", name="stub"):
    from auditkit.model.hf_gen import HFGenModel
    m = HFGenModel(name, name=f"hf:{name}")
    m._pipeline = StubPipeline(tokenizer, reply)
    return m


class _Server:
    """A local HTTP server whose JSON reply is set per test via ``respond``."""

    def __init__(self, path: str):
        self.received: list[dict] = []
        self.respond = lambda body: (200, {})
        outer, lock = self, threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with lock:
                    outer.received.append(body)
                status, payload = outer.respond(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._srv.server_address[1]}{path}"

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def agent_server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    srv = _Server("/run")
    yield srv
    srv.close()
