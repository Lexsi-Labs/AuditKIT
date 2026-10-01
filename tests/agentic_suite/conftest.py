"""Fixtures for the agentic suite: isolated run cache, mock HTTP servers, live gating."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a real model server (set AK_LIVE=1 to run)")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own run cache, so no result is served from ~/.cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)


class _Server:
    """A real local HTTP server whose reply is set per test via ``respond``."""

    def __init__(self, path: str):
        self.received: list[dict] = []
        self.respond = lambda body: (200, {})
        lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with lock:
                    outer.received.append({"path": self.path, "body": body})
                status, payload = outer.respond(body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.base = f"http://127.0.0.1:{self._srv.server_address[1]}"
        self.url = self.base + path

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture
def openai_server():
    """Mock OpenAI-compatible server; ``api_base`` is ``server.url``."""
    pytest.importorskip("requests")  # api: backend needs the [requests] extra
    srv = _Server("/v1")
    yield srv
    srv.close()


@pytest.fixture
def agent_server():
    """Mock deployed agent; ``model=f"agent:{server.url}"``."""
    srv = _Server("/run")
    yield srv
    srv.close()


# -- live gating ----------------------------------------------------------------

OLLAMA = os.environ.get("AK_LIVE_BASE", "http://localhost:11434")


def ollama_models() -> dict[str, list[str]]:
    """{model_name: capabilities} from the local Ollama, or {} when unreachable."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return {}
    out = {}
    for n in names:
        try:
            req = urllib.request.Request(OLLAMA + "/api/show", data=json.dumps({"model": n}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                out[n] = json.load(r).get("capabilities", [])
        except Exception:
            out[n] = []
    return out


def live_enabled() -> bool:
    return os.environ.get("AK_LIVE") == "1"


def live_models() -> dict[str, list[str]]:
    """{model: capabilities} for the live run, on any OpenAI-compatible server.

    - ``AK_LIVE_MODELS=a,b`` picks the models; ``AK_LIVE_NATIVE=a`` (or ``all``)
      says which serve native tool calls (vLLM/SGLang need a --tool-call-parser).
    - Otherwise Ollama is asked (``/api/tags`` + ``/api/show`` capabilities).
    - Otherwise ``/v1/models`` lists them and all run in prompt mode only.
    """
    native = os.environ.get("AK_LIVE_NATIVE", "")
    if os.environ.get("AK_LIVE_MODELS"):
        names = [m.strip() for m in os.environ["AK_LIVE_MODELS"].split(",") if m.strip()]
        discovered = ollama_models()
        if native:
            nat = set(names) if native == "all" else {m.strip() for m in native.split(",")}
            return {m: (["tools"] if m in nat else []) for m in names}
        return {m: discovered.get(m, []) for m in names}
    found = ollama_models()
    if found:
        return found
    try:
        with urllib.request.urlopen(OLLAMA + "/v1/models", timeout=5) as r:
            names = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return {}
    nat = set(names) if native == "all" else {m.strip() for m in native.split(",") if m.strip()}
    return {m: (["tools"] if m in nat else []) for m in names}
