"""Shared pytest fixtures, and the ``network`` marker."""

from __future__ import annotations

import os
import threading
import urllib.request

import pytest

_HUB_URL = "https://huggingface.co"
_hub_reachable: bool | None = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: needs the Hugging Face Hub; skipped when HF_HUB_OFFLINE=1 or the Hub is unreachable")


def hub_reachable() -> bool:
    """One HEAD request per run (urllib honours HTTPS_PROXY); HF_HUB_OFFLINE=1 means offline."""
    global _hub_reachable
    if _hub_reachable is None:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
            _hub_reachable = False
        else:
            try:
                urllib.request.urlopen(urllib.request.Request(_HUB_URL, method="HEAD"), timeout=3)
                _hub_reachable = True
            except Exception:  # noqa: BLE001 -- any failure means "not reachable from here"
                _hub_reachable = False
    return _hub_reachable


def pytest_runtest_setup(item):
    if item.get_closest_marker("network") and not hub_reachable():
        pytest.skip("needs the Hugging Face Hub (offline, unreachable, or HF_HUB_OFFLINE=1)")


@pytest.fixture
def fake_upstream():
    """A real local server standing in for the Lexsi gateway's completions
    endpoint -- records the JSON body/headers it received and returns a
    canned completions-shaped response. Used by any test exercising
    LexsiCompletionsRelay (directly or via run_benchmark(relay=True)) over
    real loopback HTTP instead of mocking requests.post."""
    pytest.importorskip("flask", reason="the relay tests need the [relay] extra (flask)")
    pytest.importorskip("werkzeug", reason="the relay tests need the [relay] extra (werkzeug)")
    from flask import Flask, jsonify
    from flask import request as flask_request
    from werkzeug.serving import make_server

    app = Flask("fake-upstream")
    app.logger.disabled = True
    received: dict = {}

    @app.route("/v1/completions", methods=["POST"])
    def upstream():
        received["json"] = flask_request.get_json(force=True)
        received["auth"] = flask_request.headers.get("Authorization")
        return jsonify({"choices": [{"text": "ok", "logprobs": {}}]}), 200

    server = make_server("127.0.0.1", 0, app)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/completions", received
    finally:
        server.shutdown()
        thread.join(timeout=5)
