"""Tests for LexsiCompletionsRelay / run_lmeval_via_relay (lexsi_relay.py).

LexsiCompletionsRelay itself is tested against a real local Flask "fake
upstream" server (the `fake_upstream` fixture, conftest.py) over real
loopback HTTP -- no mocking of `requests.post` -- to actually prove it
injects project_name/provider/client_id into a real forwarded request.

run_lmeval_via_relay is a thin, explicit-keyword wrapper around
`ak.run_lmeval(..., relay=True)`; its own relay start/stop lifecycle is
exercised where it actually lives now -- test_lmeval_engine.py's
`run_benchmark(relay=True, ...)` tests. Here it's tested as pure delegation:
does it call run_lmeval with the right kwargs.
"""

from __future__ import annotations

import sys

import pytest
import requests as requests_lib

from auditkit.errors import ExtraNotInstalled
from auditkit.lexsi_relay import LexsiCompletionsRelay, run_lmeval_via_relay


def test_extra_not_installed_when_flask_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "flask", None)
    relay = LexsiCompletionsRelay(
        target_url="https://gw/v1/completions", project_name="p",
        provider="Lexsi", client_id="c", token="t",
    )
    with pytest.raises(ExtraNotInstalled):
        relay.start()


class TestLexsiCompletionsRelay:
    def test_start_returns_a_reachable_completions_url(self, fake_upstream):
        target_url, _ = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="proj",
            provider="Lexsi", client_id="me@x.ai", token="tok",
        )
        try:
            base_url = relay.start()
            assert base_url.startswith("http://127.0.0.1:")
            assert base_url.endswith("/v1/completions")
            assert relay.running
        finally:
            relay.stop()
        assert not relay.running

    def test_forwards_request_with_injected_gateway_fields(self, fake_upstream):
        target_url, received = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="proj",
            provider="Lexsi", client_id="me@x.ai", token="tok-123",
        )
        try:
            base_url = relay.start()
            resp = requests_lib.post(base_url, json={"model": "m", "prompt": "hi"}, timeout=10)
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["text"] == "ok"
        finally:
            relay.stop()

        assert received["json"]["project_name"] == "proj"
        assert received["json"]["provider"] == "Lexsi"
        assert received["json"]["client_id"] == "me@x.ai"
        assert received["json"]["prompt"] == "hi"
        assert received["json"]["model"] == "m"
        assert received["auth"] == "Bearer tok-123"

    def test_context_manager_stops_on_exit(self, fake_upstream):
        target_url, _ = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="p", provider="Lexsi",
            client_id="c", token="t",
        )
        with relay as base_url:
            assert relay.running
            requests_lib.post(base_url, json={}, timeout=10)
        assert not relay.running

    def test_context_manager_stops_even_on_exception(self, fake_upstream):
        target_url, _ = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="p", provider="Lexsi",
            client_id="c", token="t",
        )
        with pytest.raises(RuntimeError):
            with relay:
                raise RuntimeError("boom")
        assert not relay.running

    def test_start_twice_is_idempotent(self, fake_upstream):
        target_url, _ = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="p", provider="Lexsi",
            client_id="c", token="t",
        )
        try:
            url1 = relay.start()
            url2 = relay.start()
            assert url1 == url2
        finally:
            relay.stop()

    def test_stop_without_start_is_a_noop(self):
        relay = LexsiCompletionsRelay(
            target_url="https://gw/v1/completions", project_name="p",
            provider="Lexsi", client_id="c", token="t",
        )
        relay.stop()
        assert not relay.running

    def test_stop_twice_is_safe(self, fake_upstream):
        target_url, _ = fake_upstream
        relay = LexsiCompletionsRelay(
            target_url=target_url, project_name="p", provider="Lexsi",
            client_id="c", token="t",
        )
        relay.start()
        relay.stop()
        relay.stop()


class TestRunLmevalViaRelay:
    """Pure delegation: run_lmeval_via_relay(**explicit kwargs) must call
    ak.run_lmeval(model=f"{prefix}:{model_name}", base_url=target_url,
    project_name=, provider=, client_id=, api_key=token, relay=True, ...)."""

    def test_delegates_to_run_lmeval_with_relay_true(self, monkeypatch):
        captured = {}

        def fake_run_lmeval(tasks, **kw):
            captured["tasks"] = tasks
            captured["kw"] = kw
            return "RESULT"

        monkeypatch.setattr("auditkit.api.run_lmeval", fake_run_lmeval)

        result = run_lmeval_via_relay(
            "arc_easy", model_name="my-model", target_url="https://gw/v1/completions",
            project_name="proj", provider="Lexsi", client_id="me@x.ai", token="tok",
            num_fewshot=0, limit=10,
        )

        assert result == "RESULT"
        assert captured["tasks"] == "arc_easy"
        assert captured["kw"] == {
            "model": "lexsi:my-model",
            "base_url": "https://gw/v1/completions",
            "project_name": "proj",
            "provider": "Lexsi",
            "client_id": "me@x.ai",
            "api_key": "tok",
            "relay": True,
            "num_fewshot": 0,
            "limit": 10,
        }

    def test_prefix_override(self, monkeypatch):
        captured = {}

        def fake_run_lmeval(tasks, **kw):
            captured["model"] = kw["model"]
            return "R"

        monkeypatch.setattr("auditkit.api.run_lmeval", fake_run_lmeval)

        run_lmeval_via_relay(
            "t", model_name="m", target_url="https://gw/v1/completions",
            project_name="p", provider="Lexsi", client_id="c", token="t",
            prefix="api",
        )
        assert captured["model"] == "api:m"
