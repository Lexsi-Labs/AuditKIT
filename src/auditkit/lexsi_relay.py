"""A local relay for the Lexsi gateway's completions endpoint.

lm-eval's ``local-completions`` backend (AuditKIT's ``lexsi:``/``api:`` prefix)
can't be pointed at the Lexsi gateway's completions endpoint directly with
``project_name=``/``provider=``/``client_id=`` the way the chat/generation path
can (see :func:`auditkit.lmeval_engine.run_benchmark`) -- a direct request with
those three fields still gets rerouted server-side to a different, unregistered
path and 404s. :class:`LexsiCompletionsRelay` is a tiny local HTTP server that
injects the three fields and forwards to the real completions URL, so
``ak.run_lmeval(model="lexsi:...", base_url=<relay>)`` reaches it the same way a
direct request to the chat endpoint does.

This is an optional extra: ``pip install auditkit[relay]`` (Flask + Werkzeug).
The import is lazy, so this module imports fine without it -- only
constructing a :class:`LexsiCompletionsRelay` requires the extra.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from .errors import ExtraNotInstalled


class LexsiCompletionsRelay:
    """A local HTTP relay that injects ``project_name``/``provider``/
    ``client_id`` into every request before forwarding it to the real Lexsi
    completions endpoint.

    Use as a context manager -- ``__enter__`` returns the relay's local
    ``base_url`` (already suffixed with ``/v1/completions``, ready to pass
    straight to ``base_url=``), ``__exit__`` always stops it, even if the
    call inside the ``with`` block raises::

        with LexsiCompletionsRelay(
            target_url=LEXSI_COMPLETIONS_URL,
            project_name=PROJECT_ID, provider=PROVIDER, client_id=CLIENT_ID,
            token=TOKEN,
        ) as base_url:
            result = ak.run_lmeval("arc_easy", model=f"lexsi:{MODEL_NAME}",
                                    base_url=base_url, num_fewshot=0, limit=10)

    For a one-call version that starts the relay, runs a benchmark, and stops
    the relay for you, see :func:`run_lmeval_via_relay`.
    """

    def __init__(
        self,
        target_url: str,
        project_name: str,
        provider: str,
        client_id: str,
        token: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._target_url = target_url
        self._project_name = project_name
        self._provider = provider
        self._client_id = client_id
        self._token = token
        self._host = host
        self._port = port
        self._server: Any = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> str:
        """Start the relay (a no-op if it's already running) and return its
        ``base_url``, suffixed with ``/v1/completions``."""
        if self._server is not None:
            return self._base_url()

        try:
            import requests
            from flask import Flask, jsonify
            from flask import request as flask_req
            from werkzeug.serving import make_server
        except ImportError:
            raise ExtraNotInstalled("relay", "pip install auditkit[relay]")

        target_url = self._target_url
        project_name = self._project_name
        provider = self._provider
        client_id = self._client_id
        token = self._token

        app = Flask("lexsi-completions-relay")
        app.logger.disabled = True

        @app.route("/v1/completions", methods=["POST"])
        def completions_proxy():  # noqa: ANN202 -- Flask view, not part of the public API
            body = flask_req.get_json(force=True)
            body["project_name"] = project_name
            body["provider"] = provider
            body["client_id"] = client_id
            resp = requests.post(
                target_url,
                json=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            return jsonify(resp.json()), resp.status_code

        @app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
        @app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
        def catch_all(path):  # noqa: ANN202 -- Flask view, not part of the public API
            return jsonify({"error": "not found"}), 404

        self._server = make_server(self._host, self._port, app)
        self._port = self._server.server_port  # port=0 binds an ephemeral port -- read back the real one
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self._base_url()

    def stop(self) -> None:
        """Stop the relay. Safe to call more than once, or when never started."""
        if self._server is None:
            return
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def _base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1/completions"

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


def run_lmeval_via_relay(
    tasks: Any,
    *,
    model_name: str,
    target_url: str,
    project_name: str,
    provider: str,
    client_id: str,
    token: str,
    prefix: str = "lexsi",
    **run_lmeval_kwargs: Any,
) -> Any:
    """Run :func:`auditkit.run_lmeval` against the Lexsi completions gateway
    through a short-lived local relay -- started just before the call, always
    stopped right after (success or failure), so the caller doesn't have to
    manage the relay's lifetime with a ``with`` block themselves::

        result = run_lmeval_via_relay(
            "arc_easy", model_name=MODEL_NAME, target_url=LEXSI_COMPLETIONS_URL,
            project_name=PROJECT_ID, provider=PROVIDER, client_id=CLIENT_ID,
            token=TOKEN, num_fewshot=0, limit=10,
        )

    ``prefix`` defaults to ``"lexsi:"`` (auto-infers ``tokenizer=`` from
    ``model_name`` -- see :func:`auditkit.lmeval_engine.map_model_spec`);
    pass ``prefix="api"`` to opt out of that inference.

    A thin, explicit-keyword wrapper around ``ak.run_lmeval(..., relay=True)``
    -- the relay's start/stop lifecycle lives in
    :func:`auditkit.lmeval_engine.run_benchmark` (the one place that also
    knows about the disk cache, so the fingerprint stays keyed on the real
    ``target_url``, not the relay's ephemeral local port); this function is
    just a convenient, fully-named-argument call shape for it.
    """
    from .api import run_lmeval

    return run_lmeval(
        tasks,
        model=f"{prefix}:{model_name}",
        base_url=target_url,
        project_name=project_name,
        provider=provider,
        client_id=client_id,
        api_key=token,
        relay=True,
        **run_lmeval_kwargs,
    )
