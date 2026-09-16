"""Lexsi platform model backend with optional MLflow auto-logging."""
from __future__ import annotations

import os
from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, DEFAULT_TEMPERATURE,
    reject_generation_kwargs, split_session_kwargs,
)
from ..errors import ExtraNotInstalled

# The Lexsi platform's real accepted parameter set beyond temperature/
# max_tokens isn't documented in this codebase -- kept deliberately
# conservative (only what was already hardcoded here) rather than guessing
# at unverified field names for a proprietary API.
_KEY_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
}


class LexsiModel(Model):
    """Call the Lexsi platform API with optional MLflow tracking.

    Parameters
    ----------
    model
        Model name on the Lexsi platform.
    api_key
        Lexsi API key. Falls back to ``LEXSI_API_KEY`` env var.
    api_base
        Base URL for the Lexsi API.
    mlflow_tracking_uri
        If set, auto-log every run to MLflow.
    mlflow_experiment
        MLflow experiment name (defaults to "auditkit-lexsi").
    """

    name = "lexsi"

    def __init__(
        self,
        model: str = "lexsi-3.5",
        api_key: str | None = None,
        api_base: str = "https://api.lexsi.ai/v1",
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment: str | None = None,
        name: str = "lexsi",
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "LexsiModel")
        self._model_name = model
        self._api_key = api_key or os.environ.get("LEXSI_API_KEY")
        self._api_base = api_base.rstrip("/")
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._mlflow_experiment = mlflow_experiment or "auditkit-lexsi"
        # verify=/proxies=/cert=/timeout= are real requests.Session concerns,
        # not API request-body fields -- split them out so e.g.
        # LexsiModel(verify=False) actually disables TLS verification
        # instead of silently becoming an extra, ignored JSON field.
        self._session_kwargs, self._extra_kwargs = split_session_kwargs(kwargs)
        self._session = None
        self.name = name
        self._total_requests = 0
        self._total_latency_ms = 0.0

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import requests
        except ImportError:
            raise ExtraNotInstalled("requests", "pip install requests")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        for attr in ("verify", "proxies", "cert"):
            if attr in self._session_kwargs:
                setattr(self._session, attr, self._session_kwargs[attr])

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_session()
        if self._api_key is None:
            raise ValueError("Lexsi API key required (pass api_key= or set LEXSI_API_KEY)")

        mlflow_active = False
        if self._mlflow_tracking_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._mlflow_tracking_uri)
                mlflow.set_experiment(self._mlflow_experiment)
                mlflow_active = True
            except ImportError:
                pass

        results = []
        run_metrics = {}
        defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
        last_gen_kwargs = dict(defaults)

        for r in requests:
            prompt = r.prompt if isinstance(r.prompt, str) else str(r.prompt)
            last_gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            import time
            t0 = time.monotonic()
            resp = self._session.post(
                f"{self._api_base}/completions",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    **last_gen_kwargs,
                    **self._extra_kwargs,
                },
                timeout=self._session_kwargs.get("timeout", 120),
            )
            latency_ms = (time.monotonic() - t0) * 1000
            self._total_requests += 1
            self._total_latency_ms += latency_ms
            try:
                resp.raise_for_status()
            except Exception as e:
                # requests' default HTTPError message drops the response
                # body, which is where the real API actually explains what's
                # wrong. Surface it instead of a bare "400 Client Error".
                raise type(e)(f"{e} -- response body: {resp.text}") from None
            data = resp.json()
            text = data.get("choices", [{}])[0].get("text", "")
            results.append(Result_(completions=[Generated(text=text)], latency_ms=latency_ms))

        if mlflow_active:
            import mlflow
            with mlflow.start_run(run_name=f"lexsi-batch-{self._total_requests}"):
                mlflow.log_param("model", self._model_name)
                mlflow.log_param("temperature", last_gen_kwargs.get("temperature", DEFAULT_TEMPERATURE))
                mlflow.log_param("max_tokens", last_gen_kwargs.get("max_tokens", 1024))
                mlflow.log_param("batch_size", len(requests))
                mlflow.log_metric("total_requests", self._total_requests)
                mlflow.log_metric("avg_latency_ms", self._total_latency_ms / max(self._total_requests, 1))
                if run_metrics:
                    for k, v in run_metrics.items():
                        mlflow.log_metric(k, v)

        return results

    def log_mlflow_params(self, **params: Any) -> None:
        """Manually log parameters to the active MLflow run."""
        try:
            import mlflow
            for k, v in params.items():
                mlflow.log_param(k, v)
        except ImportError:
            pass
