"""Lexsi SDK login and project resolution. The only place in
``AuditKIT-Internal`` that imports ``lexsi_sdk``.

``lmeval_engine.py``, ``LexsiCompletionsRelay``, and the ``lexsi:``/``api:``
lm-eval backends stay SDK-free, talking to the gateway over plain HTTP with
whatever ``api_key=``/``project_name=`` this resolves, or that a caller
supplies directly instead of using this at all.

``lexsi_sdk`` is an optional extra: ``pip install auditkit[lexsi-sdk]``. The
import is lazy, so this module imports fine without it. Calling
:func:`lexsi_login`/:func:`_lexsi_login` requires the extra.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .errors import AuditKitError, ExtraNotInstalled


def _current_username(lexsi: Any) -> Optional[str]:
    """Best-effort fetch of the logged-in user's own username, for ``client_id=``.

    ``lexsi_sdk``'s own docs (``image.py``'s ``register_case``) describe
    ``client_id`` as "Lexsi username or client ID", so the bare username this
    returns is a legitimate value for it, and something the gateway accepts.

    ``lexsi.organization(name)`` already calls this same endpoint
    (``USER_ORGANIZATION_URI``) internally, but keeps only ``details`` (the
    org list) and discards the top-level ``current_user`` field. This makes
    its own call to recover it. It returns ``None`` and never raises when the
    endpoint's shape changes, so a missing ``client_id`` never fails an
    otherwise-successful login.
    """
    try:
        from lexsi_sdk.common.xai_uris import USER_ORGANIZATION_URI

        res = lexsi.api_client.get(USER_ORGANIZATION_URI)
        return res.get("current_user", {}).get("username") or None
    except Exception:
        return None


def _lexsi_login(
    org: str,
    workspace: str,
    project: str,
    token: Optional[str] = None,
    api_url: str = "https://apidev.lexsi.ai",
    app_url: str = "https://dev.lexsi.ai",
) -> tuple[str, str, Optional[str]]:
    """Log in via ``lexsi_sdk`` and resolve a project. Returns
    ``(token, project_id, client_id)``.

    ``token`` falls back to the ``SDK_ACCESS_TOKEN`` env var. Requiring a
    literal token as the only option encourages pasting real credentials into
    a notebook cell, so this avoids that. ``lexsi_sdk`` reads the token from
    ``os.environ["SDK_ACCESS_TOKEN"]``, not a constructor arg, so it is set
    there for the duration of this call only: the previous value (or its
    absence) is restored before returning, even on error, so an explicit
    token never lingers in the process environment. Everything that needs it
    (login and project resolution) happens inside this call; later requests
    use the returned session token.

    ``client_id`` is the logged-in user's own username (see
    :func:`_current_username`), or ``None`` if it couldn't be determined. When
    that happens, the caller needs to pass ``client_id=`` explicitly.

    ``lexsi_sdk`` builds its HTTP client as a module-level singleton at
    import time: ``lexsi_sdk/__init__.py`` runs ``lexsi = LEXSI()`` the first
    time it (or any submodule) is imported anywhere in the process, and
    ``LEXSI.__init__`` reads ``XAI_API_URL`` right then to build
    ``api_client.base_url`` (default ``https://apiv1.lexsi.ai`` if unset, the
    prod gateway, not dev). Setting ``os.environ["XAI_API_URL"]`` after that
    first import has no effect for the rest of the process. So the env vars
    here are set before importing, and ``api_client.base_url`` is also
    force-set directly afterward, to stay correct even when an earlier call
    in the same process already imported ``lexsi_sdk`` with different or no
    env vars.
    """
    resolved_token = token or os.environ.get("SDK_ACCESS_TOKEN")
    if not resolved_token:
        raise AuditKitError(
            "Lexsi login needs an SDK access token. Set SDK_ACCESS_TOKEN in "
            "your environment, or pass sdk_access_token= explicitly."
        )
    os.environ["XAI_API_URL"] = api_url
    os.environ["XAI_APP_URL"] = app_url
    previous_token = os.environ.get("SDK_ACCESS_TOKEN")
    os.environ["SDK_ACCESS_TOKEN"] = resolved_token
    try:
        try:
            from lexsi_sdk import lexsi
        except ImportError:
            raise ExtraNotInstalled("lexsi-sdk", "pip install auditkit[lexsi-sdk]")

        lexsi.api_client.base_url = api_url  # see note above: import-time env read, so this forces it directly too
        lexsi.login()
        login_token = lexsi.api_client.auth_token
        client_id = _current_username(lexsi)
        organization = lexsi.organization(org)
        ws = organization.workspace(workspace)
        proj = ws.project(project)
        return login_token, proj.project_name, client_id
    finally:
        # scoped to this call: never leave an explicit token in the process environment
        if previous_token is None:
            os.environ.pop("SDK_ACCESS_TOKEN", None)
        else:
            os.environ["SDK_ACCESS_TOKEN"] = previous_token


def lexsi_login(
    org_name: str,
    workspace_name: str,
    project_name: str,
    *,
    sdk_access_token: Optional[str] = None,
    api_url: str = "https://apidev.lexsi.ai",
    app_url: str = "https://dev.lexsi.ai",
) -> tuple[str, str, Optional[str]]:
    """Public one-call Lexsi login and project resolution. Returns
    ``(token, project_id, client_id)``.

    ``sdk_access_token`` falls back to the ``SDK_ACCESS_TOKEN`` env var when
    not passed explicitly; an explicit value always wins. ``api_url``/
    ``app_url`` default to the dev gateway. Pass your own to point at a
    different environment. ``client_id`` is the logged-in user's own
    username, or ``None`` if it couldn't be determined (see
    :func:`_current_username`); pass your own ``client_id=`` explicitly
    wherever you'd otherwise use it.

    Call this once and reuse the result across several
    :func:`~auditkit.api.run_lmeval` calls, since each login is a real
    network round trip. For a single call, pass
    ``lexsi_org=``/``lexsi_workspace=``/``lexsi_project=`` straight to
    :func:`~auditkit.lmeval_engine.run_benchmark`/``ak.run_lmeval`` instead
    and let it do the login internally::

        TOKEN, PROJECT_ID, CLIENT_ID = ak.lexsi_login("Plans Testt", "TextNewFlow", "Evals Benchmark A")
        ak.run_lmeval(..., api_key=TOKEN, project_name=PROJECT_ID, client_id=CLIENT_ID, ...)

        # or, one-shot:
        ak.run_lmeval(..., lexsi_org="Plans Testt", lexsi_workspace="TextNewFlow",
                       lexsi_project="Evals Benchmark A")
    """
    resolved_token = sdk_access_token or os.environ.get("SDK_ACCESS_TOKEN")
    return _lexsi_login(org_name, workspace_name, project_name, token=resolved_token,
                         api_url=api_url, app_url=app_url)
