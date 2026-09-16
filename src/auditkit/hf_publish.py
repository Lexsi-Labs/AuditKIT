"""HuggingFace Hub publishing with AuditKIT dataset-card branding.

``brand_hf_repo`` creates a Hub **dataset** repo if needed, uploads the packaged
logo, and writes README.md. Logos ship inside the package
(``auditkit/assets/*.png``) and are copied into the destination repo — image
srcs point at that repo, never a personal Hugging Face CDN URL.

Same pattern as AlignTune ``aligntune.utils.hf_publish`` and CuratorKIT
``curatorkit.utils.hf_publish``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import ExtraNotInstalled

logger = logging.getLogger(__name__)

AUDITKIT_REPO_URL = "https://github.com/Lexsi-Labs/AuditKIT"
LEXSI_URL = "https://lexsi.ai/"

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_LOGO_ASSET = _ASSETS_DIR / "auditkit_logo.png"
_LOGO_REPO_NAME = "auditkit_logo.png"


def _resolve_token(token: Optional[str] = None) -> str:
    token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "No HuggingFace token found. Pass token=..., set HF_TOKEN "
            "or HUGGING_FACE_HUB_TOKEN, or run huggingface-cli login."
        )
    return token


def _hub_import():
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ExtraNotInstalled(
            "interop",
            "pip install auditkit[interop]",
        ) from e
    return HfApi


def _hub_asset_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _upload_packaged_asset(api, repo_id: str, token: str, local: Path, name: str) -> Optional[str]:
    if not local.exists():
        logger.warning("Packaged branding asset missing: %s", local)
        return None
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    return _hub_asset_url(repo_id, name)


def _branding_header(logo_url: Optional[str]) -> str:
    if not logo_url:
        return ""
    return f"""<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse;">
    <tr>
      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{LEXSI_URL}"><img src="{logo_url}" alt="AuditKIT" style="height: 60px; border-radius: 12px;"/></a>
      </td>
    </tr>
  </table>
</div>
"""


def render_dataset_card(
    repo_id: str,
    kind: str = "run",
    method: str = "",
    model: str = "",
    extra_notes: str = "",
    logo_url: Optional[str] = None,
    built_on: str = "",
) -> str:
    """Hub dataset README: YAML tags + method / model table."""
    name = repo_id.split("/")[-1]
    method = method or "evaluate"
    model_shown = model or "—"
    header = _branding_header(logo_url)
    stamp = built_on or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""---
license: other
pretty_name: {name}
task_categories:
  - text-generation
tags:
  - auditkit
  - lexsi-labs
  - evaluation
  - {kind}
---

{header}
# {name}

Built using [AuditKIT]({AUDITKIT_REPO_URL}) — evaluate any model on any dataset and any task.

| | |
|---|---|
| **Method** | {method} |
| **Model** | `{model_shown}` |
| **Artifact** | {kind} |
| **Published** | {stamp} |

{extra_notes}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
```
"""


def brand_hf_repo(
    repo_id: str,
    kind: str = "run",
    method: str = "",
    model: str = "",
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """Create the Hub dataset repo if needed, upload the packaged logo, write README.md.

    Always ``repo_type="dataset"``. Does not upload rows — call after a push helper.
    """
    HfApi = _hub_import()
    token = _resolve_token(token)
    api = HfApi(token=token)
    built_on = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    logo_url = _upload_packaged_asset(api, repo_id, token, _LOGO_ASSET, _LOGO_REPO_NAME)
    readme = render_dataset_card(
        repo_id,
        kind=kind,
        method=method,
        model=model,
        extra_notes=extra_notes,
        logo_url=logo_url,
        built_on=built_on,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    finally:
        os.remove(readme_path)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Branded %s", url)
    return url


def _brand_safe(
    repo_id: str,
    kind: str = "run",
    method: str = "",
    model: str = "",
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """Brand the repo; a branding failure does not undo a successful push."""
    try:
        return brand_hf_repo(
            repo_id,
            kind=kind,
            method=method,
            model=model,
            private=private,
            token=token,
            extra_notes=extra_notes,
        )
    except Exception as e:
        logger.warning("Dataset card branding skipped for %s: %s", repo_id, e)
        return f"https://huggingface.co/datasets/{repo_id}"


def _parquet_safe_value(value: Any) -> Any:
    """Parquet cannot write a struct with no child fields (empty ``{}``)."""
    if isinstance(value, dict):
        if not value:
            return None
        return {k: _parquet_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_parquet_safe_value(v) for v in value]
    return value


def _parquet_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _parquet_safe_value(v) for k, v in row.items()} for row in rows]


def push_rows_to_hub(
    rows: list[dict[str, Any]],
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    kind: str = "run",
    method: str = "",
    model: str = "",
    extra_notes: str = "",
) -> str:
    """Push rows as a Hub dataset, then stamp the Lexsi / AuditKIT card."""
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ExtraNotInstalled("interop", "pip install auditkit[interop]") from e
    safe = _parquet_safe_rows(rows)
    ds = Dataset.from_list(safe) if safe else Dataset.from_dict({"sample_id": []})
    ds.push_to_hub(repo_id, private=private, token=token)
    return _brand_safe(
        repo_id,
        kind=kind,
        method=method,
        model=model,
        private=private,
        token=token,
        extra_notes=extra_notes,
    )
