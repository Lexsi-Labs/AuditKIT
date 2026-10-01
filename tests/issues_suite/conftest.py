"""Fixtures for the issues suite (#43, #44, #45, and the vLLM named-template fix).

Real data, used the way users use it:
- BFCL v3 at a pinned Hub revision. ``AK_BFCL_ROOT`` points at a downloaded copy;
  otherwise the Hub cache is tried (``AK_BFCL_DOWNLOAD=1`` downloads it). Without
  either, the BFCL tests skip.
- Real tokenizers and chat templates from the local HF cache (skip when absent).
- Live: ``AK_LIVE_HF=1`` starts a real model (``AK_ISSUES_MODEL``, default
  Qwen/Qwen3-1.7B) behind four OpenAI-compatible personas: sglang, vllm, proxy, bare
  (see serve_model.py).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_XDG = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
os.environ.setdefault("HF_HOME", os.path.join(_XDG, "huggingface"))

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"


def pytest_configure(config):
    config.addinivalue_line("markers", "live: runs a real model (opt-in: AK_LIVE_HF=1)")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def bfcl_root() -> Path:
    root = os.environ.get("AK_BFCL_ROOT")
    if root:
        return Path(root)
    hub = pytest.importorskip("huggingface_hub")
    try:
        return Path(hub.snapshot_download(
            BFCL_REPO, repo_type="dataset", revision=BFCL_REVISION,
            allow_patterns=["possible_answer/*", "BFCL_v3_*.json"],
            local_files_only=os.environ.get("AK_BFCL_DOWNLOAD") != "1"))
    except Exception as e:  # not cached, no network
        pytest.skip(f"BFCL {BFCL_REVISION[:7]} not available ({type(e).__name__}); "
                    f"set AK_BFCL_ROOT or AK_BFCL_DOWNLOAD=1")


def cached_tokenizer(name: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        pytest.skip(f"tokenizer {name!r} is not in the local HF cache ({type(e).__name__})")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live():
    """{persona: base_url} for a real model served by serve_model.py, plus "stats"."""
    if os.environ.get("AK_LIVE_HF") != "1":
        pytest.skip("live: set AK_LIVE_HF=1 (loads a real model)")
    model = os.environ.get("AK_ISSUES_MODEL", "Qwen/Qwen3-1.7B")
    ports = {p: _free_port() for p in ("sglang", "vllm", "proxy", "bare")}
    env = {**os.environ, "PYTHONPATH": f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.Popen([sys.executable, str(HERE / "serve_model.py"), model,
                             *[f"{p}:{port}" for p, port in ports.items()]],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    deadline, log = time.time() + 600, []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        log.append(line)
        if line.startswith("ready"):
            break
    else:
        proc.kill()
        pytest.fail("model server did not start:\n" + "".join(log[-20:]))
    if proc.poll() is not None:
        pytest.fail("model server exited:\n" + "".join(log[-20:]))
    urls = {p: f"http://127.0.0.1:{port}/v1" for p, port in ports.items()}
    yield {**urls, "model": model}
    proc.terminate()
    proc.wait(timeout=30)


def served_requests(base_url: str) -> dict:
    import requests
    return requests.get(f"{base_url}/_stats", timeout=5).json()
