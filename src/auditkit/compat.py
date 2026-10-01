"""Environment compatibility check for the heavy, version-locked backends.

``check_compat()`` reads installed package metadata (never imports torch,
sglang or vllm unless ``check_cuda=True``) and reports clashes that pip and uv
resolve silently or not at all. Every rule below comes from PyPI metadata plus
``uv pip compile`` runs on linux x86_64 / Python 3.11 (2026-09):

(a) Pins. sglang and vllm pin most of their stack exactly (sglang 0.5.20:
    torch==2.13.0, transformers==5.12.1, flashinfer_python==0.6.18,
    xgrammar==0.2.1, ...). A later ``pip install`` of one of those packages
    breaks the pin without complaint, so every installed dependency of an
    installed sglang/vllm is checked against its declared specifiers -> error.
    Skipped (ponytail: no full PEP 440/508 engine here): requirements with an
    environment marker, which drops vllm's platform-marked llguidance and
    xgrammar pins (rule (b) covers them), and ``~=``, ``===`` and ``.*``
    wildcard clauses such as protobuf's ``!=6.30.*``.
(b) sglang and vllm in one env never co-resolve: llguidance alone blocks every
    pair (e.g. vllm 0.16 needs >=1.3,<1.4; sglang needs <0.8 or >=1.7.6), and
    the closest pair (sglang 0.5.9 / vllm 0.16) also clashes on xgrammar.
    A resolver asked for both "succeeds" by picking sglang 0.5.2, a frontend
    with no runtime -> error.
(c) sglang < 0.5.10 is almost always a silent downgrade: 0.5.9 is the last
    release on transformers 4.x, so a resolver that cannot fit a newer sglang
    next to ``auditkit[transformers]`` or ``auditkit[vllm]`` lands there
    -> warn. sglang without torch is the frontend-only package -> warn.
(d) transformers outside 5.15-5.x: AuditKit's ``hf:`` backend is validated on
    ``transformers>=5.15,<6`` (the ``[transformers]`` extra; 5.15 is the first
    with North Micro Vision). sglang 0.5.20 pins 5.12.1 -> warn, and a second warn
    naming North Micro Vision, which that pin cannot load.
(e) sglang ships Linux-only wheels (manylinux_2_34 from 0.5.11 on, CPython
    3.10-3.13); pip on macOS falls back to an old release (0.5.10). Non-Linux, glibc <
    2.34 or Python outside 3.10-3.13 with sglang -> error. On 3.13 the current release
    does not install (its outlines==0.1.11 pin needs outlines_core 0.1.26, which has
    wheels up to cp312 only) -> warn: build the SGLang venv on 3.12.
(f) AuditKit's ``vllm:`` backend supports vllm >= 0.30 (``[vllm]`` extra)
    -> warn below that.
(g) ``check_cuda=True`` imports torch: no CUDA device -> warn when a GPU
    runtime is installed; sglang >= 0.5.11 is built for CUDA 13, so a torch
    built for an older CUDA -> warn. A torchaudio built for a different CUDA than
    torch (Colab's preinstalled one after ``[vllm]`` upgrades torch) breaks every
    transformers import -> error.
(h) transformers 5 refuses ``device_map`` with accelerate < 1.0 (lm-eval's hf
    backend, ``[lmeval]``) -> warn.
(i) sglang's scheduler calls ``UnifiedRadixCache.dec_lock_ref(node)`` without its
    required ``params`` when the radix cache is disabled, which SGLang does for
    every multimodal model on its Transformers backend. Aya Vision 8B therefore
    loads and then crashes on its first request (measured: sglang 0.5.20, Colab A100,
    2026-09-30; the bug is also on sglang main). Read from the installed source, not
    imported -> warn, with the fix: ``compat.py --patch-sglang`` defaults ``params``
    to None; the disabled path returns before reading it.
(j) transformers >= 5.15 (the first with ``cohere_compass``) adds ``"embed_tokens":
    "embedding_rowwise"`` to the TP plan of tied-embedding models; sglang's Transformers
    backend rejects the style name, so North Micro Vision fails to load on sglang even
    with a new enough transformers (measured: sglang 0.5.20 + transformers 5.16.1). The
    backend reads the plan only for nn.Linear, so mapping it to "replicate" is inert
    -> warn; ``--patch-sglang`` applies it.
(l) sglang's ``Cohere2ForCausalLM`` never reports its sliding window to the runner, so the
    Triton attention backend (used where FlashInfer can't build) crashes on every
    sliding-window layer with ``kv_indptr=None`` (measured: sglang 0.5.20, RTX PRO 6000; also
    on sglang main) -> ``--patch-sglang`` adds the method, returning the layers' own window.
(m) the same gap in sglang's generic Transformers backend (``TransformersBase``), which serves
    Aya Vision: Triton decode then fails with ``kv_indptr=None`` -> ``--patch-sglang`` adds the
    method there too, returning a window only for models that have sliding-window layers.
(n) North's ``rope_parameters`` is keyed by layer type, with ``None`` for its full-attention
    layers (they use no RoPE); sglang's Transformers backend calls ``.get`` on every value while
    deciding whether torch.compile is safe, so North fails to load with "'NoneType' object has no
    attribute 'get'" (measured: sglang 0.5.20 + transformers 5.16.1, RTX PRO 6000; also on sglang
    main) -> ``--patch-sglang`` skips the entries without RoPE.
(k) ``check_cuda=True``: on Hopper/Blackwell (compute capability >= 9) SGLang enables
    DeepGEMM, whose startup JIT needs nvcc >= 12.9; with an older (or no) toolkit every
    model fails to load (measured: Colab RTX PRO 6000) -> warn, naming
    ``SGLANG_ENABLE_JIT_DEEPGEMM=0`` (DeepGEMM serves FP8 GEMMs only).
(o) ``check_cuda=True``: vLLM samples with FlashInfer by default, so on an SM 12.x GPU whose CUDA
    toolkit is older than 12.9 every vLLM model fails to load with "FlashInfer requires GPUs with
    sm75 or higher" (measured: Colab RTX PRO 6000, CUDA 12.8) -> warn. ``vllm:`` switches to
    vLLM's own sampler by itself; ``vllm serve`` needs ``VLLM_USE_FLASHINFER_SAMPLER=0``.
``sglang_launch_advice()`` / ``compat.py --sglang-launch MODEL`` and ``vllm_launch_advice()`` /
``compat.py --vllm-launch`` turn all of this into the env and flags a server needs on the machine
it runs on, whatever the GPU. ``flashinfer_jit_blocked()`` is the one check both use.

The file is stdlib-only and imports nothing from auditkit, so it also runs
as a bare script inside an environment without AuditKit (for example a
dedicated SGLang venv): ``<venv>/bin/python path/to/compat.py [--cuda] [--patch-sglang]``.
"""
from __future__ import annotations

import os
import sys

# Run as a bare script (``python path/to/compat.py``), Python puts this file's
# directory -- src/auditkit/ -- at sys.path[0], where our types.py shadows the
# stdlib ``types`` module. The first stdlib import chain that reaches ``types``
# (platform -> re -> enum) then dies with a circular ImportError. Drop the
# directory before importing anything else. Nothing here needs to import
# auditkit, which is what makes the bare-script mode work in the first place.
_HERE = os.path.dirname(os.path.abspath(__file__))
while _HERE in sys.path:
    sys.path.remove(_HERE)

import platform
import re
from dataclasses import dataclass, field
from importlib import metadata

__all__ = ["CompatError", "CompatReport", "Finding", "check_compat", "patch_sglang", "sglang_launch_advice",
           "vllm_launch_advice"]

_REPORTED = (
    "auditkit", "torch", "transformers", "vllm", "sglang", "flashinfer-python",
    "xgrammar", "llguidance", "openai", "litellm", "accelerate", "torchaudio",
)
_VLLM_MIN = "0.30"                    # the [vllm] extra's floor
_TRANSFORMERS_SUPPORTED = ("5.15", "6")   # [lo, hi) -- the [transformers] extra's range
_COMPASS_MIN_TRANSFORMERS = "5.15"          # first transformers with cohere_compass (North Micro Vision)


# The SGLang bugs AuditKit knows how to fix in place, as (name, file under the package, the
# exact buggy source, its fix). Each is read from the installed source, never imported.
_SGLANG_FIXES = (
    # (i) the scheduler calls dec_lock_ref(node) without params when the radix cache is disabled
    ("dec_lock_ref", ("srt", "mem_cache", "unified_radix_cache.py"),
     "        node_id: NodeId,\n        params: DecLockRefParams,\n        skip_swa: bool = False,\n"
     "    ) -> DecLockRefResult:",
     "        node_id: NodeId,\n        params: DecLockRefParams = None,\n        skip_swa: bool = False,\n"
     "    ) -> DecLockRefResult:"),
    # (j) transformers >= 5.15 adds "embed_tokens": "embedding_rowwise" to the TP plan of every
    # tied-embedding model; SGLang's Transformers backend only reads the plan for nn.Linear, so
    # the entry is inert, but its style table rejects the name before it gets there
    ("embedding_rowwise", ("srt", "models", "transformers.py"),
     '        "moe_tp_experts": "replicate",\n    }.get(style, style)',
     '        "moe_tp_experts": "replicate",\n        "embedding_rowwise": "replicate",\n    }.get(style, style)'),
    # (l) Cohere2ForCausalLM (Tiny Aya, Command R7B) doesn't tell the runner its sliding window, so
    # the Triton attention backend never builds its window metadata and every sliding layer gets
    # kv_indptr=None ("'NoneType' object has no attribute 'type'" in extend_attention). Its layers
    # use config.sliding_window; the method returns the same value (Gemma's does the same).
    ("cohere2_sliding_window", ("srt", "models", "commandr.py"),
     "class Cohere2ForCausalLM(CohereForCausalLM):\n    pass\n",
     "class Cohere2ForCausalLM(CohereForCausalLM):\n"
     "    def get_attention_sliding_window_size(self):\n"
     "        return getattr(self.config, \"sliding_window\", None)\n"),
    # (m) the same gap in SGLang's generic Transformers backend (Aya Vision's cohere2 text model): its
    # layers take a window from layer_types/sliding_window, but TransformersBase never reports it, so
    # Triton decode gets kv_indptr=None. Reported only when the model has sliding layers at all.
    ("transformers_sliding_window", ("srt", "models", "transformers.py"),
     "        return loader.load_weights(weights, mapper=self.weight_mapper)\n\n\nclass CausalMixin:",
     "        return loader.load_weights(weights, mapper=self.weight_mapper)\n\n"
     "    def get_attention_sliding_window_size(self):\n"
     "        layer_types = getattr(self.text_config, \"layer_types\", None) or getattr(self.config, \"layer_types\", None)\n"
     "        if not layer_types or \"sliding_attention\" not in layer_types:\n"
     "            return None\n"
     "        return getattr(self.text_config, \"sliding_window\", None) or getattr(self.config, \"sliding_window\", None)\n"
     "\n\nclass CausalMixin:"),
    # (n) North's rope_parameters is keyed by layer type, with None for its full-attention layers
    # (they use no RoPE); SGLang's torch.compile check calls .get on every value and crashes on
    # the None before the model is built. A layer without RoPE can't have dynamic RoPE: skip it.
    ("rope_parameters_none", ("srt", "models", "transformers.py"),
     '                rp.get("rope_type") == "dynamic" for rp in rope_params.values()\n',
     '                isinstance(rp, dict) and rp.get("rope_type") == "dynamic"\n'
     '                for rp in rope_params.values()\n'),
)
_EMBEDDING_ROWWISE_TRANSFORMERS = "5.15"   # first transformers that emits the style


def _sglang_path() -> str | None:
    """The installed sglang package directory, located without importing it (that pulls torch)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("sglang")
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or []) if spec else []
    return locations[0] if locations else None


def _sglang_fix_state(name: str) -> str | None:
    """'bug', 'patched', or None (no sglang source here, or a layout this check doesn't know)."""
    _, rel, old, new = next(fx for fx in _SGLANG_FIXES if fx[0] == name)
    root = _sglang_path()
    path = os.path.join(root, *rel) if root else None
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    return "patched" if new in src else ("bug" if src.count(old) == 1 else None)


def _dec_lock_state() -> str | None:
    return _sglang_fix_state("dec_lock_ref")


def patch_sglang() -> str:
    """Apply the rule (i), (j), (l), (m) and (n) fixes to the installed sglang; idempotent. Returns what happened,
    one ``name: outcome`` per fix."""
    out = []
    for name, rel, old, new in _SGLANG_FIXES:
        state = _sglang_fix_state(name)
        if state is None:
            out.append(f"{name}: not applied (no sglang with that source is installed here)")
            continue
        if state == "patched":
            out.append(f"{name}: already patched")
            continue
        path = os.path.join(_sglang_path(), *rel)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src.replace(old, new, 1))
        out.append(f"{name}: patched {path}")
    return "; ".join(out)


class CompatError(RuntimeError):
    """Raised by :meth:`CompatReport.raise_on_error`."""


@dataclass
class Finding:
    level: str      # "error" | "warn" | "info"
    package: str
    message: str


@dataclass
class CompatReport:
    findings: list[Finding] = field(default_factory=list)
    versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there is no error-level finding."""
        return not any(f.level == "error" for f in self.findings)

    def raise_on_error(self) -> None:
        if not self.ok:
            raise CompatError(str(self))

    def __str__(self) -> str:
        lines = ["AuditKit compatibility report", ""]
        lines += [f"  {k:<18} {v or '-'}" for k, v in self.versions.items()]
        lines.append("")
        for f in self.findings:
            lines.append(f"  [{f.level.upper():<5}] {f.package}: {f.message}")
        errors = sum(f.level == "error" for f in self.findings)
        lines += ["", "OK" if not errors else f"{errors} error(s)"]
        return "\n".join(lines)


# --- seams (tests monkeypatch these) -----------------------------------------

def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dist_requires(name: str) -> list[str]:
    try:
        return metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return []


def _system() -> str:
    return platform.system()


def _libc_ver() -> str | None:
    """Runtime glibc version, or None (not Linux / musl / unknown)."""
    try:
        name, ver = os.confstr("CS_GNU_LIBC_VERSION").split()   # "glibc 2.35"
        return ver if name == "glibc" else None
    except (AttributeError, ValueError, OSError, TypeError):
        lib, ver = platform.libc_ver()
        return ver if lib == "glibc" and ver else None


def _py_version() -> tuple[int, int, int]:
    return tuple(sys.version_info[:3])


# --- tiny PEP 440 / PEP 508 subset ---------------------------------------------

_VER = re.compile(
    r"v?(?:\d+!)?(?P<rel>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre>a|b|c|rc|alpha|beta|pre|preview)[-_.]?(?P<pren>\d*))?"
    r"(?:[-_.]?(?:post|rev|r)[-_.]?(?P<post>\d*)|-(?P<post2>\d+))?"
    r"(?:[-_.]?dev[-_.]?(?P<dev>\d*))?",
    re.I,
)
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


def _vkey(v: str) -> tuple:
    """Sortable key: dev < a < b < rc < final < post. Local (+cu130) and epoch ignored.

    ponytail: pre-release exclusion rules of ``<``/``>`` are not modelled;
    use ``packaging`` if exact PEP 440 semantics ever matter here.
    """
    m = _VER.match(v.strip().split("+")[0])
    if not m:
        return ((0,), (1,), -1, float("inf"))
    rel = [int(x) for x in m["rel"].split(".")]
    while len(rel) > 1 and rel[-1] == 0:
        rel.pop()
    post = m["post"] if m["post"] is not None else m["post2"]
    has_post = post is not None
    dev = float(m["dev"] or 0) if m["dev"] is not None else float("inf")
    if m["pre"]:
        phase = (0, _PRE_RANK[m["pre"].lower()], int(m["pren"] or 0))
    elif m["dev"] is not None and not has_post:
        phase = (-1,)
    else:
        phase = (1,)
    return (tuple(rel), phase, int(post or 0) if has_post else -1, dev)


def _satisfies(version: str, op: str, target: str) -> bool | None:
    """None = specifier not understood (caller skips it)."""
    if "*" in target or op in ("~=", "==="):
        return None
    a, b = _vkey(version), _vkey(target)
    return {"==": a == b, "!=": a != b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}.get(op)


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*\(?([^;()@]*)\)?\s*(;.*)?$")
_SPEC = re.compile(r"^\s*(~=|===|==|!=|<=|>=|<|>)\s*(\S+)\s*$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirement(req: str) -> tuple[str, list[tuple[str, str]]] | None:
    """``"flashinfer_python[cu13]==0.6.18"`` -> ``("flashinfer-python", [("==", "0.6.18")])``.

    Returns None for requirements this parser skips: ones with an environment
    marker (``; extra == ...``, ``; sys_platform ...``) and direct URLs.
    """
    m = _REQ.match(req)
    if not m or m[3]:
        return None
    specs = [s for s in (_SPEC.match(p) for p in m[2].split(",") if p.strip()) if s]
    return _norm(m[1]), [(s[1], s[2]) for s in specs]


# --- checks ----------------------------------------------------------------------

def _pin_violations(pkg: str, pkg_ver: str) -> list[Finding]:
    out = []
    for req in _dist_requires(pkg):
        parsed = parse_requirement(req)
        if not parsed:
            continue
        dep, specs = parsed
        have = _dist_version(dep)
        if have is None:
            continue
        bad = [f"{op}{t}" for op, t in specs if _satisfies(have, op, t) is False]
        if bad:
            out.append(Finding("error", dep, (
                f"{pkg} {pkg_ver} requires {dep}{','.join(op + t for op, t in specs)}, "
                f"found {have} (violates {','.join(bad)}). Reinstall {pkg} or recreate the env."
            )))
    return out


def _cuda_findings(sglang: str | None, vllm: str | None) -> list[Finding]:
    gpu_runtime = sglang or vllm
    try:
        import torch  # opt-in only: check_cuda=True
    except Exception as e:   # noqa: BLE001 -- a broken torch import is itself the finding
        return [Finding("warn" if gpu_runtime else "info", "torch", f"CUDA check skipped: cannot import torch ({e})")]
    cuda = getattr(torch.version, "cuda", None)
    if not torch.cuda.is_available():
        msg = (f"torch.cuda.is_available() is False (torch built for CUDA {cuda or 'none (CPU build)'}); "
               "CUDA 13 builds need an NVIDIA driver >= 580, see nvidia-smi.")
        return [Finding("warn" if gpu_runtime else "info", "torch", msg)]
    out = [Finding("info", "torch", f"CUDA {cuda}, device: {torch.cuda.get_device_name(0)}")]
    if _dist_version("torchaudio"):
        try:
            import torchaudio  # noqa: F401 -- raises on a torch/torchaudio CUDA mismatch
        except Exception as e:   # noqa: BLE001
            if "CUDA" in str(e):
                out.append(Finding("error", "torchaudio", (
                    f"torchaudio does not load against this torch ({e}). transformers imports "
                    "torchaudio whenever it is installed, so every hf:/vllm: model load fails. "
                    "Nothing in AuditKit needs audio: pip uninstall -y torchaudio (or install "
                    "the torchaudio build matching torch's CUDA)."
                )))
    if sglang and _vkey(sglang) >= _vkey("0.5.11") and cuda and int(cuda.split(".")[0]) < 13:
        out.append(Finding("warn", "torch", (
            f"sglang {sglang} is built for CUDA 13 (torch>=2.11 cu13 wheels), but torch reports CUDA {cuda}; "
            "its flashinfer/sgl-kernel wheels may fail to load. Reinstall torch from the cu13 index."
        )))
    out.extend(_deepgemm_findings(torch, sglang))
    out.extend(_flashinfer_findings(torch, vllm))
    return out


def _flashinfer_findings(torch, vllm: str | None) -> list[Finding]:
    """Rule (o): vLLM on an SM 12.x GPU whose CUDA toolkit FlashInfer can't build with. (SGLang's
    side of the same problem is in sglang_launch_advice.)"""
    gpu = _gpu(torch)
    if not vllm or gpu is None:
        return []
    why = flashinfer_jit_blocked(gpu[0][0])
    if not why or os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1").lower() in ("0", "false"):
        return []
    return [Finding("warn", "vllm", (
        f"vllm {vllm} on this GPU (compute capability {gpu[0][0]}.{gpu[0][1]}) samples with FlashInfer by "
        f"default, and {why}: every model fails to load with \"FlashInfer requires GPUs with sm75 or "
        "higher\" (measured: Colab RTX PRO 6000, CUDA 12.8). AuditKit's vllm: backend switches to vLLM's "
        "own sampler by itself; for vllm serve, set VLLM_USE_FLASHINFER_SAMPLER=0 in its env."
    ))]


_DEEPGEMM_NVCC = "12.9"      # DeepGEMM's JIT refuses older nvcc ("NVCC version must be at least 12.9")
_FLASHINFER_SM12_NVCC = "12.9"   # FlashInfer 0.6.18 compilation_context: "SM 12.x requires CUDA >= 12.9"


def _nvcc_version(nvcc: str | None = None) -> str | None:
    """``nvcc --version`` of *nvcc*, or of the one a JIT build would pick up ($CUDA_HOME/bin, then PATH)."""
    import shutil
    import subprocess
    if nvcc is None:
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        nvcc = os.path.join(home, "bin", "nvcc") if home and os.path.isfile(os.path.join(home, "bin", "nvcc")) \
            else shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        text = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
    except Exception:   # noqa: BLE001
        return None
    m = re.search(r"release (\d+\.\d+)", text)
    return m.group(1) if m else None


def _gpu(torch) -> tuple[tuple[int, int], bool] | None:
    """(compute capability, native bf16) of GPU 0, or None when torch can't say."""
    try:
        cc = tuple(torch.cuda.get_device_capability(0))
    except Exception:   # noqa: BLE001 -- no GPU, or a partial torch
        return None
    try:
        bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        bf16 = cc[0] >= 8
    except Exception:   # noqa: BLE001
        bf16 = cc[0] >= 8
    return cc, bf16


def _deepgemm_installed() -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec("deep_gemm") is not None
    except (ImportError, ValueError):
        return False


def _deepgemm_off() -> bool:
    return os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1").lower() in ("0", "false")


def flashinfer_jit_blocked(major: int) -> str | None:
    """Why FlashInfer can't JIT-compile on this GPU, or None when it can.

    FlashInfer (SGLang's default attention and sampling, vLLM's default top-k/top-p sampler)
    compiles for SM 12.x (RTX PRO 6000, RTX 50-series) only with CUDA >= 12.9. With an older
    toolkit, or none, it swallows that error, targets no architecture, and every model load fails
    with the misleading "FlashInfer requires GPUs with sm75 or higher" (measured on Colab's RTX PRO
    6000 with CUDA 12.8, through SGLang and through vLLM). Any other GPU, or a toolkit >= 12.9: None.
    """
    if major != 12:
        return None
    nvcc = _nvcc_version()
    if nvcc and _vkey(nvcc) >= _vkey(_FLASHINFER_SM12_NVCC):
        return None
    return (f"FlashInfer builds for SM 12.x only with CUDA >= {_FLASHINFER_SM12_NVCC}, and the nvcc here is "
            f"{nvcc or 'not found'}")


def vllm_launch_advice(torch=None) -> dict:
    """What vLLM (``vllm:`` or ``vllm serve``) needs on THIS machine: ``{"env": {...}, "args": [...],
    "notes": [...]}``, like :func:`sglang_launch_advice`. Imports torch (pass it in to reuse one).

    - SM 12.x GPUs with an nvcc older than 12.9 (or none): ``VLLM_USE_FLASHINFER_SAMPLER=0``.
      vLLM uses FlashInfer's top-k/top-p sampler by default on supported GPUs, and SM 12.x counts
      as one, so every load fails while FlashInfer can't compile. vLLM's own sampler (PyTorch,
      Triton for larger batches) needs no CUDA compiler. Attention is unaffected: vLLM's default on
      SM 12.x is FlashAttention, shipped prebuilt.
    - GPUs without bf16 need nothing: vLLM's ``dtype="auto"`` already casts bf16 to fp16 there.
    - A variable the user set is never overridden.
    """
    if torch is None:
        import torch  # noqa: F811
    env: dict[str, str] = {}
    notes: list[str] = []
    gpu = _gpu(torch) if torch.cuda.is_available() else None
    if gpu is None:
        return {"env": env, "args": [], "notes": ["no CUDA GPU visible to torch here"]}
    (major, minor), _ = gpu
    why = flashinfer_jit_blocked(major)
    if why and "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        notes.append(f"vLLM's own sampler instead of FlashInfer's: {why}")
    return {"env": env, "args": [], "notes": notes, "compute_capability": f"{major}.{minor}"}


def sglang_launch_advice(torch=None) -> dict:
    """What an SGLang server needs on THIS machine, whatever its GPU: ``{"env": {...}, "args": [...],
    "notes": [...]}``. Imports torch (pass it in to reuse one). Uses the machine's own CUDA toolkit.

    - GPUs without native bf16 (T4, V100): ``--dtype float16`` (SGLang's ``auto`` keeps a
      checkpoint's bf16 on any GPU).
    - SM 12.x GPUs (RTX PRO 6000, RTX 50-series) with an nvcc older than 12.9 (or none):
      ``--attention-backend triton --sampling-backend pytorch``. FlashInfer can't compile for them
      with that toolkit, and neither replacement needs a CUDA compiler.
    - Hopper/Blackwell (compute capability >= 9) with DeepGEMM installed:
      ``SGLANG_ENABLE_JIT_DEEPGEMM=0``. SGLang enables DeepGEMM there and JIT-compiles it at startup,
      which needs nvcc >= 12.9 and fails every model load without it. DeepGEMM serves FP8 GEMMs only,
      so bf16/fp16 models lose nothing; an FP8 run should drop this switch.
    - The SGLang source bugs rules (i)/(j) fix: a note to run ``compat.py --patch-sglang``.
    """
    if torch is None:
        import torch  # noqa: F811
    env: dict[str, str] = {}
    args: list[str] = []
    notes: list[str] = []
    gpu = _gpu(torch) if torch.cuda.is_available() else None
    if gpu is None:
        return {"env": env, "args": args, "notes": ["no CUDA GPU visible to torch here"]}
    (major, minor), bf16 = gpu
    if not bf16:
        args += ["--dtype", "float16"]
        notes.append(f"compute capability {major}.{minor} has no native bf16: serving in float16")
    why = flashinfer_jit_blocked(major)
    if why:
        # FlashInfer is SGLang's default attention and sampling; Triton attention and PyTorch
        # sampling need no CUDA compiler
        args += ["--attention-backend", "triton", "--sampling-backend", "pytorch"]
        notes.append(f"Triton attention + PyTorch sampling: {why}")
    if major >= 9 and _deepgemm_installed() and not _deepgemm_off():
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"
        notes.append("DeepGEMM off: its startup JIT needs nvcc >= 12.9, and it serves FP8 GEMMs only "
                     "(drop SGLANG_ENABLE_JIT_DEEPGEMM=0 for an FP8 run)")
    bugs = [name for name, *_ in _SGLANG_FIXES if _sglang_fix_state(name) == "bug"]
    if bugs:
        notes.append(f"unpatched SGLang source ({', '.join(bugs)}): run compat.py --patch-sglang first")
    return {"env": env, "args": args, "notes": notes, "compute_capability": f"{major}.{minor}"}


def _deepgemm_findings(torch, sglang: str | None) -> list[Finding]:
    # (k) Hopper/Blackwell + DeepGEMM on + nvcc < 12.9 (or none) -> every model fails to load
    if not sglang or _deepgemm_off():
        return []
    gpu = _gpu(torch)
    if gpu is None or gpu[0][0] < 9 or not _deepgemm_installed():
        return []
    nvcc = _nvcc_version()
    if nvcc and _vkey(nvcc) >= _vkey(_DEEPGEMM_NVCC):
        return []
    major, minor = gpu[0]
    return [Finding("warn", "sglang", (
        f"sglang {sglang} on this GPU (compute capability {major}.{minor}) enables DeepGEMM, whose "
        f"startup JIT needs nvcc >= {_DEEPGEMM_NVCC}; the nvcc here is {nvcc or 'not found'}, so every "
        "model fails to load (\"NVCC version must be at least 12.9\"; measured on Colab's RTX PRO 6000). "
        "Set SGLANG_ENABLE_JIT_DEEPGEMM=0 in the server's env (DeepGEMM serves FP8 GEMMs only). "
        "`compat.py --sglang-launch MODEL` prints the full command for this machine."
    ))]


def _launch_command(model: str, advice: dict) -> str:
    import shlex
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in advice["env"].items())
    cmd = f"python -m sglang.launch_server --model-path {shlex.quote(model)} " + " ".join(advice["args"])
    return (env + " " if env else "") + cmd.strip()


def check_compat(check_cuda: bool = False) -> CompatReport:
    """Check the current environment for backend version clashes.

    Stdlib only; torch is imported only when ``check_cuda=True``. Returns a
    :class:`CompatReport` -- ``print()`` it, test ``.ok``, or call
    ``.raise_on_error()``.
    """
    system, glibc, py = _system(), _libc_ver(), _py_version()
    v = {name: _dist_version(name) for name in _REPORTED}
    report = CompatReport(versions={
        "python": ".".join(map(str, py)),
        "platform": f"{system} {platform.machine()}".strip(),
        "glibc": glibc,
        **v,
    })
    f = report.findings
    sglang, vllm, torch, trf = v["sglang"], v["vllm"], v["torch"], v["transformers"]

    for pkg in ("sglang", "vllm"):                                          # (a)
        if v[pkg]:
            f.extend(_pin_violations(pkg, v[pkg]))

    if sglang and vllm:                                                     # (b)
        f.append(Finding("error", "sglang", (
            f"sglang {sglang} and vllm {vllm} are installed together. They never co-resolve "
            "(llguidance: e.g. vllm 0.16 needs >=1.3,<1.4, sglang needs <0.8 or >=1.7.6; "
            "xgrammar/torch pins clash too). Give SGLang its own venv and reach it over HTTP "
            "with the api: backend."
        )))

    if sglang:                                                              # (c)
        if _vkey(sglang) < _vkey("0.5.10"):
            f.append(Finding("warn", "sglang", (
                f"sglang {sglang} is older than 0.5.10 -- usually a silent downgrade from resolving "
                "sglang next to auditkit[transformers] or vllm. Install "
                "'sglang>=0.5.18,<0.6' into a separate env."
            )))
        if not torch:
            f.append(Finding("warn", "sglang", (
                f"sglang {sglang} is installed without torch: frontend only, no server runtime "
                "(a resolver picking sglang 0.5.2 next to vllm ends up here)."
            )))

    lo, hi = _TRANSFORMERS_SUPPORTED
    if trf and not (_vkey(lo) <= _vkey(trf) < _vkey(hi)):                   # (d)
        f.append(Finding("warn", "transformers", (
            f"transformers {trf} is outside {lo}-{hi}; AuditKit's hf: backend is validated on "
            f"transformers>={lo},<{hi} (the [transformers] extra). Expected inside an SGLang env; "
            "run hf: models elsewhere."
        )))
        if sglang and _vkey(trf) < _vkey(_COMPASS_MIN_TRANSFORMERS):
            f.append(Finding("warn", "sglang", (
                f"sglang {sglang} runs on transformers {trf}, which cannot load North Micro Vision "
                f"(cohere_compass needs transformers>={_COMPASS_MIN_TRANSFORMERS}). Serve North with "
                "vllm: / vllm serve or evaluate it with hf:. With transformers>=5.15 in this env it "
                "also needs --patch-sglang (the embedding_rowwise and rope_parameters_none fixes). Measured on sglang 0.5.20: "
                "Tiny Aya and Aya Expanse load natively; Aya Vision works through SGLang's Transformers "
                "backend with --patch-sglang."
            )))

    if sglang and _dec_lock_state() == "bug":                                # (i)
        f.append(Finding("warn", "sglang", (
            f"sglang {sglang}: the scheduler calls UnifiedRadixCache.dec_lock_ref() without its "
            "required params when the radix cache is disabled, as it is for multimodal models on "
            "SGLang's Transformers backend. Aya Vision loads, then crashes on its first request "
            "(TypeError: ... missing 1 required positional argument: 'params'). Fix in this env: "
            "python compat.py --patch-sglang (defaults params to None; that path never reads it). "
            "Upstream bug, also on sglang main as of 2026-09-30."
        )))
    if sglang and trf and _vkey(trf) >= _vkey(_EMBEDDING_ROWWISE_TRANSFORMERS) \
            and _sglang_fix_state("embedding_rowwise") == "bug":                # (j)
        f.append(Finding("warn", "sglang", (
            f"sglang {sglang} with transformers {trf}: transformers>={_EMBEDDING_ROWWISE_TRANSFORMERS} adds "
            "'embedding_rowwise' to the TP plan of every tied-embedding model, and SGLang's Transformers "
            "backend rejects the name (ValueError: Unsupported TP style 'embedding_rowwise'), so North "
            "Micro Vision fails to load even on transformers>=5.15. The entry is inert there (the plan is "
            "read for nn.Linear only). Fix in this env: python compat.py --patch-sglang."
        )))

    if sglang:                                                              # (e)
        if system != "Linux":
            f.append(Finding("error", "sglang", (
                f"sglang runtime wheels are Linux-only (manylinux_2_34, glibc>=2.34); this is {system}. "
                "Run the SGLang server on a Linux GPU host and point api: at it."
            )))
        elif _vkey(sglang) >= _vkey("0.5.11") and (glibc is None or _vkey(glibc) < _vkey("2.34")):
            f.append(Finding("error", "sglang", (
                f"sglang {sglang} ships manylinux_2_34 wheels only; this system's glibc is "
                f"{glibc or 'unknown/non-glibc'}. Use a newer distro (e.g. Ubuntu 22.04+) or the "
                "official SGLang Docker image."
            )))
        if not ((3, 10) <= py[:2] <= (3, 13)):
            f.append(Finding("error", "python", (
                f"sglang wheels exist for CPython 3.10-3.13; this is Python {report.versions['python']}."
            )))
        elif py[:2] == (3, 13):
            f.append(Finding("warn", "python", (
                "Python 3.13: current sglang pins outlines==0.1.11 -> outlines_core 0.1.26, which has "
                "wheels up to cp312 only, so pip tries a Rust source build that fails. Build the SGLang "
                "venv on 3.12: uv venv --python 3.12 --seed <env>."
            )))

    if vllm:                                                                # (f)
        if _vkey(vllm) < _vkey(_VLLM_MIN):
            f.append(Finding("warn", "vllm", (
                f"vllm {vllm} is older than {_VLLM_MIN}, the first release AuditKit's vllm: backend "
                f"is validated on (auditkit[vllm] pins >={_VLLM_MIN})."
            )))

    if check_cuda:                                                          # (g)
        f.extend(_cuda_findings(sglang, vllm))

    acc = v["accelerate"]
    if acc and trf and _vkey(trf) >= _vkey("5") and _vkey(acc) < _vkey("1.0"):   # (h)
        f.append(Finding("warn", "accelerate", (
            f"accelerate {acc} with transformers {trf}: transformers 5 treats accelerate < 1.0 as "
            "missing and refuses device_map (lm-eval's hf backend fails). pip install -U 'accelerate>=1.0'."
        )))

    if not sglang and not vllm:
        f.append(Finding("info", "sglang", (
            "neither sglang nor vllm is installed here. Serving SGLang over HTTP and evaluating it "
            "with the api: backend needs only auditkit[requests] in this env."
        )))
    return report


if __name__ == "__main__":
    if "--patch-sglang" in sys.argv:
        print("sglang patches:", patch_sglang())
    if "--vllm-launch" in sys.argv:
        print("vllm launch advice:", __import__("json").dumps(vllm_launch_advice()))
        sys.exit(0)
    if "--sglang-launch" in sys.argv:
        _idx = sys.argv.index("--sglang-launch")
        _model = sys.argv[_idx + 1] if len(sys.argv) > _idx + 1 else "<model>"
        _advice = sglang_launch_advice()
        print("sglang launch for this machine:", _launch_command(_model, _advice))
        print("sglang launch advice:", __import__("json").dumps(_advice))
        sys.exit(0)
    _report = check_compat(check_cuda="--cuda" in sys.argv)
    print(_report)
    sys.exit(0 if _report.ok else 1)
