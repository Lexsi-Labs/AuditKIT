"""auditkit.compat: environment clash detection, simulated via its metadata seams."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from auditkit import compat

SGLANG_0520_REQS = [
    "torch==2.13.0",
    "transformers==5.12.1",
    "flashinfer_python[cu13]==0.6.18",
    "xgrammar==0.2.1",
    "llguidance<2.0.0,>=1.7.6",
    "openai==2.6.1",
    "protobuf!=6.30.*,>=5.29",
    'tomli; python_version < "3.11"',
    'checkpoint-engine==0.1.2; extra == "checkpoint-engine"',
]
SGLANG_0520_ENV = {
    "sglang": "0.5.20", "torch": "2.13.0+cu130", "transformers": "5.12.1",
    "flashinfer-python": "0.6.18", "xgrammar": "0.2.1", "llguidance": "1.8.0",
    "openai": "2.6.1", "protobuf": "6.30.2",
}
VLLM_030_REQS = ["torch==2.11.0", "transformers>=5.15", "numba==0.61.2", "compressed-tensors==0.13.0"]


def run(monkeypatch, versions, requires=None, system="Linux", glibc="2.35", py=(3, 11, 9), **kw):
    versions = {compat._norm(k): v for k, v in {"auditkit": "1.0.0", **versions}.items()}
    monkeypatch.setattr(compat, "_dist_version", lambda n: versions.get(compat._norm(n)))
    monkeypatch.setattr(compat, "_dist_requires", lambda n: (requires or {}).get(n, []))
    monkeypatch.setattr(compat, "_system", lambda: system)
    monkeypatch.setattr(compat, "_libc_ver", lambda: glibc)
    monkeypatch.setattr(compat, "_py_version", lambda: py)
    return compat.check_compat(**kw)


def levels(report, level):
    return [f for f in report.findings if f.level == level]


def test_clean_auditkit_vllm_env(monkeypatch):
    r = run(monkeypatch, {"torch": "2.11.0+cu128", "transformers": "5.17.0", "vllm": "0.30.0",
                          "numba": "0.61.2", "compressed-tensors": "0.13.0"},
            {"vllm": VLLM_030_REQS})
    assert r.ok and not levels(r, "warn") and not levels(r, "error")
    assert r.versions["vllm"] == "0.30.0" and r.versions["glibc"] == "2.35"


def test_clean_sglang_env_only_warns_about_transformers5(monkeypatch):
    r = run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS})
    assert r.ok
    # the transformers range, and (S2) that this pin cannot load North Micro Vision
    assert [f.package for f in levels(r, "warn")] == ["transformers", "sglang"]
    assert "North Micro Vision" in levels(r, "warn")[1].message


def test_s2_sglang_on_a_recent_transformers_does_not_name_north(monkeypatch):
    r = run(monkeypatch, {**SGLANG_0520_ENV, "transformers": "5.17.0"})
    assert not any("North" in f.message for f in r.findings)


def test_s2_transformers_below_5_15_without_sglang_does_not_name_north(monkeypatch):
    r = run(monkeypatch, {"transformers": "5.12.1", "torch": "2.11.0"})
    assert not any("North" in f.message for f in r.findings)


def test_sglang_and_vllm_together(monkeypatch):
    r = run(monkeypatch, {**SGLANG_0520_ENV, "vllm": "0.16.0"}, {"sglang": SGLANG_0520_REQS})
    assert not r.ok
    assert any("never co-resolve" in f.message for f in levels(r, "error"))
    with pytest.raises(compat.CompatError, match="never co-resolve"):
        r.raise_on_error()


def test_sglang_059_downgrade(monkeypatch):
    r = run(monkeypatch, {"sglang": "0.5.9", "torch": "2.9.1", "transformers": "4.57.1"})
    assert r.ok
    assert any("silent downgrade" in f.message for f in levels(r, "warn"))


def test_pin_violation(monkeypatch):
    r = run(monkeypatch, {**SGLANG_0520_ENV, "torch": "2.10.0"}, {"sglang": SGLANG_0520_REQS})
    (err,) = levels(r, "error")
    assert err.package == "torch" and "torch==2.13.0" in err.message and "found 2.10.0" in err.message


def test_macos_with_sglang(monkeypatch):
    r = run(monkeypatch, {"sglang": "0.5.10", "torch": "2.9.1"}, system="Darwin", glibc=None)
    assert any("Linux-only" in f.message for f in levels(r, "error"))


def test_old_glibc(monkeypatch):
    r = run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}, glibc="2.31")
    assert any("manylinux_2_34" in f.message for f in levels(r, "error"))


def test_python_out_of_range(monkeypatch):
    r = run(monkeypatch, SGLANG_0520_ENV, py=(3, 14, 0))
    assert [f.package for f in levels(r, "error")] == ["python"]


def test_sglang_without_torch_is_frontend_only(monkeypatch):
    r = run(monkeypatch, {"sglang": "0.5.2"})
    assert any("frontend only" in f.message for f in levels(r, "warn"))


def test_transformers_and_vllm_out_of_range(monkeypatch):
    r = run(monkeypatch, {"transformers": "4.57.1", "vllm": "0.21.0"})
    assert sorted(f.package for f in levels(r, "warn")) == ["transformers", "vllm"]


def test_nothing_installed_is_info_only(monkeypatch):
    r = run(monkeypatch, {}, system="Darwin", glibc=None)
    assert r.ok and r.findings and {f.level for f in r.findings} == {"info"}
    assert "OK" in str(r) and "python" in str(r)


@pytest.mark.parametrize("a, b", [
    ("2.13.0+cu130", "2.13.0"), ("2.13", "2.13.0"), ("0.6.18.post1", "0.6.18.post1"),
])
def test_version_equal(a, b):
    assert compat._vkey(a) == compat._vkey(b)


@pytest.mark.parametrize("lo, hi", [
    ("1.0.dev1", "1.0a1"), ("1.6.0rc5", "1.6.0"), ("0.6.18", "0.6.18.post1"),
    ("0.5.9", "0.5.10"), ("4.57.1", "5"), ("5.0.0rc1", "5.0.0"),
])
def test_version_order(lo, hi):
    assert compat._vkey(lo) < compat._vkey(hi)


def test_parse_requirement():
    assert compat.parse_requirement("flashinfer_python[cu13]==0.6.18") == ("flashinfer-python", [("==", "0.6.18")])
    assert compat.parse_requirement("llguidance<2.0.0,>=1.7.6") == ("llguidance", [("<", "2.0.0"), (">=", "1.7.6")])
    assert compat.parse_requirement('av==16.1.0; sys_platform == "linux"') is None
    assert compat.parse_requirement("pkg @ https://example.com/pkg.whl") is None


def test_import_pulls_no_heavy_modules():
    code = "import sys, auditkit.compat; print(sorted({'torch', 'sglang', 'vllm'} & set(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(Path(compat.__file__).parents[1])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]"


def test_cuda_check_flags_cuda12_torch_under_sglang(monkeypatch):
    from types import SimpleNamespace
    fake = SimpleNamespace(version=SimpleNamespace(cuda="12.8"),
                           cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "Tesla T4"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    r = run(monkeypatch, SGLANG_0520_ENV, check_cuda=True)
    assert any(f.level == "info" and "Tesla T4" in f.message for f in r.findings)
    assert any(f.package == "torch" and "CUDA 13" in f.message for f in levels(r, "warn"))


# -- (i): the sglang dec_lock_ref bug (Aya Vision crashes on its first request) --------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "sglang_0520"
SGLANG_EXCERPT = FIXTURES / "unified_radix_cache_excerpt.py"
TF_BACKEND_EXCERPT = FIXTURES / "transformers_backend_excerpt.py"
COMMANDR_EXCERPT = FIXTURES / "commandr_excerpt.py"


def fake_sglang(tmp_path, source=None, tf_source=None):
    """A package directory laid out like an installed sglang, holding the real 0.5.20 code."""
    root = tmp_path / "site" / "sglang"
    (root / "srt" / "mem_cache").mkdir(parents=True)
    (root / "srt" / "models").mkdir(parents=True)
    (root / "__init__.py").write_text("raise ImportError('compat must never import sglang')\n")
    (root / "srt" / "mem_cache" / "unified_radix_cache.py").write_text(source or SGLANG_EXCERPT.read_text())
    (root / "srt" / "models" / "transformers.py").write_text(tf_source or TF_BACKEND_EXCERPT.read_text())
    (root / "srt" / "models" / "commandr.py").write_text(COMMANDR_EXCERPT.read_text())
    return root


def dec_lock_findings(report):
    return [f for f in report.findings if "calls UnifiedRadixCache.dec_lock_ref() without" in f.message]


def test_i_the_unpatched_sglang_is_named_with_its_fix(monkeypatch, tmp_path):
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    [f] = dec_lock_findings(run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}))
    assert "Aya Vision" in f.message and "--patch-sglang" in f.message


def test_i_the_patch_is_applied_once_and_the_warning_goes(monkeypatch, tmp_path):
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert compat.patch_sglang().startswith("dec_lock_ref: patched ")
    assert compat.patch_sglang() == ("dec_lock_ref: already patched; embedding_rowwise: already patched; "
                                     "cohere2_sliding_window: already patched; "
                                     "transformers_sliding_window: already patched; "
                                     "rope_parameters_none: already patched")
    src = (root / "srt" / "mem_cache" / "unified_radix_cache.py").read_text()
    assert src.count("params: DecLockRefParams = None,") == 1 and "params: DecLockRefParams,\n" not in src
    assert not dec_lock_findings(run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}))


def test_i_a_changed_sglang_is_left_alone(monkeypatch, tmp_path):
    # a future sglang that fixed or moved the method: no warning, and the patch touches nothing
    root = fake_sglang(tmp_path, "class UnifiedRadixCache:\n    def dec_lock_ref(self, node_id, params=None): ...\n")
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert not dec_lock_findings(run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}))
    assert compat.patch_sglang().startswith("dec_lock_ref: not applied")


def test_i_no_sglang_source_no_finding(monkeypatch):
    monkeypatch.setattr(compat, "_sglang_path", lambda: None)
    assert not dec_lock_findings(run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}))


def test_i_the_script_patches_a_real_install_without_importing_sglang(tmp_path):
    # as run in the SGLang venv: python compat.py --patch-sglang, sglang found by the import system
    root = fake_sglang(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(root.parent)}
    script = Path(compat.__file__)
    out = subprocess.run([sys.executable, str(script), "--patch-sglang"], capture_output=True, text=True, env=env)
    assert f"dec_lock_ref: patched {root}" in out.stdout, out.stdout + out.stderr
    assert f"embedding_rowwise: patched {root}" in out.stdout
    assert "params: DecLockRefParams = None," in (root / "srt" / "mem_cache" / "unified_radix_cache.py").read_text()


# -- (j): transformers >= 5.15 'embedding_rowwise' vs SGLang's Transformers backend (North) ---------------------

def embedding_findings(report):
    return [f for f in report.findings if "backend rejects the name" in f.message]


def test_j_named_only_when_the_env_has_a_transformers_that_emits_the_style(monkeypatch, tmp_path):
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert not embedding_findings(run(monkeypatch, SGLANG_0520_ENV, {"sglang": SGLANG_0520_REQS}))   # pinned 5.12.1
    [f] = embedding_findings(run(monkeypatch, {**SGLANG_0520_ENV, "transformers": "5.16.1"},
                                 {"sglang": SGLANG_0520_REQS}))
    assert "North" in f.message and "--patch-sglang" in f.message


def test_j_the_patched_style_table_maps_it_to_replicate_and_nothing_else(monkeypatch, tmp_path):
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    compat.patch_sglang()
    ns = {"Style": str}
    exec((root / "srt" / "models" / "transformers.py").read_text(), ns)    # sglang's real function, patched
    assert ns["_normalize_tp_style"]("embedding_rowwise") == "replicate"
    assert ns["_normalize_tp_style"]("colwise") == "colwise"
    with pytest.raises(ValueError, match="Unsupported TP style"):
        ns["_normalize_tp_style"]("something_else")                        # only the inert style is added
    assert not embedding_findings(run(monkeypatch, {**SGLANG_0520_ENV, "transformers": "5.16.1"},
                                      {"sglang": SGLANG_0520_REQS}))


# -- (k): Blackwell + DeepGEMM + an nvcc older than 12.9 (every SGLang model fails to load) ---------------------

def blackwell(monkeypatch, nvcc, deepgemm=True, capability=(12, 0)):
    from types import SimpleNamespace
    import importlib.util
    fake = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                           cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda i: capability,
                                                get_device_name=lambda i: "NVIDIA RTX PRO 6000 Blackwell"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: nvcc)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if deepgemm else None) if name == "deep_gemm" else real(name, *a))


def deepgemm_findings(report):
    return [f for f in report.findings if "SGLANG_ENABLE_JIT_DEEPGEMM" in f.message]


@pytest.mark.parametrize("nvcc,capability", [("12.8", (12, 0)), (None, (12, 0)), ("12.8", (10, 0)), ("12.4", (9, 0))])
def test_k_old_or_missing_nvcc_on_hopper_or_blackwell_names_the_switch(monkeypatch, nvcc, capability):
    monkeypatch.delenv("SGLANG_ENABLE_JIT_DEEPGEMM", raising=False)
    blackwell(monkeypatch, nvcc, capability=capability)
    [f] = deepgemm_findings(run(monkeypatch, SGLANG_0520_ENV, check_cuda=True))
    assert f.level == "warn" and (nvcc or "not found") in f.message


@pytest.mark.parametrize("case", ["nvcc 12.9", "switched off", "no deep_gemm", "ampere"])
def test_k_no_finding_when_it_would_not_fail(monkeypatch, case):
    monkeypatch.delenv("SGLANG_ENABLE_JIT_DEEPGEMM", raising=False)
    blackwell(monkeypatch, "12.9" if case == "nvcc 12.9" else "12.8",
              deepgemm=case != "no deep_gemm", capability=(8, 0) if case == "ampere" else (12, 0))
    if case == "switched off":
        monkeypatch.setenv("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    assert not deepgemm_findings(run(monkeypatch, SGLANG_0520_ENV, check_cuda=True))


def test_k_the_old_fake_torch_without_capability_still_works(monkeypatch):
    # partial torch objects (the T4 test above) must not crash the check
    from types import SimpleNamespace
    fake = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                           cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "x"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert not deepgemm_findings(run(monkeypatch, SGLANG_0520_ENV, check_cuda=True))


# -- sglang_launch_advice: the env and flags an SGLang server needs, on any GPU ---------------------------------

def gpu_torch(capability, bf16):
    from types import SimpleNamespace
    return SimpleNamespace(version=SimpleNamespace(cuda="13.0"), cuda=SimpleNamespace(
        is_available=lambda: True, get_device_capability=lambda i: capability,
        is_bf16_supported=lambda including_emulation=True: bf16))


def advise(monkeypatch, capability, bf16, deepgemm=True, nvcc="12.8"):
    import importlib.util
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: nvcc)
    monkeypatch.setattr(compat, "_sglang_path", lambda: None)
    monkeypatch.delenv("SGLANG_ENABLE_JIT_DEEPGEMM", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if deepgemm else None) if name == "deep_gemm" else real(name, *a))
    return compat.sglang_launch_advice(gpu_torch(capability, bf16))


@pytest.mark.parametrize("gpu,capability,bf16,dtype_fp16,deepgemm_off", [
    ("T4", (7, 5), False, True, False),
    ("V100", (7, 0), False, True, False),
    ("A100", (8, 0), True, False, False),
    ("L4", (8, 9), True, False, False),
    ("H100", (9, 0), True, False, True),
    ("B200", (10, 0), True, False, True),
    ("RTX PRO 6000 (Colab)", (12, 0), True, False, True),
])
def test_advice_per_gpu(monkeypatch, gpu, capability, bf16, dtype_fp16, deepgemm_off):
    advice = advise(monkeypatch, capability, bf16)
    assert (advice["args"] == ["--dtype", "float16"]) == dtype_fp16, gpu
    assert (advice["env"].get("SGLANG_ENABLE_JIT_DEEPGEMM") == "0") == deepgemm_off, gpu
    assert "CUDA_HOME" not in advice["env"], gpu               # the machine's own CUDA toolkit, always


def test_advice_without_deep_gemm_never_switches_it_off(monkeypatch):
    advice = advise(monkeypatch, (12, 0), True, deepgemm=False)
    assert "SGLANG_ENABLE_JIT_DEEPGEMM" not in advice["env"]


def test_advice_names_the_unpatched_sglang(monkeypatch, tmp_path):
    advise(monkeypatch, (8, 0), True)
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    advice = compat.sglang_launch_advice(gpu_torch((8, 0), True))
    assert any("--patch-sglang" in n for n in advice["notes"])


def test_the_launch_command_is_ready_to_paste():
    cmd = compat._launch_command("CohereLabs/tiny-aya-global", {
        "env": {"SGLANG_ENABLE_JIT_DEEPGEMM": "0"}, "args": ["--dtype", "float16"]})
    assert cmd == ("SGLANG_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server "
                   "--model-path CohereLabs/tiny-aya-global --dtype float16")


def test_advice_on_a_machine_without_a_gpu():
    from types import SimpleNamespace
    cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert compat.sglang_launch_advice(cpu) == {"env": {}, "args": [], "notes": ["no CUDA GPU visible to torch here"]}


TRITON = ["--attention-backend", "triton", "--sampling-backend", "pytorch"]


@pytest.mark.parametrize("capability,nvcc,triton", [
    ((12, 0), "12.8", True),     # Colab RTX PRO 6000: "FlashInfer requires GPUs with sm75 or higher"
    ((12, 0), None, True),
    ((12, 0), "12.9", False),
    ((12, 1), "13.0", False),
    ((10, 0), "12.8", False),    # FlashInfer's 12.9 rule is for SM 12.x only
    ((9, 0), "12.4", False),
    ((8, 0), "12.2", False),     # the A100 setup that worked stays untouched
])
def test_sm12_on_an_old_toolkit_starts_on_triton(monkeypatch, capability, nvcc, triton):
    advice = advise(monkeypatch, capability, True, nvcc=nvcc)
    assert (advice["args"][-4:] == TRITON) == triton
    assert "CUDA_HOME" not in advice["env"]



# -- (l): Cohere2ForCausalLM reports its sliding window (Triton backend, kv_indptr=None) --------------------------

def test_l_the_patched_class_reports_the_layers_window(monkeypatch, tmp_path):
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert compat._sglang_fix_state("cohere2_sliding_window") == "bug"
    compat.patch_sglang()
    assert compat._sglang_fix_state("cohere2_sliding_window") == "patched"
    ns = {"CohereForCausalLM": type("CohereForCausalLM", (), {})}
    exec((root / "srt" / "models" / "commandr.py").read_text(), ns)        # sglang's real class, patched
    model = ns["Cohere2ForCausalLM"]()
    from types import SimpleNamespace
    model.config = SimpleNamespace(sliding_window=4096)                    # Tiny Aya's real value
    assert model.get_attention_sliding_window_size() == 4096
    model.config = SimpleNamespace()
    assert model.get_attention_sliding_window_size() is None               # no window: as before


def test_l_the_advice_names_it_until_patched(monkeypatch, tmp_path):
    advise(monkeypatch, (12, 0), True)
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert any("cohere2_sliding_window" in n for n in compat.sglang_launch_advice(gpu_torch((12, 0), True))["notes"])
    compat.patch_sglang()
    assert not any("unpatched" in n for n in compat.sglang_launch_advice(gpu_torch((12, 0), True))["notes"])



# -- (m): SGLang's Transformers backend reports a sliding window (Aya Vision, Triton decode) ----------------------

def test_m_the_transformers_backend_reports_the_window_only_for_sliding_models(monkeypatch, tmp_path):
    import ast
    from types import SimpleNamespace as NS
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    assert compat._sglang_fix_state("transformers_sliding_window") == "bug"
    compat.patch_sglang()
    src = (root / "srt" / "models" / "transformers.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransformersBase")
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "get_attention_sliding_window_size")
    ns = {}
    exec(compile(ast.Module([fn], []), "m", "exec"), ns)
    window = ns["get_attention_sliding_window_size"]
    aya_vision_text = NS(layer_types=["sliding_attention"] * 24 + ["full_attention"] * 8, sliding_window=4096)
    assert window(NS(text_config=aya_vision_text, config=NS())) == 4096        # Aya Vision 8B's real shape
    assert window(NS(text_config=NS(layer_types=["full_attention"], sliding_window=4096), config=NS())) is None
    assert window(NS(text_config=NS(sliding_window=4096), config=NS())) is None



# -- (n): North's per-layer rope_parameters has None for its no-RoPE layers ---------------------------------------

NORTH_ROPE = {"sliding_attention": {"mrope_interleaved": True, "mrope_section": [24, 20, 20],
                                    "rope_type": "default", "rope_theta": 50000},
              "full_attention": None}                                   # North Micro Vision's real config


def _torch_compile_check(root):
    import ast
    src = (root / "srt" / "models" / "transformers.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "can_enable_torch_compile")
    fn.args.args[0].annotation = None
    ns = {}
    exec(compile(ast.Module([fn], []), "n", "exec"), ns)
    return ns["can_enable_torch_compile"]


def test_n_sglangs_check_crashes_on_north_and_the_patch_fixes_it(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace as NS
    root = fake_sglang(tmp_path)
    monkeypatch.setattr(compat, "_sglang_path", lambda: str(root))
    north = NS(text_config=NS(rope_scaling=None, rope_parameters=NORTH_ROPE))
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        _torch_compile_check(root)(north)                                # the Colab failure, reproduced
    assert compat._sglang_fix_state("rope_parameters_none") == "bug"
    compat.patch_sglang()
    assert compat._sglang_fix_state("rope_parameters_none") == "patched"
    check = _torch_compile_check(root)
    assert check(north) is True
    dynamic = dict(NORTH_ROPE, sliding_attention={"rope_type": "dynamic", "rope_theta": 1})
    assert check(NS(rope_scaling=None, rope_parameters=dynamic)) is False    # still refuses dynamic RoPE
    assert check(NS(rope_scaling=None, rope_parameters={"rope_type": "default"})) is True



# -- (o): FlashInfer can't build on SM 12.x with CUDA < 12.9, whichever backend uses it ----------------------------

def vllm_advise(monkeypatch, capability, nvcc):
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: nvcc)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    return compat.vllm_launch_advice(gpu_torch(capability, True))


@pytest.mark.parametrize("gpu,capability,nvcc,blocked", [
    ("A100", (8, 0), "12.8", False),
    ("L4", (8, 9), None, False),
    ("H100", (9, 0), "12.8", False),
    ("B200", (10, 0), "12.8", False),
    ("RTX PRO 6000, Colab's CUDA 12.8", (12, 0), "12.8", True),     # the measured failure
    ("RTX 5090, no toolkit", (12, 0), None, True),
    ("RTX PRO 6000, CUDA 12.9", (12, 0), "12.9", False),
    ("RTX PRO 6000, CUDA 13.0", (12, 0), "13.0", False),
])
def test_o_vllm_samples_without_flashinfer_only_where_it_cant_build(monkeypatch, gpu, capability, nvcc, blocked):
    advice = vllm_advise(monkeypatch, capability, nvcc)
    assert (advice["env"] == {"VLLM_USE_FLASHINFER_SAMPLER": "0"}) == blocked, gpu
    assert advice["args"] == []
    # one check for every backend: SGLang moves to Triton exactly where vLLM drops FlashInfer's sampler
    assert (advise(monkeypatch, capability, True, nvcc=nvcc)["args"][-4:] == TRITON) == blocked, gpu


def test_o_a_setting_the_user_made_is_left_alone(monkeypatch):
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: "12.8")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    assert compat.vllm_launch_advice(gpu_torch((12, 0), True))["env"] == {}


def test_o_check_compat_warns_until_the_sampler_is_off(monkeypatch):
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: "12.8")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    [f] = compat._flashinfer_findings(gpu_torch((12, 0), True), "0.30.0")
    assert f.level == "warn" and "VLLM_USE_FLASHINFER_SAMPLER=0" in f.message and "sm75" in f.message
    assert not compat._flashinfer_findings(gpu_torch((12, 0), True), None)          # no vLLM here
    assert not compat._flashinfer_findings(gpu_torch((8, 0), True), "0.30.0")       # A100
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    assert not compat._flashinfer_findings(gpu_torch((12, 0), True), "0.30.0")


def test_o_the_vllm_backend_applies_it_before_the_engine_starts(monkeypatch):
    import sys
    from types import SimpleNamespace
    from auditkit.model.vllm_gen import VLLMModel
    monkeypatch.setitem(sys.modules, "torch", gpu_torch((12, 0), True))
    monkeypatch.setattr(compat, "_nvcc_version", lambda path=None: "12.8")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    seen = {}

    class LLM:                                   # records the env the engine starts with
        def __init__(self, model, **kw):
            seen["env"] = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=dict))
    m = VLLMModel("CohereLabs/aya-expanse-32b")
    m._ensure_llm()
    assert seen["env"] == "0"
    assert m.run_notes() == {"vllm_launch_env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"}}

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")      # the user's choice wins
    m = VLLMModel("CohereLabs/aya-expanse-32b")
    m._ensure_llm()
    assert seen["env"] == "1" and m.run_notes() == {}
