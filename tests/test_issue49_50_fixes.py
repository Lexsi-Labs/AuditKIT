"""Issues #49 (a bare callable in ``scorers=``) and #50 (the HF run fingerprint)."""
from __future__ import annotations

import os

import pytest

import auditkit as ak
from auditkit.metric import ExactMatch
from auditkit.model import CallableModel, Result_, Generated
from auditkit.model.hf_gen import HFGenModel
from auditkit.model.vllm_gen import VLLMModel

SAMPLES = [ak.Sample(input="2+2?", target="4", id="a")]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


# ---- #49 -----------------------------------------------------------------

def test_bare_callable_scorer_is_refused_with_the_fix_in_the_message():
    with pytest.raises(TypeError, match=r"bare callable.*direction.*@ak\.scorer"):
        ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=[lambda s, o: 1.0], cache=False)


def test_a_metric_class_is_asked_for_as_an_instance():
    with pytest.raises(TypeError, match=r"pass an instance: ExactMatch\(\)"):
        ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=ExactMatch, cache=False)


def test_declared_scorers_still_work():
    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def one(sample, output):
        return 1.0

    def bare(sample, output):
        return 0.0

    for scorers in (one, [one], ak.scorer(bare, direction=ak.Direction.MINIMIZE)):
        res = ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=scorers, cache=False)
        assert "one: " in res.summary() or "bare: " in res.summary()


# ---- #50 -----------------------------------------------------------------

@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"a" * 10)
    return d


def test_load_and_call_settings_change_the_identity(ckpt):
    base = HFGenModel(model=str(ckpt)).identity()
    assert HFGenModel(model=str(ckpt), batch_size=8).identity() != base
    assert HFGenModel(model=str(ckpt), dtype="float16").identity() != base
    assert HFGenModel(model=str(ckpt), device="cpu").identity() != base
    assert HFGenModel(model=str(ckpt), revision="abc").identity() != base
    assert HFGenModel(model=str(ckpt), batch_size=1).identity() != HFGenModel(model=str(ckpt), batch_size=8).identity()


def test_equivalent_configs_share_an_identity(ckpt):
    a = HFGenModel(model=str(ckpt), dtype="float16", batch_size=4)
    b = HFGenModel(model=str(ckpt), batch_size=4, dtype="torch.float16", device="auto", hf_token="secret")
    assert a.identity() == b.identity()                      # key order, torch. prefix, auto device
    assert "secret" not in str(b.identity())                 # the token never enters the fingerprint


def test_a_hub_model_with_default_settings_keeps_its_identity():
    assert HFGenModel(model="no/such-hub-model").identity() == {"name": "hf", "model": "no/such-hub-model"}


def test_rewriting_the_local_weights_changes_the_identity(ckpt):
    m = HFGenModel(model=str(ckpt))
    before = m.identity()
    assert m.identity() == before                            # stable while nothing changes
    (ckpt / "model.safetensors").write_bytes(b"b" * 20)
    assert m.identity() != before


def test_vllm_identity_sees_engine_kwargs_and_weights(ckpt):
    base = VLLMModel(model=str(ckpt)).identity()
    assert VLLMModel(model=str(ckpt), dtype="float16").identity() != base
    (ckpt / "model.safetensors").write_bytes(b"b" * 20)
    assert VLLMModel(model=str(ckpt)).identity() != base


def test_evaluate_does_not_replay_another_batch_size(ckpt):
    calls = []

    class _Fake(HFGenModel):
        def generate(self, requests):
            calls.append(self._extra_kwargs.get("batch_size"))
            return [Result_(completions=[Generated(text="4")]) for _ in requests]

    for bs in (1, 8, 8):
        ak.evaluate(SAMPLES, model=_Fake(model=str(ckpt), batch_size=bs))
    assert calls == [1, 8]                                   # the third is a genuine cache hit


# ---- #50, hosted backends ------------------------------------------------

import json
import warnings
from urllib.parse import urlsplit

from auditkit.model import _EXCLUDED_WARNED, _identity_value
from auditkit.model.anthropic import AnthropicModel
from auditkit.model.groq_gen import GroqModel
from auditkit.model.litellm_gen import LiteLLMModel
from auditkit.model.lexsi import LexsiModel
from auditkit.model.openai import OpenAIModel
from auditkit.model.openrouter_gen import OpenRouterModel

DEFAULTS = [
    (OpenAIModel, "gpt-4o-mini", "openai"),
    (AnthropicModel, "claude-sonnet-4-20250514", "anthropic"),
    (GroqModel, "llama-3.3-70b-versatile", "groq"),
    (LiteLLMModel, "gpt-4o-mini", "litellm"),
    (OpenRouterModel, "openai/gpt-4o-mini", "openrouter"),
    (LexsiModel, "lexsi-3.5", "lexsi"),
]
SECRET = "sk-live-0123456789abcdef"


@pytest.mark.parametrize("cls,model,name", DEFAULTS)
def test_hosted_default_config_keeps_its_identity(cls, model, name, monkeypatch):
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LEXSI_API_KEY"):
        monkeypatch.setenv(var, "k")
    assert cls().identity() == {"name": name, "model": model}
    assert cls(api_key="a").identity() == cls(api_key="b").identity()         # a key never keys the run


@pytest.mark.parametrize("kw", [{"organization": "org-b"}, {"project": "p"}, {"api_version": "2024-10-21"},
                                {"reasoning_effort": "high"}, {"api_base": "https://proxy.example/v1"}])
def test_openai_settings_change_the_identity(kw):
    assert OpenAIModel(**kw).identity() != OpenAIModel().identity()


@pytest.mark.parametrize("cls,default_base", [
    (GroqModel, "https://api.groq.com/openai/v1"),
    (OpenRouterModel, "https://openrouter.ai/api/v1"),
    (LexsiModel, "https://api.lexsi.ai/v1"),
])
def test_requests_backends_see_endpoint_and_body_fields(cls, default_base):
    base = cls().identity()
    assert cls(api_base="https://other.example/v1").identity() != base
    assert cls(api_base=default_base + "/").identity() == base                # the default, trailing slash or not
    assert cls(reasoning_effort="high").identity() != base
    assert cls(timeout=5, verify=False).identity() == base                    # transport settings stay out


def test_litellm_and_anthropic_see_their_kwargs():
    assert LiteLLMModel(api_base="https://x.example").identity() != LiteLLMModel().identity()
    assert AnthropicModel(base_url="https://x.example").identity() != AnthropicModel().identity()
    assert AnthropicModel(thinking={"type": "enabled", "budget_tokens": 1000}).identity() != AnthropicModel().identity()
    assert AnthropicModel(thinking={"budget_tokens": 1}).identity() != AnthropicModel(thinking={"budget_tokens": 2}).identity()
    assert OpenAIModel(max_retries=1, timeout=3).identity() == OpenAIModel().identity()


LEAKY = [
    lambda s: OpenAIModel(default_headers={"Authorization": f"Bearer {s}", "X-Route": "a"}),
    lambda s: OpenAIModel(default_headers={"x-api-key": s}),
    lambda s: OpenAIModel(default_headers=[("Authorization", s), ("X-Route", "a")]),          # pair list
    lambda s: OpenAIModel(api_base=f"https://user:{s}@proxy.example/v1?api_key={s}"),
    lambda s: OpenAIModel(api_base=f"https://proxy.example/v1;token={s}?{s}&token#{s}"),     # ; query, bare parts
    lambda s: OpenAIModel(apiKey=s, accessToken=s, access_tokens=s, apiauthors=s, clientSecret=s),   # camelCase, odd names
    lambda s: OpenAIModel(extra_headers=[f"api_key=b: {s}", f"a=b: {s}"], default_query={"?": s}),    # header strings
    lambda s: OpenAIModel(default_headers={"X-Custom": s}, response_format={"type": "json", "api_key": s}),
    lambda s: AnthropicModel(default_headers={"X-Custom": s}, base_url=f"https://h.example/?key={s}"),
    lambda s: LiteLLMModel(api_key=s, extra_headers={"Authorization": s}),
    lambda s: LiteLLMModel(api_base=f"https://{s}@h.example/v1", custom_llm_provider="azure"),
    lambda s: GroqModel(extra_headers={"api-key": s}),
    lambda s: LexsiModel(api_base=f"https://{s}@h.example/v1", client_blob=_pem("PGP PRIV" "ATE KEY BLOCK")),
    lambda s: OpenRouterModel(default_query={"token": s}, organization=s),
]


@pytest.mark.parametrize("make", LEAKY)
def test_a_secret_never_reaches_the_identity_or_the_cache_key(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ident = make(SECRET).identity()
        rotated = make("sk-rotated-9999999999999999").identity()
    text = json.dumps(ident, default=str)
    assert SECRET not in text and "BEGIN" not in text and "Bearer" not in text
    assert ident == rotated                                                   # rotation keeps the cache


@pytest.mark.parametrize("make", LEAKY[:3] + LEAKY[5:9])
def test_non_allowlisted_kwargs_are_left_out_with_one_warning(make):
    _EXCLUDED_WARNED.clear()
    with pytest.warns(UserWarning, match="identity_extra") as rec:
        make(SECRET).identity()
        make(SECRET).identity()
    assert len([w for w in rec if "identity_extra" in str(w.message)]) == 1   # once
    assert all(SECRET not in str(w.message) for w in rec)                     # names, never values


def test_identity_extra_opts_a_kwarg_in_and_it_is_still_filtered():
    m = OpenAIModel(default_headers={"X-Route": "a", "Authorization": f"Bearer {SECRET}"},
                    identity_extra=["default_headers"])
    ident = m.identity()
    assert ident["extra"]["default_headers"] == {"Authorization": "<redacted>", "X-Route": "a"}
    assert SECRET not in json.dumps(ident)
    assert ident != OpenAIModel().identity()
    assert m.identity() != OpenAIModel(default_headers={"X-Route": "b", "Authorization": "z"},
                                       identity_extra=["default_headers"]).identity()


def test_allowlisted_ids_must_be_plain_ids():
    assert OpenAIModel(organization="org-b").identity()["extra"] == {"organization": "org-b"}
    assert OpenAIModel(organization=SECRET + " x").identity()["extra"] == {"organization": "<redacted>"}
    assert OpenAIModel(api_version=f"{SECRET}").identity()["extra"] == {"api_version": "<redacted>"}
    assert OpenAIModel(response_format={"type": "json_object"}).identity()["extra"] == {
        "response_format": {"type": "json_object"}}


def test_hosted_endpoint_is_scheme_host_and_path_only():
    ident = OpenAIModel(api_base=f"https://u:{SECRET}@H.example:8443/v1/?api-version=1&k=v#f").identity()
    assert ident["api_base"] == "https://H.example:8443/v1"


def test_a_secret_never_reaches_the_cache_files(tmp_path):
    m = OpenAIModel(default_headers={"Authorization": f"Bearer {SECRET}"}, identity_extra=["default_headers"])
    ak.evaluate(SAMPLES, model=_Echo(m))
    blob = b"".join(p.read_bytes() for p in (tmp_path / "cache").rglob("*") if p.is_file())
    assert blob and SECRET.encode() not in blob


class _Echo:
    """An OpenAIModel that answers locally, so the run caches without a network."""
    def __new__(cls, model):
        model.generate = lambda requests: [Result_(completions=[Generated(text="4")]) for _ in requests]
        return model


def test_output_changing_names_that_contain_token_are_kept():
    assert _identity_value({"max_completion_tokens": 5}) == {"max_completion_tokens": 5}
    assert _identity_value({"budget_tokens": 5, "access_token": "x"}) == {"budget_tokens": 5, "access_token": "<redacted>"}


def test_the_api_backend_hides_secrets_too():
    from auditkit.model.api_gen import APIModel
    a = APIModel(model="m", api_base="http://h/v1", extra_headers={"Authorization": SECRET}).identity()
    assert SECRET not in json.dumps(a, default=str)


# ---- #50, the deny-list fails closed -------------------------------------------------------

from auditkit.model import _scrub_url


@pytest.mark.parametrize("key", ["apiKey", "accessToken", "clientSecret", "AuthToken", "x-api-key", "API_KEY",
                                 "X-Auth-Token", "privateKey", "Authorization", "use_auth_token", "dbPassword"])
def test_secret_names_in_any_case_style_are_redacted(key):
    assert _identity_value({key: "anything", "n": 1})[key] == "<redacted>"
    assert _identity_value({key: 1234})[key] == "<redacted>"                  # a PIN is a number


@pytest.mark.parametrize("key", ["max_tokens", "max_new_tokens", "num_tokens", "max_completion_tokens",
                                 "budget_tokens", "maxTokens", "tokenizer_kwargs", "token_type_ids", "keyword"])
def test_count_and_tokenizer_names_stay(key):
    assert _identity_value({key: 7}) == {key: 7}


def test_pair_lists_are_walked_as_key_value_pairs():
    out = _identity_value({"default_headers": [("Authorization", f"Bearer {SECRET}"), ["X-Route", "a"], ("apiKey", "z")]})
    assert out == {"default_headers": [["Authorization", "<redacted>"], ["X-Route", "a"], ["apiKey", "<redacted>"]]}
    assert SECRET not in json.dumps(out)
    assert _identity_value({"stop": ["a", "b"]}) == {"stop": ["a", "b"]}      # a plain list is not pairs
    assert OpenAIModel(default_headers=[("X-Route", "a"), ("Authorization", "x")],
                       identity_extra=["default_headers"]).identity() != OpenAIModel().identity()


def test_header_strings_are_redacted():
    assert _identity_value({"h": ["Authorization: Bearer abcdefgh1234"]}) == {"h": ["<redacted>"]}
    assert _identity_value({"h": ["x-api-key: zz"]}) == {"h": ["x-api-key: <redacted>"]}


def test_urls_lose_userinfo_fragment_and_every_query_value_but_version_settings():
    url = f"https://user:{SECRET}@h.example/v1/?api-version=2024-10-21&key=a1;token=b2&foo=bar#frag-{SECRET}"
    out = _scrub_url(url)
    assert out == "https://h.example/v1?api-version=2024-10-21&key=<redacted>&token=<redacted>&foo=<redacted>"
    assert _scrub_url("https://h.example/v1/") == _scrub_url("https://h.example/v1")
    assert _identity_value({"proxies": {"https": f"http://u:{SECRET}@p.example:3128"}}) == {"proxies": {"https": "http://p.example:3128"}}
    ident = OpenAIModel(api_base=f"https://u:{SECRET}@h.example/v1?foo={SECRET}").identity()
    assert SECRET not in json.dumps(ident) and ident["api_base"] == "https://h.example/v1"
    from auditkit.model.api_gen import APIModel
    assert SECRET not in json.dumps(APIModel(model="m", api_base=f"http://u:{SECRET}@h.example/v1").identity())


@pytest.mark.parametrize("value", ["sk-proj-abcdefghijklmnop", "ghp_abcdefghijklmnopqrstuvwx", "xoxb-1234-abcdefgh",
                                   "eyJhbGciOiJI.eyJzdWIiOiIx.abcdefgh", "a1B2c3D4e5F6g7H8i9J0k1L2m3",
                                   "AKIAABCDEFGHIJKLMNOP", "Bearer abcdefgh12345"])
def test_key_looking_values_are_redacted_under_any_name(value):
    assert _identity_value({"X-Custom": value, "note": value}) == {"X-Custom": "<redacted>", "note": "<redacted>"}
    assert _identity_value({"extra": [value]}) == {"extra": ["<redacted>"]}


def test_ordinary_values_stay():
    kept = {"revision": "3f786850e387550fdab836ed7e6dc881de23001b", "model": "meta-llama/Llama-3.1-8B-Instruct",
            "attn_implementation": "flash_attention_2", "chat_template": "You are a helpful assistant, be brief.",
            "path": "/data/checkpoints/run-2026-11-13-final", "reasoning_effort": "high", "n": 2}
    assert _identity_value(kept) == kept
    assert HFGenModel(model="x/y", revision="3f786850e387550fdab836ed7e6dc881de23001b").identity() != HFGenModel(
        model="x/y", revision="0" * 40).identity()


# ---- armoured blocks and the other value patterns --------------------------------------------

def _pem(kind, tail="-----"):
    return f"-----BEGIN {kind}{tail}\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END {kind}-----"


@pytest.mark.parametrize("kind", ["PGP PRIV" "ATE KEY BLOCK", "RSA PRIV" "ATE KEY", "OPENSSH PRIV" "ATE KEY",
                                  "PRIV" "ATE KEY", "EC PRIV" "ATE KEY", "ENCRYPTED PRIV" "ATE KEY", "PGP MESSAGE"])
def test_armoured_blocks_are_redacted(kind):
    assert _identity_value({"x": _pem(kind)}) == {"x": "<redacted>"}
    assert _identity_value({"x": [_pem(kind)]}) == {"x": ["<redacted>"]}


@pytest.mark.parametrize("text", ["-----BEGIN RSA PRIV", "-----BEGIN", "-----BEGIN OPENSSH PRIV" "ATE KEY",
                                  "-----BEGIN KEY-----MIIEvQIBADANBg"])
def test_truncated_and_odd_headers_are_redacted(text):
    assert _identity_value({"x": text}) == {"x": "<redacted>"}


def test_a_public_certificate_stays_in_the_fingerprint():
    cert = _pem("CERTIFICATE")
    assert _identity_value({"x": cert}) == {"x": cert}


@pytest.mark.parametrize("text", ["eyJhbGciOiJIUzI1NiJ9", "eyJ", "Bearer abc", "basic dXNlcjpwdw==", "sk-abc", "hf_x",
                                  "see sk-abcdefghijkl here", "Authorization: Bearer abcdefghijkl", "ghp_abc"])
def test_the_earlier_value_patterns_still_match(text):
    assert _identity_value({"x": text}) == {"x": "<redacted>"}


def test_prose_that_merely_resembles_a_pattern_stays():
    kept = {"a": "a bearer of news", "b": "task_description_long", "c": "basic usage is fine".lower(), "d": "begin"}
    assert _identity_value(kept) == {"a": "a bearer of news", "b": "task_description_long",
                                     "c": "basic usage is fine", "d": "begin"}


# ---- the sanitiser behind identity_extra / hf: / vllm: / api: -------------------------------

def test_bare_query_parts_are_redacted():
    assert _scrub_url("https://h.example/v1?sk-abc123&token&api-version&x=1") == \
        "https://h.example/v1?<redacted>&<redacted>&api-version&x=<redacted>"


@pytest.mark.parametrize("key", ["access_tokens", "refresh_tokens", "apiauthors", "authTokens", "accessTokens"])
def test_only_exact_count_names_are_exempt(key):
    assert _identity_value({key: "abc"}) == {key: "<redacted>"}


@pytest.mark.parametrize("text,expected", [
    ("api_key=b: c", "api_key= <redacted>"), ("Authorization: x=y", "Authorization: <redacted>"),
    ("  x-api-key =  zz ", "x-api-key= <redacted>"), ("a=b: c", "a=b: c"),
])
def test_header_strings_split_on_the_first_delimiter(text, expected):
    assert _identity_value({"h": [text]}) == {"h": [expected]}


# ---- credentials in a URL path ---------------------------------------------------------------

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"
PATH_SECRETS = [SECRET, "hf_AbCdEfGhIjKlMnOpQrSt", "ghp_abcdefghijklmnopqrstuvwx", JWT,
                "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                "dGhpcyBpcyBhIHNlY3JldCB0b2tlbiB2YWx1ZQ==", "%73k-live-0123456789abcdef"]


@pytest.mark.parametrize("secret", PATH_SECRETS)
@pytest.mark.parametrize("make", [
    lambda u: OpenAIModel(api_base=u),
    lambda u: LiteLLMModel(api_base=u),
    lambda u: AnthropicModel(base_url=u),
    lambda u: LiteLLMModel(base_url=u),
    lambda u: GroqModel(api_base=u),
    lambda u: OpenRouterModel(api_base=u),
    lambda u: LexsiModel(api_base=u),
])
def test_a_credential_in_the_url_path_never_reaches_the_identity(make, secret):
    for path in (f"/v1/{secret}/", f"/{secret}", f"/key/{secret}/v1", f"/v1/{secret}/chat"):
        ident = make(f"https://host.example{path}").identity()
        text = json.dumps(ident)
        assert secret not in text and secret.lstrip("%73") not in text, path
        assert urlsplit(ident["api_base" if "api_base" in ident else "base_url"]).netloc == "host.example"
    assert make(f"https://host.example/v1/{secret}").identity() == make("https://host.example/v1/sk-rotated-9999999999999").identity()


def test_a_short_token_after_a_key_segment_is_redacted():
    assert OpenAIModel(api_base="https://h.example/key/abc/v1").identity()["api_base"] == "https://h.example/key/<redacted>/v1"
    assert OpenAIModel(api_base="https://h.example/token/xyz").identity()["api_base"] == \
        "https://h.example/token/<redacted>"


@pytest.mark.parametrize("path", ["/v1", "/openai/deployments/gpt-4o", "/openai/deployments/gpt-4o-mini-2024-07-18/v1",
                                  "/api/v1/chat", "/v1beta/models/gemini-2.5-pro"])
def test_ordinary_paths_stay_intact(path):
    assert OpenAIModel(api_base=f"https://host.example{path}/").identity()["api_base"] == f"https://host.example{path}"
    assert OpenAIModel(api_base=f"https://host.example{path}").identity() != OpenAIModel(
        api_base="https://host.example/other").identity()


def test_urls_inside_proxies_and_the_api_backend_are_cleaned_too():
    from auditkit.model.api_gen import APIModel
    url = f"https://u:p@host.example/v1/{SECRET}/x"
    assert _identity_value({"proxies": {"https": url}}) == {"proxies": {"https": "https://host.example/v1/<redacted>/x"}}
    assert SECRET not in json.dumps(APIModel(model="m", api_base=url).identity())
