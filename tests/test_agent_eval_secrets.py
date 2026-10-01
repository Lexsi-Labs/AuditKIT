"""Automatic secret detection in shareable exports (UX-A9).

Conservative: strong patterns + a credential key-name + a high-entropy heuristic,
never touching AuditKit reproducibility fields (hashes, ids). Benign SYNTHETIC
secrets only (AWS's documented example key, fake `sk-`/bearer tokens).
"""

from __future__ import annotations

import pytest

from auditkit.agent_eval import (
    AgentEvalRunner,
    AgentEvalSpec,
    FinalStateAssertion,
    detect_secrets,
    episode_from_openai_messages,
    redact,
)
from auditkit.agent_eval.types import REDACTED

# High entropy (~5.0), mixed alnum, not hex/uuid -> flagged by the heuristic.
_HIGH_ENTROPY = "Zk9Qw3Xy7Lm2Rp8Vn4Tb6Hs1Dc5Fg0Aj7Yw2"

_OBJ = {
    "authorization": "Bearer synthetic-abcd1234efgh5678",   # credential key name
    "note": "aws AKIAIOSFODNN7EXAMPLE and openai sk-abcdefghijklmnopqrstuvwx1234",
    "blob": _HIGH_ENTROPY,
    "id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",            # uuid: must survive
    "digest": "sha256:" + "a1b2c3d4" * 8,                    # hash: must survive
    "usage": {"total_tokens": 42, "prompt_tokens": 10},      # ints: never redacted
}


def test_detect_finds_strong_patterns_key_names_and_entropy():
    kinds = {(f["path"], f["kind"]) for f in detect_secrets(_OBJ)}
    assert ("authorization", "secret_key_name") in kinds
    assert ("note", "openai_key") in kinds
    assert ("note", "aws_access_key_id") in kinds
    assert ("blob", "high_entropy") in kinds


def test_redact_auto_redacts_secrets_but_preserves_repro_and_ints():
    out = redact(_OBJ, auto=True)
    assert out["authorization"] == REDACTED
    assert REDACTED in out["note"] and "AKIA" not in out["note"] and "sk-" not in out["note"]
    assert out["blob"] == REDACTED
    # reproducibility identity and numeric telemetry are untouched
    assert out["id"] == _OBJ["id"]
    assert out["digest"] == _OBJ["digest"]
    assert out["usage"] == {"total_tokens": 42, "prompt_tokens": 10}


def test_allow_whitelists_a_token():
    out = redact({"note": "sk-abcdefghijklmnopqrstuvwx1234"}, auto=True,
                 allow=["sk-abcdefghijklmnopqrstuvwx1234"])
    assert out["note"] == "sk-abcdefghijklmnopqrstuvwx1234"
    assert detect_secrets({"note": "sk-abcdefghijklmnopqrstuvwx1234"},
                          allow=["sk-abcdefghijklmnopqrstuvwx1234"]) == []


def test_no_false_positive_on_a_clean_run_result():
    # The discriminating check: a real exported result carries case digests and
    # uuid-ish ids; auto-detection must not fire on any of them.
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "t"}, {"role": "assistant", "content": "done"}],
        case_id="c", final_answer="done")
    ep.final_state = {"shipped": True}
    from auditkit.agent_eval import AgentCase
    case = AgentCase(id="c", task="ship it", allowed_tools=["ship"],
                     outcome=FinalStateAssertion("shipped", equals=True))
    result = AgentEvalRunner().run(AgentEvalSpec(
        cases=[case], mode="recorded", episodes=[ep], scorers=["tool_call_validity"]))
    assert detect_secrets(result.to_dict()) == []


def test_secret_under_generic_key_and_nested_credential_is_redacted():
    """A high-entropy secret under a generic name, and a secret nested under a
    credential-named key, must both be redacted; repro fields survive."""
    from auditkit.agent_eval.types import redact
    aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    out = redact({"key": aws_secret,
                  "password": {"value": "topsecretpassword123"},
                  "api_key": ["sk-proj-" + "a" * 40, "plainsecretvalue"],
                  "digest": "sha256:" + "a" * 64, "id": "b" * 32}, auto=True)
    assert out["key"] == "[REDACTED]"
    assert out["password"]["value"] == "[REDACTED]"
    assert all(x == "[REDACTED]" for x in out["api_key"])
    assert out["digest"] == "sha256:" + "a" * 64 and out["id"] == "b" * 32  # repro survives


# -- D1-D3 (fix plan #15) --------------------------------------------------------------------------------

_KEY_SECRET = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"


def test_d1_a_secret_used_as_a_key_is_redacted_and_reported_without_echo():
    other = "sk-proj-" + "Z9y8X7w6V5u4T3s2R1q0P9o8"
    out = redact({_KEY_SECRET: "cached", other: 1, "safe": "x"}, auto=True)
    assert out == {REDACTED: "cached", f"{REDACTED}#2": 1, "safe": "x"}      # two keys, no collision
    found = detect_secrets({"cache": {_KEY_SECRET: 1}})
    assert found == [{"path": f"cache.{REDACTED}", "key": REDACTED, "kind": "openai_key"}]
    assert _KEY_SECRET not in repr(found)


def test_d1_a_secret_key_is_not_echoed_by_its_string_values_finding():
    # the value matches a strong pattern too: its finding must not carry the raw key
    found = detect_secrets({_KEY_SECRET: "sk-proj-" + "Q1w2E3r4T5y6U7i8O9p0A1s2"})
    assert {f["kind"] for f in found} == {"openai_key"} and len(found) == 2
    assert all(f["key"] == REDACTED for f in found)
    assert _KEY_SECRET not in repr(found)


def test_d2_numbers_under_a_credential_key_are_redacted_but_counts_are_kept():
    out = redact({"password_min_length": 12, "token": 5, "password": "hunter2", "pin": 1234,
                  "otp": 481516, "api_key": {"retries": 3, "value": 99, "enabled": True,
                                             "note": None, "ttl": 60}}, auto=True)
    assert out == {"password_min_length": 12, "token": REDACTED, "password": REDACTED,
                   "pin": REDACTED, "otp": REDACTED,
                   "api_key": {"retries": 3, "value": REDACTED, "enabled": True,
                               "note": None, "ttl": 60}}


def test_d2_the_count_key_allowlist_is_an_argument():
    assert redact({"pin_count": 3}, auto=True) == {"pin_count": 3}
    assert redact({"pin_count": 3}, auto=True, count_keys=()) == {"pin_count": REDACTED}
    assert redact({"password": {"n": 3}}, auto=True, count_keys=["n"]) == {"password": {"n": 3}}


@pytest.mark.parametrize("text,redacted", [
    ("The fund holds bearer instruments and registered bonds.", False),     # banking prose
    ("Authorization: Bearer abc123def456ghi789", True),                     # a token shape
    ("Bearer eyJhbGciOiJIUzI1NiJ9", True),
])
def test_d3_the_bearer_pattern_needs_a_token_shape(text, redacted):
    out = redact({"note": text}, auto=True)["note"]
    assert (REDACTED in out) is redacted
