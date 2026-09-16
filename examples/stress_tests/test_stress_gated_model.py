#!/usr/bin/env python3
"""Stress test: a genuinely gated HuggingFace model, run through AuditKit itself.

Uses google/gemma-2b -- a real, currently-gated model requiring the caller to
(a) have a HuggingFace account, (b) accept Google's license on the model page,
and (c) pass a valid access token. This script deliberately does NOT provide
a token, to reproduce exactly what a typical user without pre-approved access
sees when they try to evaluate a gated model through auditkit.

Run with: python3 examples/stress_tests/test_stress_gated_model.py
Needs: pip install -e ".[transformers,interop]"
"""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import auditkit as ak
from auditkit.model.hf_gen import HFGenModel
from auditkit.adapter import MCQAdapter

GATED_MODEL = "google/gemma-2b"


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


section("1. Does HFGenModel even accept a token= argument at all?")
import inspect
sig = inspect.signature(HFGenModel.__init__)
print(f"HFGenModel.__init__ signature: {sig}")
has_token_param = "token" in sig.parameters or "use_auth_token" in sig.parameters
print(f"Has a token/use_auth_token parameter: {has_token_param}")
if not has_token_param:
    print("-> CONFIRMED: there is no way to pass an access token through auditkit's")
    print("   own API for this backend. Even a user who HAS a valid token and has")
    print("   accepted the model's license has no documented way to supply it.")


section("2. Force os.environ['HF_TOKEN'] to be invalid/absent, then try loading the gated model directly")
os.environ["HF_TOKEN"] = "invalid_token_forcing_denial"
try:
    m = HFGenModel(model=GATED_MODEL, device="cpu", max_new_tokens=8)
    from auditkit.model import Request
    result = m.generate([Request(prompt="What is the capital of France?")])
    print("[UNEXPECTED] generate() succeeded:", result[0].text)
except Exception as e:
    print(f"[REPRODUCED] {type(e).__name__}")
    print(f"Full message:\n{e}")


section("3. Same thing, through the full public ak.evaluate() pipeline")
try:
    result = ak.evaluate(
        [ak.Sample(input="capital of France?", choices=["Paris", "Rome"], target="A")],
        model=HFGenModel(model=GATED_MODEL, device="cpu", max_new_tokens=8),
        adapter=MCQAdapter(method="mcq_joint"),
        scorers="acc",
    )
    print("[UNEXPECTED] evaluate() succeeded:", result.headline)
except Exception as e:
    print(f"[REPRODUCED through evaluate()] top-level message: {type(e).__name__}: {e}")
    print()
    print("Note: the top-level message alone ('generate failed after 3 retries')")
    print("gives NO indication this was an access/auth problem specifically -- it")
    print("looks identical to what you'd see for a network blip or a typo'd model")
    print("name. The real cause IS preserved (Python exception chaining, e.__cause__),")
    print("so it's recoverable by a developer who knows to look, but it's not")
    print("surfaced anywhere in the message a user actually sees by default:")
    cause = e.__cause__
    if cause is not None:
        print(f"  e.__cause__ = {type(cause).__name__}: {str(cause)[:200]}...")


section("WHY THIS HAPPENS -- root cause")
print("""
1. google/gemma-2b is a real, currently-gated model on HuggingFace. Accessing
   it requires: a HF account, accepting Google's usage license on the model's
   page, and a valid access token with read permission for gated repos.
   Confirmed directly: a bare tokenizer load with no token gets a clean,
   explicit error straight from HuggingFace's server:
     "401 Client Error... Access to model google/gemma-2b is restricted.
      You must have access to it and be authenticated to access it."

2. auditkit's HFGenModel (src/auditkit/model/hf_gen.py) has NO token/
   use_auth_token constructor parameter at all -- confirmed above via
   inspect.signature(). It always calls transformers.pipeline() with only
   `model=`, `device=`, and generation kwargs; nothing related to auth is
   ever threaded through.

3. Because of that, the *only* way a gated model can work through auditkit is
   if the machine running it already has a valid HF token cached globally
   (e.g. via `huggingface-cli login`, or an HF_TOKEN env var picked up
   automatically by the `transformers`/`huggingface_hub` libraries
   underneath -- NOT something auditkit itself manages or exposes).

4. With no valid credential, going through the full pipeline (auto-selected
   model class + fallback to the specific architecture class) produces a
   *different-looking* error than the bare tokenizer test: a ValueError
   claiming the repo "does not appear to have a file named pytorch_model.bin
   or model.safetensors" -- because the unauthenticated file-listing call
   can't see the (present but access-restricted) weight files at all, so it
   looks like they're simply missing rather than access-denied. Same root
   cause (no auth), different and more confusing surface error depending on
   which code path hits the 401 first.

5. Going through auditkit's own Runner, this gets retried 3 times (the
   built-in retry-with-backoff logic correctly kicks in, since ModelError is
   treated as possibly-transient) and finally surfaces as a generic
   "ModelError: generate failed after 3 retries" -- which does NOT mention
   authentication, gating, or the model name at all in its own message. The
   real cause is preserved via Python's exception chaining (e.__cause__),
   so it's recoverable by a developer who knows to check, but a typical user
   reading just the printed error has no way to tell this was an access
   problem rather than, say, a flaky network or a mistyped model name.

Net finding: auditkit's local-model backend cannot use any gated model
unless the *host machine* is already authenticated outside of auditkit
entirely, and when it fails, the error surfaced to the user does not explain
why.
""")
