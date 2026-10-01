"""``Sample.images`` must reach an ``api:`` model, or the run must say it did not.

The backend used to read ``params["images"]`` nowhere. A vision eval over ``api:``
-- a hosted provider, a self-hosted vLLM or SGLang server -- sent the text, dropped
the image, raised nothing, and returned a real-looking score for a model that had
never seen the picture. That is worse than a failure, and these tests pin both halves
of the fix: the image goes out, and the one mode that cannot carry it refuses.

No network and no pillow: the tests exercise message construction and capture the
posted body with a stub session.
"""

from __future__ import annotations

import base64

import pytest

from auditkit.model import Request
from auditkit.model.api_gen import APIModel, _image_part, messages_with_images


class _FakeImage:
    """A PIL-shaped object, so the tests need no pillow."""

    format = "PNG"

    def save(self, buffer, format=None):  # noqa: A002 - matches PIL's signature
        # A 1x1 PNG, so the encoded data URI is a real, decodable image.
        buffer.write(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))


def _request(**params):
    base = {"messages": [{"role": "user", "content": "what is this?"}]}
    base.update(params)
    return Request(prompt="what is this?", params=base)


# -- where the image goes --------------------------------------------------------

def test_images_go_at_the_start_of_the_last_user_turn():
    # Same rule as hf:, so one sample builds the same prompt on either backend.
    messages = messages_with_images(_request(images=[_FakeImage()]))
    content = messages[-1]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert content[-1]["text"] == "what is this?"


def test_earlier_turns_are_untouched():
    request = Request(prompt="p", params={"messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"}], "images": [_FakeImage()]})
    messages = messages_with_images(request)
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "first"}
    assert messages[2] == {"role": "assistant", "content": "ok"}


def test_several_images_all_land_on_that_turn():
    messages = messages_with_images(_request(images=[_FakeImage(), _FakeImage()]))
    assert [c["type"] for c in messages[-1]["content"]] == ["image_url", "image_url", "text"]


def test_an_already_structured_content_list_keeps_its_order():
    request = Request(prompt="p", params={"messages": [
        {"role": "user", "content": [{"type": "text", "text": "compare these"}]}],
        "images": [_FakeImage()]})
    content = messages_with_images(request)[-1]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert content[-1]["text"] == "compare these"


def test_a_request_with_no_images_is_unchanged():
    messages = [{"role": "user", "content": "hi"}]
    assert messages_with_images(Request(prompt="hi", params={"messages": messages})) == messages


# -- how the image is encoded ----------------------------------------------------

def test_a_pil_image_becomes_a_png_data_uri():
    part = _image_part(_FakeImage())
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(part["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG")


def test_a_remote_url_is_passed_through_rather_than_inlined():
    # A hosted provider can fetch it itself, and inlining would bloat every request.
    for url in ("https://example.com/a.png", "http://example.com/b.jpg", "data:image/png;base64,AAA"):
        assert _image_part(url)["image_url"]["url"] == url


def test_a_local_path_needs_pillow_and_says_so():
    # pillow is an optional extra, so the failure has to name it rather than be
    # an ImportError from three frames down.
    try:
        import PIL  # noqa: F401
    except ImportError:
        from auditkit.errors import ExtraNotInstalled
        with pytest.raises(ExtraNotInstalled, match="vision"):
            _image_part("/nonexistent/does-not-matter.png")


# -- the case that used to fail silently ----------------------------------------

def test_images_with_no_user_turn_raise_rather_than_vanish():
    request = Request(prompt="p", params={
        "messages": [{"role": "system", "content": "only a system turn"}], "images": [_FakeImage()]})
    with pytest.raises(ValueError, match="no user turn"):
        messages_with_images(request)


def test_raw_prompt_mode_refuses_images_instead_of_sending_text_only():
    # /v1/completions takes a flat string; there is nowhere to put a content part.
    # The old behaviour was to send the text and let the score imply the image was seen.
    model = APIModel(model="m", api_base="http://localhost:8000/v1", chat_template=False)
    request = _request(images=[_FakeImage()])
    with pytest.raises(ValueError, match="cannot carry an image"):
        model._one(request)


# -- the wire --------------------------------------------------------------------

def test_the_posted_body_carries_the_image(monkeypatch):
    captured = {}

    class _Response:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "a red square"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    class _Session:
        def post(self, url, json=None, timeout=None):
            captured.update(url=url, body=json)
            return _Response()

    model = APIModel(model="CohereLabs/aya-vision-8b", api_base="http://localhost:8000/v1", api_key="EMPTY")
    monkeypatch.setattr(model, "_session", _Session())
    model.generate([_request(images=[_FakeImage()])])

    assert captured["url"].endswith("/chat/completions")
    content = captured["body"]["messages"][-1]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
