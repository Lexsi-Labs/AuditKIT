"""Hackathon models on the hf: backend, CPU only, tiny random weights built
in-test (no downloads, no gated repos):

* Aya Expanse (``cohere``): a chat-templated prompt reaches the model with ONE
  ``<BOS_TOKEN>``, not two (the template already renders it).
* Aya Vision (``aya_vision``): a ``Sample`` carrying an image goes through the
  processor with ``pixel_values``; text-only samples on the same model still work.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")

import auditkit as ak  # noqa: E402
from auditkit.model.hf_gen import HFGenModel  # noqa: E402
from auditkit.sample import Sample  # noqa: E402

BOS = "<BOS_TOKEN>"
IMAGE_TOKENS = ["<image>", "<|START_OF_IMG|>", "<|END_OF_IMG|>", "<|IMG_LINE_BREAK|>", "<|IMG_PATCH|>"]
SPECIALS = ["<PAD>", "<UNK>", BOS, "<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>",
            "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>", *IMAGE_TOKENS]
# Cohere-style template: starts with the BOS, like Aya's real one. Image items
# render as <image>, which the Aya Vision processor expands to patch tokens.
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<image>{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endif %}"
    "<|END_OF_TURN_TOKEN|>{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"
)
SMALL = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2)


def _tokenizer():
    """A tiny word-level fast tokenizer that, like Aya's, prepends <BOS_TOKEN>."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    words = "what is two plus the capital of france describe this image a b c".split()
    vocab = {t: i for i, t in enumerate(SPECIALS + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<UNK>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", special_tokens=[(BOS, vocab[BOS])])
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token=BOS, eos_token="<|END_OF_TURN_TOKEN|>",
        pad_token="<PAD>", unk_token="<UNK>", additional_special_tokens=SPECIALS[4:])
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _ids(tok):
    return dict(vocab_size=len(tok), pad_token_id=tok.pad_token_id,
                bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)


def _spy_generate(model):
    """Record the input_ids (and whether pixel_values came along) per generate()."""
    seen = []
    orig = model.generate

    def spy(*args, **kwargs):
        seen.append((kwargs["input_ids"][0].tolist(), "pixel_values" in kwargs))
        return orig(*args, **kwargs)

    model.generate = spy
    return seen


@pytest.fixture(scope="module")
def cohere_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cohere")
    tok = _tokenizer()
    torch.manual_seed(0)
    transformers.CohereForCausalLM(transformers.CohereConfig(**SMALL, **_ids(tok))).save_pretrained(d)
    tok.save_pretrained(d)
    return d, tok


@pytest.fixture(scope="module")
def aya_vision_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("aya_vision")
    tok = _tokenizer()
    cfg = transformers.AyaVisionConfig(
        vision_config=dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=2, image_size=64, patch_size=16),
        text_config=dict(model_type="cohere2", **SMALL, **_ids(tok), sliding_window=64,
                         layer_types=["sliding_attention", "full_attention"]),
        downsample_factor=2, image_token_index=tok.convert_tokens_to_ids("<|IMG_PATCH|>"))
    torch.manual_seed(0)
    transformers.AyaVisionForConditionalGeneration(cfg).save_pretrained(d)
    ip = transformers.GotOcr2ImageProcessor(size={"height": 64, "width": 64},
                                             crop_to_patches=False, max_patches=1)
    transformers.AyaVisionProcessor(image_processor=ip, tokenizer=tok, patch_size=16, img_size=64,
                                    downsample_factor=2, chat_template=CHAT_TEMPLATE).save_pretrained(d)
    return d, tok


def _run(model_dir, samples):
    m = HFGenModel(model=str(model_dir), device="cpu")
    m._ensure_pipeline()
    seen = _spy_generate(m._pipeline.model)
    res = ak.evaluate(samples, m, ["exact_match"], config=ak.RunConfig(max_tokens=4))
    return res, seen


def test_cohere_chat_prompt_gets_one_bos(cohere_dir):
    d, tok = cohere_dir
    _, seen = _run(d, [Sample(input="what is two plus two", target="a")])
    (ids, _), = seen
    assert ids[0] == tok.bos_token_id
    assert ids.count(tok.bos_token_id) == 1


def test_base_model_still_gets_its_bos(cohere_dir, tmp_path):
    """No chat template -> raw prompt, and the tokenizer's BOS is still added."""
    d, tok = cohere_dir
    base = tmp_path / "base"
    transformers.CohereForCausalLM.from_pretrained(d).save_pretrained(base)
    tok_base = _tokenizer()
    tok_base.chat_template = None
    tok_base.save_pretrained(base)
    _, seen = _run(base, [Sample(input="what is two plus two", target="a")])
    (ids, _), = seen
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1


def test_aya_vision_routes_images_through_processor(aya_vision_dir):
    from PIL import Image
    d, tok = aya_vision_dir
    img = Image.new("RGB", (64, 64), color=(200, 30, 30))
    samples = [Sample(input="describe this image", target="a", images=[img]),
               Sample(input="what is the capital of france", target="a")]
    res, seen = _run(d, samples)
    assert res.failed_count == 0 and len(res.predictions) == 2
    assert all(isinstance(p.raw_output, str) for p in res.predictions)
    with_image = [ids for ids, has_px in seen if has_px]
    without = [ids for ids, has_px in seen if not has_px]
    assert len(with_image) == 1 and len(without) == 1
    image_id = tok.convert_tokens_to_ids("<|IMG_PATCH|>")  # what <image> expands to
    assert image_id in with_image[0] and image_id not in without[0]
    for ids in with_image + without:
        assert ids.count(tok.bos_token_id) == 1


def test_images_on_text_only_model_raise(cohere_dir):
    from PIL import Image
    from auditkit.model import Request
    d, _ = cohere_dir
    m = HFGenModel(model=str(d), device="cpu")
    with pytest.raises(ValueError, match="not a vision model"):
        m.generate([Request(prompt="describe this image", params={"images": [Image.new("RGB", (8, 8))]})])
