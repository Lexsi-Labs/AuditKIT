"""HuggingFace generation backend (optional extra: auditkit[transformers])."""
from __future__ import annotations

from typing import Any

from . import (
    Model, Request, Result_, Generated, LogLikelihood, resolve_params,
    DEFAULT_TEMPERATURE, reject_generation_kwargs, _free_torch_memory,
)
from ..errors import ExtraNotInstalled
from ..types import Capability

# stop_sequences maps to generate()'s real stop_strings kwarg (needs the
# tokenizer passed alongside it -- handled in generate() below, not here).
# seed has no generate()-kwarg equivalent at all; transformers' own
# set_seed() (a global RNG reset) is the only per-call reproducibility knob,
# so it's applied separately in generate() rather than through this map.
_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_new_tokens",
    "stop_sequences": "stop_strings",
}

# Model-LOADING kwargs `transformers.pipeline()` accepts at construction
# time (forwarded internally to from_pretrained) -- these must reach
# pipeline() itself in _ensure_pipeline(), not the per-call generate()
# invocation. The pipeline's __call__ doesn't accept any of these at all;
# passing e.g. torch_dtype= here previously raised TypeError at generate()
# time since it was blindly spread into **self._extra_kwargs on the call,
# never on construction. Anything in **kwargs NOT in this set is assumed
# to be a real generation-time kwarg (repetition_penalty, num_beams,
# no_repeat_ngram_size, ...) and keeps going to the pipeline call, as
# before -- this fix only redirects the loading-specific ones.
_LOAD_TIME_KWARGS = frozenset({
    "torch_dtype", "dtype", "device_map", "trust_remote_code", "revision",
    "low_cpu_mem_usage", "quantization_config", "model_kwargs", "use_fast",
    "attn_implementation", "load_in_8bit", "load_in_4bit",
})


class HFGenModel(Model):
    """Generate text using a HuggingFace transformers model."""

    name = "hf"
    is_local = True

    def __init__(self, model: str = "gpt2", device: str | None = "auto",
                 name: str = "hf", hf_token: str | None = None,
                 token: str | None = None, **kwargs: Any) -> None:
        reject_generation_kwargs(kwargs, "HFGenModel")
        self._model_name = model
        self._device = device
        # Gated models (Llama, Gemma, ...) need a HuggingFace access token at
        # load time; accept `hf_token` or `token` and pass it to from_pretrained.
        self._token = hf_token or token
        self._extra_kwargs = kwargs
        self._pipeline = None
        self.name = name

    def capabilities(self) -> set[Capability]:
        return {Capability.GENERATE, Capability.LOGLIKELIHOOD}

    def model_info(self) -> dict[str, Any]:
        """Real parameter count/sparsity/on-disk-equivalent size, introspected
        from the loaded checkpoint -- never a caller-supplied guess.

        Loads the pipeline first if it isn't already (so calling this before
        any ``generate()``/``loglikelihood()`` call still works, e.g. from a
        comparison that wants size info without necessarily having run yet).
        ``size_mb`` is each parameter's actual in-memory dtype size summed up
        (``numel() * element_size()``) -- the same bytes an on-disk safetensors
        checkpoint stores, not a fixed fp32/fp16 assumption.
        """
        self._ensure_pipeline()
        model = self._pipeline.model
        params = list(model.parameters())
        total_params = sum(p.numel() for p in params)
        # nonzero count is what actually shows pruning -- a structurally
        # pruned model still allocates the same tensor shapes, just with
        # zeroed-out weights, so total_params alone can't tell two
        # differently-pruned checkpoints apart.
        nonzero_params = sum(int((p != 0).sum().item()) for p in params)
        size_bytes = sum(p.numel() * p.element_size() for p in params)
        return {
            "is_local": True,
            "model_name": self._model_name,
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "sparsity": 1.0 - (nonzero_params / total_params) if total_params else 0.0,
            "size_mb": size_bytes / (1024 * 1024),
        }

    def unload(self) -> None:
        """Drop the loaded transformers pipeline (model + tokenizer) and free
        the GPU cache -- so ``compare_models()`` can reclaim this checkpoint's
        memory before loading the next. Lazily reloads on the next call."""
        self._pipeline = None
        _free_torch_memory()

    def _resolve_device(self) -> str:
        """Best available device: CUDA GPU > Apple Silicon MPS > CPU.

        transformers.pipeline(device=...) forwards straight to
        torch.device(...), which has no concept of "auto" -- it only
        understands concrete device-type strings ("cpu", "cuda", "mps",
        ...). "auto" is a real HF concept, but it belongs to a different
        parameter (device_map="auto" on from_pretrained's multi-GPU
        placement), not this one. Previously "auto" was forwarded verbatim
        and crashed immediately on every zero-config construction. An
        explicit device (e.g. "cpu") always wins; detection only runs when
        the caller left it at the default or passed None.
        """
        if self._device not in (None, "auto"):
            return self._device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _logits_processor(self):
        """A processor that sanitizes non-finite logits before sampling,
        WITHOUT touching genuinely finite ones.

        Some checkpoints (confirmed: gpt2-medium, on this environment) emit
        nan/inf logits straight out of the forward pass for certain prompts --
        a numerical-precision issue in the model computation itself, not
        something caused by our decoding settings. torch.multinomial (real
        sampling) rejects a distribution containing any nan/inf/negative
        value outright, crashing the whole run.

        A previous version of this fix blanket-clamped every logit
        (finite or not) to a fixed [-50, 50] range. That's a real,
        confirmed bug for checkpoints whose legitimate logit scale exceeds
        that range: bigscience/bloom-560m's real, non-NaN logits for a
        typical prompt reach ~430 for the correct top token, and clamping
        collapses every one of its top ~30 tokens down to the SAME ceiling
        value of 50.0 -- creating an artificial tie that argmax/greedy then
        breaks by lowest token ID, which favors a low-ID special token
        (confirmed: the real EOS token, id 2, won the tie over the actual
        best token " Paris"), producing empty generated text instead of a
        real answer. Only the positions that are ACTUALLY non-finite are
        replaced now; every genuinely finite logit -- regardless of
        magnitude -- passes through completely untouched, so a model's
        real, correctly-ranked distribution is never altered unless it
        would otherwise crash generation outright.
        """
        import torch
        from transformers import LogitsProcessor

        class _SanitizeLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                non_finite = ~torch.isfinite(scores)
                if not non_finite.any():
                    return scores
                scores = scores.clone()
                # A fixed, safe replacement for whatever was actually
                # broken -- not a clamp applied to the whole tensor.
                scores[non_finite & (scores == float("inf"))] = 1e4
                scores[non_finite & (scores == float("-inf"))] = -1e4
                scores[non_finite & torch.isnan(scores)] = -1e4
                return scores

        return _SanitizeLogits()

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch  # noqa: F401 -- forces a clear ExtraNotInstalled if torch is missing
            from transformers import pipeline
        except ImportError:
            raise ExtraNotInstalled("transformers", "pip install auditkit[transformers]")
        pipe_kwargs: dict[str, Any] = {}
        if self._token:
            pipe_kwargs["token"] = self._token
        for k in _LOAD_TIME_KWARGS:
            if k in self._extra_kwargs:
                pipe_kwargs[k] = self._extra_kwargs[k]
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_name,
            device=self._resolve_device(),
            **pipe_kwargs,
        )
        # generate() always drives length via max_new_tokens (see _KEY_MAP) --
        # never max_length. Most checkpoints still ship a generation_config.json
        # with a default max_length (commonly 20), which transformers otherwise
        # warns about on every single call ("Both max_new_tokens and max_length
        # seem to have been set ... max_new_tokens will take precedence").
        # Harmless in practice (the warning itself confirms max_new_tokens wins),
        # but noisy on every generate() call across an entire run.
        #
        # Pipeline.__init__ deep-copies model.generation_config into its OWN
        # self.generation_config attribute at construction time (see
        # transformers/pipelines/base.py), and TextGenerationPipeline reads
        # that copy, not the model's live one -- so clearing only
        # self._pipeline.model.generation_config.max_length here has no
        # effect on the warning; the pipeline's own already-copied
        # generation_config has to be cleared too.
        #
        # Guarded with getattr: not every transformers version/model exposes a
        # `generation_config` on the pipeline (older TextGenerationPipeline
        # lacks the attribute) or a non-None one on the model. Clearing it is
        # only warning-suppression, so a missing attribute must never crash
        # pipeline construction (it previously raised AttributeError on, e.g.,
        # sshleifer/tiny-gpt2, taking down generate()/loglikelihood() entirely).
        for _holder in (self._pipeline.model, self._pipeline):
            _gen_cfg = getattr(_holder, "generation_config", None)
            if _gen_cfg is not None:
                _gen_cfg.max_length = None

    def _render_prompt(self, request: Request) -> str:
        """Render *request* to the literal text sent to the model.

        The backend owns the model-specific wire format, so *any* adapter's
        output is formatted correctly for the model at hand — routing decides
        the technique (MCQ/RAG/generation), never the chat-vs-base formatting:

        * **Instruct/chat model** (tokenizer has a ``chat_template``): render
          through the model's *own* template with its real special tokens
          (``<|im_start|>``, ``[INST]``, ...). Structured turns
          (``request.params["messages"]``, e.g. from ``ChatAdapter``) render
          as-is; a flat prompt from any other adapter is wrapped as a single
          ``user`` turn so it still gets the model's real formatting instead
          of raw text it was never trained on.
        * **Base model** (no ``chat_template``): the plain prompt text.
        """
        text = request.prompt if isinstance(request.prompt, str) else str(request.prompt)
        # Explicit opt-out: send the prompt verbatim, skipping the chat template.
        # Classifier-style models (e.g. some safety guards like WildGuard /
        # HarmBench) are trained on their own fixed prompt string; wrapping that
        # in the tokenizer's chat template would corrupt it.
        if request.params.get("apply_chat_template") is False:
            return text
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        if not chat_template:
            return text
        messages = request.params.get("messages") or [{"role": "user", "content": text}]
        # Extra kwargs some chat templates need (e.g. a safety guard's per-call
        # policy: ShieldGemma's `guideline=`, Granite Guardian's `guardian_config=`).
        extra = request.params.get("chat_template_kwargs") or {}
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_pipeline()
        prompts = [self._render_prompt(r) for r in requests]
        # The pipeline runs all prompts in one batched call sharing one
        # sampling config, so per-request param variation isn't meaningful
        # here -- the first request's params represent the whole batch
        # (in practice uniform, since one evaluate() run shares one RunConfig).
        defaults = {"temperature": DEFAULT_TEMPERATURE, "max_new_tokens": 128}
        gen_kwargs = resolve_params(requests[0], defaults, _KEY_MAP) if requests else defaults
        # Never hand the pipeline an ambiguous "sample, but with no temperature"
        # combination -- some checkpoints' own generation_config defaults to
        # do_sample=True, and sampling with temperature=None can produce an
        # invalid (nan/inf) probability tensor for some models (confirmed:
        # gpt2-medium crashes this way, plain gpt2 doesn't). Explicitly choose
        # greedy vs. sampling instead of leaving it to be inferred.
        temp = gen_kwargs.pop("temperature")
        if temp is not None and temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False
        # generate()'s stop_strings criterion needs the tokenizer passed
        # alongside it to check decoded output against -- the pipeline
        # doesn't supply this on its own.
        if "stop_strings" in gen_kwargs:
            gen_kwargs["tokenizer"] = self._pipeline.tokenizer
        # No per-call seed kwarg exists for generate()/pipeline() -- transformers'
        # own set_seed() resets the global torch/numpy/python RNGs, the only
        # reproducibility knob available. A real, if blunt, effect: same seed
        # + same prompt + do_sample=True reliably reproduces the same output.
        seed = requests[0].params.get("seed") if requests else None
        if seed is not None:
            from transformers import set_seed
            set_seed(seed)
        call_time_extra_kwargs = {
            k: v for k, v in self._extra_kwargs.items() if k not in _LOAD_TIME_KWARGS
        }
        outputs = self._pipeline(
            prompts,
            # transformers' text-generation pipeline defaults to echoing the
            # full prompt back as part of "generated_text" -- every other
            # backend (openai.py, anthropic.py, vllm_gen.py, ...) returns only
            # the new completion, never the input. That inconsistency broke
            # LLMJudge: its instructions necessarily list every valid verdict
            # label, so with the prompt echoed back, the parser reliably
            # mistook its own instructions for the model's real answer.
            return_full_text=False,
            logits_processor=[self._logits_processor()],
            **gen_kwargs,
            **call_time_extra_kwargs,
        )
        results = []
        for out in outputs:
            if isinstance(out, list):
                text = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            elif isinstance(out, dict):
                text = out.get("generated_text", "")
            else:
                text = str(out)
            results.append(Result_(completions=[Generated(text=text)]))
        return results

    def loglikelihood(self, requests: list[Request]) -> list[LogLikelihood]:
        """Real per-candidate log-probability scoring via teacher forcing.

        For each request, feeds ``prompt + target`` through the model with
        the target's own tokens forced as the continuation (never sampled
        or generated), and sums the log-probability the model assigns to
        each of the target's actual tokens, conditioned on everything
        before it. This is the standard MCQ-by-loglikelihood technique
        (the same approach lm-eval-harness uses for ``hf:`` models) -- it
        measures how likely the model considers a *specific* candidate
        answer, without requiring it to generate that exact text itself.

        Reuses the same lazily-loaded pipeline's model/tokenizer as
        ``generate()`` (``transformers.pipeline`` exposes both as plain
        attributes) -- no separate model load. One forward pass per
        request, not yet batched across requests (a real cost -- see the
        design note this followed up on -- but batching ragged
        prompt/continuation lengths correctly, with proper padding and
        attention masks, is real additional surface area for a subtle
        numerical bug; left as a follow-up once this is verified correct).
        """
        self._ensure_pipeline()
        import torch

        model = self._pipeline.model
        tokenizer = self._pipeline.tokenizer
        device = self._pipeline.device

        results: list[LogLikelihood] = []
        with torch.no_grad():
            for r in requests:
                prompt = r.prompt if isinstance(r.prompt, str) else str(r.prompt)
                target = r.params.get("target", "")

                # Tokenize context alone and context+target together (not
                # context and target separately) so continuation_ids reflects
                # how the tokenizer actually splits the continuation as part
                # of one contiguous string -- BPE merges can span the
                # context/continuation boundary, so naive separate
                # tokenization can silently produce different token ids
                # than what the model will actually see.
                context_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
                continuation_ids = full_ids[len(context_ids):]

                if not continuation_ids:
                    # Empty target (or one that tokenizes to nothing new) --
                    # nothing to score.
                    results.append(LogLikelihood(logprob=0.0, is_greedy=True))
                    continue

                input_ids = torch.tensor([full_ids], device=device)
                logits = model(input_ids=input_ids).logits[0]  # [seq_len, vocab]
                # log_probs[j] is the model's distribution for the token at
                # position j+1, predicted from positions [0..j].
                log_probs = torch.log_softmax(logits[:-1], dim=-1)
                target_ids = torch.tensor(full_ids[1:], device=device)
                token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

                # The first continuation token is at full_ids[len(context_ids)],
                # predicted by log_probs[len(context_ids) - 1]. (Empty context
                # has no preceding token to condition the first continuation
                # token on -- clamped to 0 as a best-effort approximation,
                # not mathematically exact for that edge case.)
                start = max(len(context_ids) - 1, 0)
                continuation_log_probs = token_log_probs[start:]
                greedy_ids = log_probs[start:].argmax(dim=-1)
                is_greedy = bool(torch.equal(greedy_ids, target_ids[start:]))

                results.append(LogLikelihood(
                    logprob=float(continuation_log_probs.sum().item()),
                    is_greedy=is_greedy,
                ))
        return results
