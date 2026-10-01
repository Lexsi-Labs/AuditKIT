# Verbatim excerpt of sglang 0.5.20 python/sglang/srt/models/transformers.py (Apache-2.0),
# _normalize_tp_style, for auditkit.compat rule (j).
def _normalize_tp_style(style: str) -> Style:
    style = style.lower().replace("-", "_")
    style = {
        "colwiseparallel": "colwise",
        "packed_colwise": "colwise",
        "local_colwise": "colwise",
        "rowwiseparallel": "rowwise",
        "packed_rowwise": "rowwise",
        "local_rowwise": "rowwise",
        "local_packed_rowwise": "rowwise",
        "isolated": "replicate",
        "local": "replicate",
        "replicated_with_grad_allreduce": "replicate",
        "moe_tp_experts": "replicate",
    }.get(style, style)
    if style not in {"colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"}:
        raise ValueError(f"Unsupported TP style '{style}' for Transformers backend.")
    return style


# ... end of TransformersBase (load_weights) and the next class, verbatim, for rule (m)
class TransformersBase:
    # -- Weight loading -----------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=self.skip_prefixes,
            skip_substrs=self.skip_substrs,
            ignore_unexpected_prefixes=self.ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=self.ignore_unexpected_suffixes,
        )
        return loader.load_weights(weights, mapper=self.weight_mapper)


class CausalMixin:
    def __init__(self, *args, prefix: str = "", **kwargs):
        pass


# can_enable_torch_compile, verbatim, for rule (n)
def can_enable_torch_compile(config: PretrainedConfig) -> bool:
    """Check whether the model config is compatible with torch.compile.

    Dynamic rope scaling triggers data-dependent control flow that prevents
    capturing a single computation graph, so we disable compilation for it.
    """
    text_config = getattr(config, "text_config", config)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type == "dynamic":
            return False
    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        if isinstance(next(iter(rope_params.values()), None), dict):
            return not any(
                rp.get("rope_type") == "dynamic" for rp in rope_params.values()
            )
        if rope_params.get("rope_type") == "dynamic":
            return False
    return True
