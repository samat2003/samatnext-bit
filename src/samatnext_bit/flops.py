from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlopEstimate:
    total_params: int
    active_params: int
    active_layers: int
    forward_flops_per_token: int
    backward_update_flops_per_token: float
    total_training_flops_per_token: float
    formula: str


def estimate_decoder_params(
    vocab_size: int,
    seq_len: int,
    hidden: int,
    layers: int,
) -> int:
    """Estimate DecoderLM parameters from architecture dimensions."""
    embeddings = vocab_size * hidden + seq_len * hidden
    norm = hidden
    lm_head = hidden * vocab_size + vocab_size
    per_block = (
        2 * hidden
        + (hidden * (hidden * 3) + hidden * 3)
        + (hidden * hidden + hidden)
        + (hidden * (hidden * 4) + hidden * 4)
        + ((hidden * 4) * hidden + hidden)
    )
    return int(embeddings + norm + lm_head + layers * per_block)


def estimate_training_flops(
    *,
    training_rule: str,
    total_params: int,
    active_params: int,
    active_layers: int,
    update_every: int,
) -> FlopEstimate:
    """Simple transparent FLOP/token estimate.

    This intentionally approximates transformer training by parameter count. It
    does not try to model every attention, softmax, norm, embedding, or optimizer
    operation exactly.
    """
    if update_every < 1:
        raise ValueError("update_every must be >= 1")
    forward = 2 * active_params
    if training_rule == "chainrule":
        backward_update = 4 * active_params
        total = 6 * active_params
        formula = "chainrule: forward=2*active_params; total_training=6*active_params"
    elif training_rule == "mono":
        no_update = 2 * active_params
        update = 6 * active_params
        backward_update = update - forward
        total = no_update * ((update_every - 1) / update_every) + update * (1 / update_every)
        formula = (
            "mono: no_update=2*active_params; update_step=6*active_params; "
            "avg=no_update*(N-1)/N + update_step/N"
        )
    else:
        raise ValueError(f"unsupported training_rule={training_rule!r}")
    return FlopEstimate(
        total_params=int(total_params),
        active_params=int(active_params),
        active_layers=int(active_layers),
        forward_flops_per_token=int(forward),
        backward_update_flops_per_token=float(backward_update),
        total_training_flops_per_token=float(total),
        formula=formula,
    )
