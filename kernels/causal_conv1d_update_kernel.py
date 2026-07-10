"""Fused single-token causal depthwise convolution update."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update(
    x_ptr,
    state_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    x_stride_b,
    x_stride_c,
    state_stride_b,
    state_stride_c,
    state_stride_k,
    weight_stride_c,
    weight_stride_k,
    out_stride_b,
    out_stride_c,
    channels,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // channels
    channel = row - batch * channels
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K

    state_base = state_ptr + batch * state_stride_b + channel * state_stride_c
    x = tl.load(x_ptr + batch * x_stride_b + channel * x_stride_c).to(
        tl.float32
    )
    next_offsets = offsets + 1
    next_state = tl.load(
        state_base + next_offsets * state_stride_k,
        mask=next_offsets < K,
        other=0.0,
    ).to(tl.float32)
    shifted = tl.where(next_offsets < K, next_state, x)
    weight = tl.load(
        weight_ptr
        + channel * weight_stride_c
        + offsets * weight_stride_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.sum(shifted * weight, axis=0)
    if HAS_BIAS:
        value += tl.load(bias_ptr + channel).to(tl.float32)
    value = value * tl.sigmoid(value)

    tl.store(
        state_base + offsets * state_stride_k,
        shifted.to(state_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_ptr + batch * out_stride_b + channel * out_stride_c,
        value.to(out_ptr.dtype.element_ty),
    )


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Update a causal-convolution state and return one SiLU-activated token."""
    if hidden_states.shape[-1] != 1:
        raise ValueError("causal_conv1d_update only supports single-token decode")
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation: {activation}")

    batch, channels, _ = hidden_states.shape
    state_len = conv_state.shape[-1]
    if conv_state.shape[:2] != (batch, channels):
        raise ValueError("convolution state shape does not match input")
    if weight.shape != (channels, state_len):
        raise ValueError("convolution weight shape does not match state")

    output = torch.empty_like(hidden_states)
    block_k = triton.next_power_of_2(state_len)
    bias_arg = bias if bias is not None else weight
    _causal_conv1d_update[(batch * channels,)](
        hidden_states,
        conv_state,
        weight,
        bias_arg,
        output,
        hidden_states.stride(0),
        hidden_states.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        channels,
        K=state_len,
        BLOCK_K=block_k,
        HAS_BIAS=bias is not None,
        num_warps=1,
    )
    return output
