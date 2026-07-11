"""Fused RMSNorm and SiLU gate for Qwen DeltaNet outputs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_rmsnorm(
    hidden_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    hidden_stride,
    gate_stride,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    hidden = tl.load(
        hidden_ptr + row * hidden_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * gate_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    variance = tl.sum(hidden * hidden, axis=0) / N
    normed = (hidden * tl.rsqrt(variance + eps)).to(
        hidden_ptr.dtype.element_ty
    )
    weighted = (normed * weight).to(hidden_ptr.dtype.element_ty)
    output = weighted.to(tl.float32) * gate * tl.sigmoid(gate)
    tl.store(
        output_ptr + row * N + offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def gated_rmsnorm(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm before a SiLU gate, matching Qwen3.5 ordering."""
    if hidden_states.shape != gate.shape:
        raise ValueError("hidden states and gate must have the same shape")
    if hidden_states.shape[-1] != weight.numel():
        raise ValueError("weight size does not match hidden width")
    if hidden_states.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("gated RMSNorm requires unit-stride rows")

    width = hidden_states.shape[-1]
    rows = hidden_states.numel() // width
    hidden_2d = hidden_states.reshape(rows, width)
    gate_2d = gate.reshape(rows, width)
    output = torch.empty_like(hidden_2d)
    _gated_rmsnorm[(rows,)](
        hidden_2d,
        gate_2d,
        weight,
        output,
        hidden_2d.stride(0),
        gate_2d.stride(0),
        N=width,
        BLOCK_N=triton.next_power_of_2(width),
        eps=eps,
        num_warps=4,
    )
    return output.reshape_as(hidden_states)
