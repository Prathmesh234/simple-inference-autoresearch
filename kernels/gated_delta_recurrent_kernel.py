"""Fused single-token recurrent Gated DeltaNet update."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_delta_recurrent(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_gb,
    stride_gh,
    stride_betab,
    stride_betah,
    stride_state_b,
    stride_state_h,
    stride_state_k,
    stride_state_v,
    stride_out_b,
    stride_out_h,
    stride_out_v,
    H,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    v_block = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_k = tl.arange(0, K)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_k * stride_qd).to(
        tl.float32
    )
    k = tl.load(k_ptr + b * stride_kb + h * stride_kh + offs_k * stride_kd).to(
        tl.float32
    )
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * (K**-0.5)
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)

    state_ptrs = (
        state_ptr
        + b * stride_state_b
        + h * stride_state_h
        + offs_k[:, None] * stride_state_k
        + offs_v[None, :] * stride_state_v
    )
    state = tl.load(state_ptrs, mask=mask_v[None, :], other=0.0).to(tl.float32)
    decay = tl.exp(
        tl.load(g_ptr + b * stride_gb + h * stride_gh).to(tl.float32)
    )
    state = state * decay

    value = tl.load(
        v_ptr + b * stride_vb + h * stride_vh + offs_v * stride_vd,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + b * stride_betab + h * stride_betah
    ).to(tl.float32)
    memory = tl.sum(state * k[:, None], axis=0)
    delta = (value - memory) * beta
    state = state + k[:, None] * delta[None, :]
    tl.store(state_ptrs, state, mask=mask_v[None, :])

    output = tl.sum(state * q[:, None], axis=0)
    tl.store(
        out_ptr + b * stride_out_b + h * stride_out_h + offs_v * stride_out_v,
        output,
        mask=mask_v,
    )


def gated_delta_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Update recurrent state and emit one token.

    Shapes match Transformers/FLA: q/k/v are (B, 1, H, D), g/beta are
    (B, 1, H), and state is (B, H, K, V) float32.
    """
    if not use_qk_l2norm_in_kernel:
        raise ValueError("fused recurrence requires in-kernel Q/K L2 normalization")
    B, T, H, K = query.shape
    V = value.shape[-1]
    if T != 1 or key.shape != query.shape:
        raise ValueError("fused recurrence supports single-token matching Q/K")
    if initial_state.shape != (B, H, K, V):
        raise ValueError(
            f"state shape {tuple(initial_state.shape)} != {(B, H, K, V)}"
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    output = torch.empty((B, 1, H, V), dtype=query.dtype, device=query.device)

    block_v = 8
    _gated_delta_recurrent[(B * H, triton.cdiv(V, block_v))](
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output,
        query.stride(0),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(2),
        value.stride(3),
        g.stride(0),
        g.stride(2),
        beta.stride(0),
        beta.stride(2),
        initial_state.stride(0),
        initial_state.stride(1),
        initial_state.stride(2),
        initial_state.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(3),
        H,
        K=K,
        V=V,
        BLOCK_V=block_v,
        num_warps=4,
    )
    return output, initial_state if output_final_state else None
