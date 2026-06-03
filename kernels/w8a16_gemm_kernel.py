"""
kernels/w8a16_gemm_kernel.py — weight-only int8 GEMM (W8A16).

What this is
------------
Decode is weight-bandwidth bound: every token streams ~16 GB of bf16 weights
through HBM (aten::mm ~76% of decode time, ~flat across batch). The single
biggest remaining lever is therefore to move FEWER weight bytes — not fewer
launches. This kernel stores the linear weights as int8 (1 byte) instead of
bf16 (2 bytes) and dequantizes them inside the matmul, halving the dominant
weight stream.

Numerics (weight-only, W8A16)
-----------------------------
Activations stay bf16 (A16). Only the weights are quantized:

    w_fp  ≈ w_int8 * scale[out_channel]          # per-output-channel symmetric

The GEMM computed is the same as F.linear, i.e. y = x @ w.T with
w of shape (N, K) (out_features, in_features):

    y[m, n] = scale[n] * sum_k  x[m, k] * w_int8[n, k]

We accumulate x * w_int8 in fp32 (x promoted to fp32, w_int8 promoted to fp32)
and apply the per-output-channel scale once at the end, then round to the
output dtype. Per-output-channel symmetric quantization keeps the relative
error small (each output neuron gets its own scale), which is what preserves
coherence through 32 layers.

Why a custom kernel (cuBLAS can't help here)
--------------------------------------------
There is no cuBLAS path that consumes an int8 weight + bf16 activation and
dequantizes on the fly. The bf16 GEMM cuBLAS uses would first need the weight
expanded back to bf16 (defeating the bandwidth saving). For the skinny decode
shapes (M = B*T small, K/N large) the GEMM is memory-bound on the weight read,
so reading 1 byte/weight instead of 2 is a direct win even though Triton's raw
GEMM efficiency is below cuBLAS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_gemm(
    x_ptr,        # (M, K)  bf16 activations
    w_ptr,        # (N, K)  int8 weights  (row n = out channel n)
    s_ptr,        # (N,)    per-output-channel fp32 scale
    y_ptr,        # (M, N)  output
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped/swizzled program ordering for L2 reuse (standard Triton matmul).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k[None, :] < k_remaining
        x = tl.load(x_ptrs, mask=k_mask, other=0.0)          # (BM, BK) bf16
        w = tl.load(w_ptrs, mask=k_mask, other=0)            # (BN, BK) int8
        # int8 values (|v|<=127) are exact in bf16; cast and use tensor-core
        # tl.dot (bf16 inputs, fp32 accumulate) — same math as the bf16 GEMM
        # but the weight was streamed as 1 byte instead of 2.
        acc += tl.dot(x, w.to(x.dtype).T)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Apply per-output-channel scale once, after the int accumulation.
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale = tl.load(s_ptr + offs_yn, mask=offs_yn < N, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 64-bit row offset: for large prefill M the product offs_ym*stride_ym (= M*N)
    # can exceed int32 range (e.g. M~1e5, N=28672 -> ~3e9 > 2^31), which would wrap
    # to a negative address and trigger an illegal memory access. Promote to int64.
    y_ptrs = (y_ptr + offs_ym[:, None].to(tl.int64) * stride_ym
              + offs_yn[None, :].to(tl.int64) * stride_yn)
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def quantize_int8_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-output-channel int8 quantization of a linear weight.

    Args:
        w: (out_features, in_features) float weight (the F.linear weight).
    Returns:
        w_int8: (out_features, in_features) int8
        scale:  (out_features,) fp32, such that w ≈ w_int8 * scale[:, None]
    """
    w_f = w.detach().to(torch.float32)
    amax = w_f.abs().amax(dim=1).clamp_min(1e-8)      # (out_features,)
    scale = amax / 127.0
    w_int8 = torch.round(w_f / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return w_int8.contiguous(), scale.contiguous()


def w8a16_linear_triton(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Weight-only int8 linear: y = (x @ w_int8.T) * scale, matching F.linear(x, w).

    Args:
        x:      (..., K) bf16/fp16 activations.
        w_int8: (N, K) int8 weights (out_features, in_features).
        scale:  (N,) per-output-channel fp32 scale.
    Returns:
        (..., N) tensor in x.dtype.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K)
    if x2d.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x2d = x2d.to(torch.bfloat16)
    M = x2d.shape[0]
    N = w_int8.shape[0]
    assert w_int8.shape[1] == K, f"K mismatch: x K={K}, w K={w_int8.shape[1]}"

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Fixed launch config (no autotune): autotune's do_bench is unsafe to run for
    # arbitrary M (it can hit a fragile config and it allocates+syncs, which is
    # illegal under CUDA-graph capture). The MLP weight shapes are fixed, K and N
    # are always multiples of the block sizes below, and M is the only varying
    # dim — so per-M-bucket hardcoded tensor-core tiles are both safe and fast.
    # Tiles were found offline (workspace/tune_w8a16.py) as the fastest safe
    # config for the two MLP shapes (gate_up N=28672/K=4096, down N=4096/K=14336)
    # at the representative M of each bucket: M=1 decode (b1), M=128 decode (b128),
    # M=2048 prefill. The kernel masks offs_m by `% M`, so a BLOCK_M larger than M
    # is safe.
    if M <= 16:
        # b1 decode: tuner best gate_up (16,128,128,4,8), down (16,64,128,4,8).
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 128, 128
        num_stages, num_warps = 4, 8
    elif M <= 256:
        # b128 decode: the two MLP shapes want different BLOCK_M, so select by N.
        # tuner best: gate_up (N=28672) -> (128,128,128,2,4); down (N=4096) -> (32,128,128,3,8).
        if N >= 8192:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
            num_stages, num_warps = 2, 4
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 128, 128
            num_stages, num_warps = 3, 8
    else:
        # prefill (large M): tuner best gate_up/down (128,256,64,*,8).
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        num_stages, num_warps = 4, 8
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a16_gemm[grid](
        x2d, w_int8, scale, y,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N)
