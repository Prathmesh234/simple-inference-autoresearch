"""
kernels/add_rmsnorm_kernel.py — fused residual-add + RMSNorm.

Why fuse the residual add into the norm?
----------------------------------------
Every Llama block does, twice:

    x = x + sublayer(norm(x))      # an explicit elementwise add (aten::add)
    ...
    normed = norm(x)               # then the next norm reads x again

The `x = x + h` add is its own kernel launch and its own pair of HBM
round-trips (read x, read h, write x), and then the *very next* thing the
model does is an RMSNorm that reads that same x straight back out of HBM.

This kernel folds the add into the norm: it reads `residual` and `hidden`
once, computes `new_residual = residual + hidden`, and in the SAME pass
normalises it — writing both `new_residual` (needed for the next residual
connection) and `normed` (the norm output). That removes one kernel launch
per residual connection (64 per decode step across 32 layers × 2) and a
redundant HBM read of the residual stream. RMSNorm is purely memory-bound,
so cutting launches + traffic is a direct decode win — exactly the
"add_rmsnorm" fusion vLLM/TensorRT-LLM ship.

Numerics
--------
`new_residual = (residual + hidden)` is accumulated in float32 and rounded
once to the storage dtype, so it matches torch's bf16 `x = x + h` (which also
accumulates in f32 and rounds to nearest-even) bit-for-bit. The normalisation
statistics are accumulated in float32, identical math to the standalone
RMSNorm kernel and the PyTorch reference. The normed output is numerically
equivalent to "add then rmsnorm" up to float32 reduction-order noise (both
this kernel and rmsnorm_triton reduce under @triton.autotune, which selects
num_warps by timing, so the last bits of the sum-of-squares can differ); it is
not required to be bit-identical to rmsnorm_triton. The residual *stream* the
model threads through its layers is bit-exact, because that is carried by
`new_residual`, not by the normed output.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _add_rmsnorm_fwd(
    h_ptr,            # input  hidden    (n_rows, N)
    r_ptr,            # input  residual  (n_rows, N)
    w_ptr,            # weight (N,)
    out_ptr,          # output normed        (n_rows, N) contiguous
    new_r_ptr,        # output new residual  (n_rows, N) contiguous
    stride_h,         # row stride of hidden
    stride_r,         # row stride of residual
    stride_o,         # row stride of the contiguous outputs (== N)
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row: new_r = h + r; out = rmsnorm(new_r) * w."""
    pid = tl.program_id(0)
    h_row     = h_ptr + pid * stride_h
    r_row     = r_ptr + pid * stride_r
    out_row   = out_ptr + pid * stride_o
    new_r_row = new_r_ptr + pid * stride_o

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Residual add in float32 then rounded once to the storage dtype — this
    # matches torch's bf16 `x + h` (which also accumulates in f32 and rounds to
    # nearest-even) bit-for-bit, avoiding hardware bf16-add ULP drift that would
    # otherwise compound across the 32 layers.
    h = tl.load(h_row + cols, mask=mask).to(tl.float32)
    r = tl.load(r_row + cols, mask=mask).to(tl.float32)
    new_r = (h + r).to(out_ptr.dtype.element_ty)
    tl.store(new_r_row + cols, new_r, mask=mask)

    # RMSNorm over the freshly-added residual, in float32. Normalising the
    # *stored* (rounded) residual makes this bit-identical to the unfused
    # "bf16 add, then rmsnorm" path the engine runs today.
    xf = new_r.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    rms = tl.rsqrt(var + eps)
    out = xf * w * rms
    tl.store(out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def add_rmsnorm_triton(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual-add + RMSNorm.

    Computes:
        new_residual = residual + hidden
        normed       = rmsnorm(new_residual) * weight

    Args:
        hidden:   (..., hidden_size) — the sublayer output to add in
        residual: (..., hidden_size) — the running residual stream
        weight:   (hidden_size,)
        eps:      RMSNorm epsilon

    Returns:
        (normed, new_residual) — both same shape/dtype as hidden.
    """
    assert hidden.shape == residual.shape
    orig_shape = hidden.shape
    h_2d = hidden.reshape(-1, hidden.shape[-1])
    r_2d = residual.reshape(-1, residual.shape[-1])
    n_rows, N = h_2d.shape

    # The two passes assume a row fits one tile (true for H=4096). The standalone
    # rmsnorm kernel makes the same assumption.
    BLOCK_SIZE = triton.next_power_of_2(N)

    out = torch.empty_like(h_2d)
    new_r = torch.empty_like(h_2d)

    grid = (n_rows,)
    _add_rmsnorm_fwd[grid](
        h_2d, r_2d, weight, out, new_r,
        h_2d.stride(0), r_2d.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape), new_r.reshape(orig_shape)


# ── EXP-D: add_rmsnorm fused with per-token int8 activation quant ─────────────
# The MLP pre-norm (post_attention_layernorm) output feeds ONLY the gate_up
# projection, which runs W8A8 (int8 activation x int8 weight) at the b128 decode
# bucket. This kernel folds that per-token int8 quantization INTO the add_rmsnorm
# pass: the normalized row is already live in registers, so computing its row-max
# and emitting (int8, scale) is nearly free and removes the standalone
# _quant_per_token launch (~18-29us/layer) entirely. It still emits the bf16
# `normed` so the b1 (M<=16) and prefill (M>256) W8A16 fallbacks keep working.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _add_rmsnorm_quant_fwd(
    h_ptr, r_ptr, w_ptr,
    out_ptr,          # bf16 normed       (n_rows, N)
    new_r_ptr,        # bf16 new residual (n_rows, N)
    qi_ptr,           # int8 normed       (n_rows, N)
    qs_ptr,           # fp32 per-row scale (n_rows,)
    stride_h, stride_r, stride_o,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    h_row     = h_ptr + pid * stride_h
    r_row     = r_ptr + pid * stride_r
    out_row   = out_ptr + pid * stride_o
    new_r_row = new_r_ptr + pid * stride_o

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    h = tl.load(h_row + cols, mask=mask).to(tl.float32)
    r = tl.load(r_row + cols, mask=mask).to(tl.float32)
    new_r = (h + r).to(out_ptr.dtype.element_ty)
    tl.store(new_r_row + cols, new_r, mask=mask)

    xf = new_r.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    rms = tl.rsqrt(var + eps)
    out = xf * w * rms
    tl.store(out_row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

    # Fused per-token symmetric int8 quant of the normed row (same math as
    # kernels/w8a8_gemm_kernel._quant_per_token, but the data is already here).
    amax = tl.maximum(tl.max(tl.where(mask, tl.abs(out), 0.0)), 1e-8)
    scale = amax / 127.0
    qi = tl.extra.cuda.libdevice.round(out / scale)
    qi = tl.minimum(tl.maximum(qi, -127.0), 127.0).to(tl.int8)
    tl.store(qi_ptr + pid * N + cols, qi, mask=mask)
    tl.store(qs_ptr + pid, scale)


def add_rmsnorm_quant_triton(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
):
    """add_rmsnorm that ALSO returns a per-token int8 quantization of the normed
    output for the W8A8 gate_up GEMM.

    Returns:
        (normed_bf16, new_residual, normed_int8, act_scale)
        - normed_bf16:  (..., N)        for the W8A16 fallback (b1 / prefill)
        - new_residual: (..., N)        the threaded residual stream
        - normed_int8:  (n_rows, N) int8 ready for _w8a8_gemm
        - act_scale:    (n_rows,)  fp32 per-row scale
    """
    assert hidden.shape == residual.shape
    orig_shape = hidden.shape
    h_2d = hidden.reshape(-1, hidden.shape[-1])
    r_2d = residual.reshape(-1, residual.shape[-1])
    n_rows, N = h_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    out = torch.empty_like(h_2d)
    new_r = torch.empty_like(h_2d)
    qi = torch.empty((n_rows, N), dtype=torch.int8, device=h_2d.device)
    qs = torch.empty((n_rows,), dtype=torch.float32, device=h_2d.device)

    grid = (n_rows,)
    _add_rmsnorm_quant_fwd[grid](
        h_2d, r_2d, weight, out, new_r, qi, qs,
        h_2d.stride(0), r_2d.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape), new_r.reshape(orig_shape), qi, qs
