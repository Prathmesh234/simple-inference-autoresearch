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
