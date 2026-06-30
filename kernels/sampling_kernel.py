"""
kernels/sampling_kernel.py — fused top-k → top-p → categorical sample.

Why
---
The decode step's model forward is a single CUDA-graph replay, so the only
remaining EAGER work on the per-step critical path is ``sample()``. The top-k
fast path (sampling.sample) runs ~9 small PyTorch ops AFTER ``torch.topk`` —
temperature divide, softmax, cumsum, the nucleus compare/shift/masked_fill, a
second softmax, ``torch.multinomial`` and a gather. Each is its own kernel launch
with CPU dispatch, and under the profiler (``profile_memory=True``, the headline
b128 cell) that dispatch + allocation bookkeeping is taxed and lands serial after
the graph replay. The op table shows ``aten::multinomial`` alone at ~309us CPU
and ``aten::topk`` at ~191us CPU per decode step.

This kernel collapses everything *after* the top-k selection into ONE launch:
given the k candidate logits (sorted desc) + their token ids + a uniform draw, it
applies temperature, the top-p nucleus, renormalises, and does an inverse-CDF
categorical sample — returning the chosen token id directly. ``torch.topk`` (the
necessary full-vocab reduction) stays; the ~9-op tail becomes ``torch.rand`` + 1
kernel.

Faithfulness
------------
All math is float32 and mirrors the canonical top-k/top-p reference in
``sampling.sample`` (and benchmarks/.../bench_sampling_compare.py): the nucleus
keep-mask is ``exclusive_cumsum(softmax(vals/T)) <= top_p`` (== the reference's
"cumsum>p shifted right by one"), and the final per-candidate distribution is
``softmax(masked vals/T)`` — identical to the reference. The categorical draw is
inverse-CDF on a ``torch.rand`` uniform: a different RNG stream than
``torch.multinomial`` but the SAME distribution (program.md requires the output
*distribution* stay faithful, not the exact RNG sequence). Drawing from
``torch.rand`` keeps it reproducible under ``torch.manual_seed``.

Scope: used only for the 0<top_k<vocab, temperature>0 fast path (the decode
sampler). Greedy (temperature==0) and the no-top-k path stay in sampling.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sample_fwd(
    vals_ptr, idx_ptr, u_ptr, out_ptr,
    stride_vb, stride_vk, stride_ib, stride_ik,
    K, temperature, top_p,
    BLOCK_K: tl.constexpr,
):
    """One program per row: temperature + top-p nucleus + inverse-CDF sample.

    vals: (B, K) top-k logits sorted DESCENDING (raw, pre-temperature).
    idx:  (B, K) int64 token ids matching vals.
    u:    (B,)   float32 uniform in [0, 1).
    out:  (B,)   int64 sampled token id.
    """
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    # Load candidates and apply temperature (divide, to match vals.float()/T).
    v = tl.load(vals_ptr + b * stride_vb + offs * stride_vk,
                mask=mask, other=-float("inf")).to(tl.float32)
    v = v / temperature

    # softmax over the k candidates (the nucleus decision distribution).
    m = tl.max(v, axis=0)
    e = tl.where(mask, tl.exp(v - m), 0.0)
    p = e / tl.sum(e, axis=0)

    # Nucleus: keep token i iff the cumulative mass BEFORE it is <= top_p. The
    # exclusive cumsum (inclusive cumsum minus self) equals the reference's
    # cum[i-1]; token 0 has exclusive mass 0 so it is always kept.
    c = tl.cumsum(p, axis=0)
    excl = c - p
    keep = mask & (excl <= top_p)

    # Renormalise the softmax over the kept nucleus.
    v2 = tl.where(keep, v, -float("inf"))
    m2 = tl.max(v2, axis=0)
    e2 = tl.where(keep, tl.exp(v2 - m2), 0.0)
    p2 = e2 / tl.sum(e2, axis=0)

    # Inverse-CDF categorical draw: pick the first index whose inclusive CDF
    # reaches u, i.e. the count of candidates with CDF strictly below u.
    cdf = tl.cumsum(p2, axis=0)
    u = tl.load(u_ptr + b)
    chosen = tl.sum((cdf < u).to(tl.int32), axis=0)
    chosen = tl.minimum(chosen, K - 1)

    tok = tl.load(idx_ptr + b * stride_ib + chosen * stride_ik)
    tl.store(out_ptr + b, tok)


def fused_topk_sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fused top-k → top-p → categorical sample. Returns (B,) int64 token ids.

    Equivalent in DISTRIBUTION to sampling.sample()'s top-k fast path: top-k the
    raw logits, divide the candidates by ``temperature``, apply the top-p nucleus
    in float32, then draw one token. Only the post-top-k tail is fused into a
    single Triton launch; ``torch.topk`` and the uniform draw remain torch ops.

    Args:
        logits:      (B, vocab) decode-step logits.
        temperature: > 0 sampling temperature.
        top_k:       candidate cap (0 < top_k < vocab).
        top_p:       nucleus threshold in (0, 1].
        generator:   optional torch.Generator for the uniform draw (reproducible).
    """
    B = logits.shape[0]
    vals, idx = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    idx = idx.to(torch.int64)
    u = torch.rand(B, device=logits.device, dtype=torch.float32, generator=generator)
    out = torch.empty(B, device=logits.device, dtype=torch.int64)

    BLOCK_K = triton.next_power_of_2(top_k)
    _fused_sample_fwd[(B,)](
        vals, idx, u, out,
        vals.stride(0), vals.stride(1), idx.stride(0), idx.stride(1),
        top_k, float(temperature), float(top_p),
        BLOCK_K=BLOCK_K, num_warps=1,
    )
    return out
