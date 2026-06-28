"""
kernels/flash_decode_int8_kernel.py

FlashAttention decode (single query, Tq==1) reading an INT8 KV cache with a
per-(batch, kv-head, position) symmetric scale, dequantising on the fly.

Why
---
At the instruct b128 headline the decode attention reads the K/V prefix FRESH
from HBM every step and it GROWS with context: per layer the read is
2 * B * Hkv * kv_len * D * 2 bytes (bf16). For B=128, Hkv=8, D=128, kv_len~64
that is ~32 MB/layer => ~1 GB/step across 32 layers, and _flash_decode_fwd
measures ~43 us/call of which ~33 us is the KV HBM read => it is
KV-bandwidth-bound, NOT compute-bound. (The jun11 "KV-int8 dead" verdict was an
unmeasured analogy to the L2-resident *weights*; the KV cache is not a static
weight.) Storing K/V as int8 halves that read; the dequant is a single FMA per
element folded into the existing online-softmax math, so it is nearly free.

Layout
------
  k/v       : (B, Hkv, S, D)  int8   — the full preallocated cache
  k/v scale : (B, Hkv, S)     fp32   — one symmetric scale per cached vector
                                       (scale = max|vec|/127, qi = round(vec/scale))

Dequant folds cleanly into online softmax:
  qk[n] = scale_k[n] * sum_d q[d] * k_int8[n,d]          (scalar folded after dot)
  acc  += (p[n] * scale_v[n]) * v_int8[n,:]              (scalar folded into p)

Graph safety: identical structure to flash_decode_kernel — static B*Hq grid,
kv_len read from a device scalar, no autotune. Hardcoded BLOCK_N=16/nw=1/ns=2
(the tuned decode config).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# This kernel runs ONLY for the int8-KV long-context regime (kv_len > 256), so it is
# tuned for LONG kv_len, not the short bf16 headline. A wider BLOCK_N=32 (half the
# loop trips of BLOCK_N=16) wins across the long range: 1.41x at kv_len~455 (code),
# 1.19x at ~1012 (chat/long_ctx/summarize). num_stages=1 (the serial single-query
# reduction needs no deep pipeline at this size). See tune_int8_flash_long_jun28.py.
_DECODE_BLOCK_N = 32
_DECODE_NUM_WARPS = 1
_DECODE_NUM_STAGES = 1


@triton.jit
def _flash_decode_int8_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, ks_ptr, vs_ptr, kvlen_ptr, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ksb, stride_ksh, stride_ksn,
    stride_vsb, stride_vsh, stride_vsn,
    stride_ob, stride_oh, stride_od,
    Hq, KV_GROUP,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (batch, query-head). Single query row, online softmax,
    int8 K/V dequantised with a per-(b, kv-head, position) scale."""
    off_bh = tl.program_id(0)
    b = off_bh // Hq
    hq = off_bh % Hq
    hkv = hq // KV_GROUP

    kv_len = tl.load(kvlen_ptr)

    q_base = q_ptr + b * stride_qb + hq * stride_qh
    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh
    ks_base = ks_ptr + b * stride_ksb + hkv * stride_ksh
    vs_base = vs_ptr + b * stride_vsb + hkv * stride_vsh
    o_base = o_ptr + b * stride_ob + hq * stride_oh

    offs_d = tl.arange(0, D)
    q = tl.load(q_base + offs_d * stride_qd).to(tl.float32)  # (D,)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < kv_len

        k = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)  # (BLOCK_N, D)
        ks = tl.load(ks_base + offs_n * stride_ksn, mask=n_mask, other=0.0)  # (BLOCK_N,)

        # raw int8 dot, then fold the per-position key scale (a scalar per row)
        qk = tl.sum(q[None, :] * k, axis=1) * ks * sm_scale       # (BLOCK_N,)
        qk = tl.where(n_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)                                    # (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)  # (BLOCK_N, D)
        vs = tl.load(vs_base + offs_n * stride_vsn, mask=n_mask, other=0.0)  # (BLOCK_N,)
        pv = p * vs                                               # fold value scale into p
        acc = acc * alpha + tl.sum(pv[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe

    tl.store(o_base + offs_d * stride_od, acc.to(o_ptr.dtype.element_ty))


def attention_flash_decode_int8(
    q: torch.Tensor,
    k_int8: torch.Tensor,
    v_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    kv_len: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Decode FlashAttention over an int8 KV cache with per-vector scales.

    Args:
        q:        (B, Hq,  1, D)  bf16/fp16 — the single new query per sequence.
        k_int8:   (B, Hkv, S, D)  int8 — full preallocated key cache.
        v_int8:   (B, Hkv, S, D)  int8 — full preallocated value cache.
        k_scale:  (B, Hkv, S)     fp32 — per-(b,head,pos) symmetric key scale.
        v_scale:  (B, Hkv, S)     fp32 — per-(b,head,pos) symmetric value scale.
        kv_len:   int32 device scalar — number of valid cached positions.
        sm_scale: 1/sqrt(D) if None.

    Returns:
        (B, 1, Hq, D) token-major output, same dtype as q.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, S, _ = k_int8.shape
    assert Tq == 1, "decode kernel expects a single query position"
    assert D in (16, 32, 64, 128, 256), f"unsupported head_dim {D}"
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty((B, Tq, Hq, D), dtype=q.dtype, device=q.device)

    grid = (B * Hq,)
    _flash_decode_int8_fwd[grid](
        q, k_int8, v_int8, out, k_scale, v_scale, kv_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP,
        BLOCK_N=_DECODE_BLOCK_N, D=D,
        num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
    )
    return out
