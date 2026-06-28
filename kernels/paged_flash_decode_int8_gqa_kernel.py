"""
kernels/paged_flash_decode_int8_gqa_kernel.py

GQA-GROUPED int8-KV decode flash: one program per (batch, KV-head), which reads that
KV-head's int8 K/V ONCE and computes all KV_GROUP sibling query-heads from it.

Why (long-context only)
-----------------------
The per-query-head flash reads each KV-head's K/V KV_GROUP times (once per sibling q
head). At SHORT context the K/V fits the 96 MB L2 so the redundant reads are free
(jun3 measured GQA-grouping at exactly 1.00x). But the long int8-KV cells read
~8.4 GB/step with K/V >> L2, so that 4x redundancy is 4x of HBM bandwidth. Grouping
the read across the KV_GROUP q heads cuts the dominant KV read ~KV_GROUP-fold.

Math: q_g · (k_int8 · k_scale) folds the per-position key scale as a scalar after the
dot; the value scale folds into the softmax weights. Per-(b,head,pos) int8 K/V scales.
Graph-safe (static B*Hkv grid, kv_len from a device scalar, no autotune).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_N = 32
_NUM_WARPS = 2
_NUM_STAGES = 1


@triton.jit
def _gqa_int8_flash_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, ks_ptr, vs_ptr, kvlen_ptr, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ksb, stride_ksh, stride_ksn,
    stride_vsb, stride_vsh, stride_vsn,
    stride_ob, stride_oh, stride_od,
    Hkv,
    KV_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (batch, kv-head). Reads this kv-head's int8 K/V once and runs
    online softmax for all KV_GROUP query heads that map to it."""
    off = tl.program_id(0)
    b = off // Hkv
    hkv = off % Hkv
    hq0 = hkv * KV_GROUP

    kv_len = tl.load(kvlen_ptr)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, KV_GROUP)
    # q tile (KV_GROUP, D) for the sibling query heads
    q = tl.load(
        q_ptr + b * stride_qb + (hq0 + offs_g)[:, None] * stride_qh + offs_d[None, :] * stride_qd
    ).to(tl.float32)

    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh
    ks_base = ks_ptr + b * stride_ksb + hkv * stride_ksh
    vs_base = vs_ptr + b * stride_vsb + hkv * stride_vsh

    m_i = tl.zeros((KV_GROUP,), dtype=tl.float32) + float("-inf")
    l_i = tl.zeros((KV_GROUP,), dtype=tl.float32)
    acc = tl.zeros((KV_GROUP, D), dtype=tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < kv_len
        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0).to(tl.float32)   # (BLOCK_N, D)
        ks = tl.load(ks_base + offs_n * stride_ksn, mask=n_mask, other=0.0).to(tl.float32)
        # qk for all KV_GROUP queries: (KV_GROUP, BLOCK_N)
        qk = tl.dot(q, k.T) * ks[None, :] * sm_scale
        qk = tl.where(n_mask[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))                  # (KV_GROUP,)
        p = tl.exp(qk - m_new[:, None])                              # (KV_GROUP, BLOCK_N)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0).to(tl.float32)    # (BLOCK_N, D)
        vs = tl.load(vs_base + offs_n * stride_vsn, mask=n_mask, other=0.0).to(tl.float32)
        pv = (p * vs[None, :]).to(tl.float32)                        # fold value scale
        acc = acc * alpha[:, None] + tl.dot(pv, v)                   # (KV_GROUP, D)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + b * stride_ob + (hq0 + offs_g)[:, None] * stride_oh + offs_d[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )


def gqa_int8_flash_decode(q, k_int8, v_int8, k_scale, v_scale, kv_len, sm_scale=None):
    """GQA-grouped int8-KV decode flash.

    q: (B, Hq, 1, D); k/v_int8: (B, Hkv, S, D) int8; k/v_scale: (B, Hkv, S);
    kv_len: int32 device scalar. Returns (B, 1, Hq, D)."""
    B, Hq, Tq, D = q.shape
    _, Hkv, S, _ = k_int8.shape
    assert Tq == 1
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)
    out = torch.empty((B, Tq, Hq, D), dtype=q.dtype, device=q.device)
    grid = (B * Hkv,)
    _gqa_int8_flash_fwd[grid](
        q, k_int8, v_int8, out, k_scale, v_scale, kv_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        out.stride(0), out.stride(2), out.stride(3),
        Hkv, KV_GROUP=KV_GROUP, BLOCK_N=_BLOCK_N, D=D,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out
