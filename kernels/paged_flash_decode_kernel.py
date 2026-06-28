"""
kernels/paged_flash_decode_kernel.py

PagedAttention decode kernel (single query, Tq==1) — reads K/V from a BLOCK-PAGED
KV cache via a per-sequence block table, the way vLLM/SGLang store the cache.

Why paged
---------
The contiguous KV cache (model/kv_cache.py) pre-reserves one max_seq_len-long buffer
per request, so long-context flavors OOM long before the GPU is full and the memory
is fragmented. A paged cache stores KV in fixed-size BLOCKS drawn from a flat pool;
a per-request block table maps logical block -> physical block id, so a request only
holds the blocks it actually fills and blocks can be shared across requests (prefix
reuse / RadixAttention build on this).

Headline-safe design
---------------------
PAGE_SIZE == the flash BLOCK_N (16): each loop iteration reads exactly ONE physical
page, which is contiguous in the pool — identical memory access pattern to the
contiguous decode kernel, plus one int32 block-table load per page (cheap). So at the
short-context instruct headline the paged kernel matches the contiguous one; the win
is at long context / high batch (no per-request over-reservation, no OOM cliff).

Graph safety: static B*Hq grid, kv_len read from a device scalar, block table read
from device memory at replay — one captured graph serves every decode step. No
autotune (hardcoded BLOCK_N=PAGE_SIZE, num_warps=1, the EXP-L tuned config).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_DECODE_NUM_WARPS = 1
_DECODE_NUM_STAGES = 2


@triton.jit
def _paged_flash_decode_fwd(
    q_ptr, k_pool_ptr, v_pool_ptr, o_ptr, bt_ptr, kvlen_ptr, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kp_blk, stride_kp_h, stride_kp_n, stride_kp_d,
    stride_vp_blk, stride_vp_h, stride_vp_n, stride_vp_d,
    stride_bt_b, stride_bt_l,
    stride_ob, stride_oh, stride_od,
    Hq, KV_GROUP,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (batch, query-head). Single query row, online softmax over the
    paged K/V: iterate logical blocks, resolve each to a physical page via the block
    table, read that contiguous page, update the running (m, l, acc)."""
    off_bh = tl.program_id(0)
    b = off_bh // Hq
    hq = off_bh % Hq
    hkv = hq // KV_GROUP

    kv_len = tl.load(kvlen_ptr)

    q_base = q_ptr + b * stride_qb + hq * stride_qh
    o_base = o_ptr + b * stride_ob + hq * stride_oh
    bt_base = bt_ptr + b * stride_bt_b

    offs_d = tl.arange(0, D)
    offs_p = tl.arange(0, PAGE_SIZE)
    q = tl.load(q_base + offs_d * stride_qd).to(tl.float32)  # (D,)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    num_blocks = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    for blk in range(0, num_blocks):
        phys = tl.load(bt_base + blk * stride_bt_l)        # physical block id (int32)
        offs_n = blk * PAGE_SIZE + offs_p
        n_mask = offs_n < kv_len

        k_base = k_pool_ptr + phys * stride_kp_blk + hkv * stride_kp_h
        k = tl.load(
            k_base + offs_p[:, None] * stride_kp_n + offs_d[None, :] * stride_kp_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)  # (PAGE_SIZE, D)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale       # (PAGE_SIZE,)
        qk = tl.where(n_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v_base = v_pool_ptr + phys * stride_vp_blk + hkv * stride_vp_h
        v = tl.load(
            v_base + offs_p[:, None] * stride_vp_n + offs_d[None, :] * stride_vp_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)  # (PAGE_SIZE, D)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe
    tl.store(o_base + offs_d * stride_od, acc.to(o_ptr.dtype.element_ty))


def paged_attention_flash_decode(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: torch.Tensor,
    kv_len: torch.Tensor,
    page_size: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Decode FlashAttention over a paged KV cache.

    Args:
        q:           (B, Hq, 1, D) bf16 — the single new query per sequence.
        k_pool:      (num_blocks, Hkv, page_size, D) — flat key block pool.
        v_pool:      (num_blocks, Hkv, page_size, D) — flat value block pool.
        block_table: (B, max_blocks) int32 — logical->physical block id per sequence.
        kv_len:      int32 device scalar — valid cached positions (== start_pos+1).
        page_size:   tokens per block (== the flash BLOCK_N).
        sm_scale:    1/sqrt(D) if None.

    Returns:
        (B, 1, Hq, D) token-major output, same dtype as q.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, P, _ = k_pool.shape
    assert Tq == 1, "decode kernel expects a single query position"
    assert P == page_size, f"k_pool page dim {P} != page_size {page_size}"
    assert D in (16, 32, 64, 128, 256), f"unsupported head_dim {D}"
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty((B, Tq, Hq, D), dtype=q.dtype, device=q.device)
    grid = (B * Hq,)
    _paged_flash_decode_fwd[grid](
        q, k_pool, v_pool, out, block_table, kv_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP,
        PAGE_SIZE=page_size, D=D,
        num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
    )
    return out
