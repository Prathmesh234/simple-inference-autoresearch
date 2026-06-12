"""
kernels/flash_decode_kernel.py

FlashAttention-2 forward specialized for the single-token decode step, with the
key/value length read from a DEVICE scalar so the launch is CUDA-graph safe.

Why a separate kernel
---------------------
The generic `attention_flash_triton` (kernels/attention_kernel.py) takes the
cache length `Tk` as a Python int. That is fine for the eager path, but a
captured CUDA graph bakes Python ints at capture time, so a single graph could
not serve a growing decode. The obvious graph-safe alternative — SDPA over the
full preallocated cache with a boolean prefix mask — silently disables the fused
FlashAttention backend (an explicit mask forces the math/efficient fallback),
which is dramatically slower at batch 128 (it showed up as gemv kernels eating
~20% of decode time).

This kernel keeps the proven FlashAttention path and just reads the valid prefix
length `kv_len` from device memory (`kvlen_ptr`). The launch grid is static
(B*Hq programs, one query row each), and the inner K/V loop bound + the
out-of-range key mask both derive from the device `kv_len`, so one captured
graph replays correctly at every decode position with no mask tensor and no
fallback. Output is written token-major (B, 1, Hq, D) like the eager decode path
so the caller can view straight into (B, 1, Hq*D) for the output projection.

Specialization vs the generic kernel: Tq == 1 (one query per sequence), causal
mask is unnecessary (the single new query attends to the whole cached prefix
[0, kv_len)), so there is no per-row causal bound — just the prefix mask.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Hardcoded launch config for the decode attention kernel. No @triton.autotune:
# its do_bench allocates + syncs, which is illegal under CUDA-graph capture, and
# its key=["D"] caches whichever config the FIRST call's shape (e.g. chat b1 in a
# full sweep) selected, then reuses it for every shape — including the b128
# headline, where it lands on a slow config (~138us/call). A direct sweep of the
# headline shape (B*Hq=4096 single-query programs, kv_len~93) showed num_warps=1
# is fastest (each program is a tiny single-query attention; minimal warps =>
# maximal occupancy), but the old autotune list only offered num_warps>=2 so it
# could never reach it. BLOCK_N=16 is fastest at the short headline kv_len and at
# long contexts (kv_len 2048), losing only ~5% at the mid kv_len~512 range.
_DECODE_BLOCK_N = 16
_DECODE_NUM_WARPS = 1
_DECODE_NUM_STAGES = 2


@triton.jit
def _flash_decode_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr, kvlen_ptr, sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    Hq, KV_GROUP,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """One program per (batch, query-head). Single query row, online softmax.

    The key/value prefix length is read from device memory so the same captured
    graph serves every decode position. GQA: query head hq reads KV head
    hq // KV_GROUP.
    """
    off_bh = tl.program_id(0)
    b  = off_bh // Hq
    hq = off_bh % Hq
    hkv = hq // KV_GROUP

    kv_len = tl.load(kvlen_ptr)

    q_base = q_ptr + b * stride_qb + hq * stride_qh
    k_base = k_ptr + b * stride_kb + hkv * stride_kh
    v_base = v_ptr + b * stride_vb + hkv * stride_vh
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
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)  # (BLOCK_N, D)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale          # (BLOCK_N,)
        qk = tl.where(n_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_new)                                  # (BLOCK_N,)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)  # (BLOCK_N, D)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe

    tl.store(o_base + offs_d * stride_od, acc.to(o_ptr.dtype.element_ty))


def attention_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_len: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Decode FlashAttention with a device-scalar prefix length.

    Args:
        q:      (B, Hq,  1, D)  — the single new query per sequence.
        k, v:   (B, Hkv, S, D)  — the FULL preallocated KV cache for this layer.
        kv_len: int32 device scalar (shape (1,) or ()) — number of valid cached
                positions, i.e. start_pos + 1. Read inside the kernel so a single
                captured CUDA graph serves every decode step.
        sm_scale: 1/sqrt(D) if None.

    Returns:
        (B, 1, Hq, D) token-major output, same dtype as q.
    """
    B, Hq, Tq, D = q.shape
    _, Hkv, S, _ = k.shape
    assert Tq == 1, "decode kernel expects a single query position"
    assert D in (16, 32, 64, 128, 256), f"unsupported head_dim {D}"
    KV_GROUP = Hq // Hkv
    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty((B, Tq, Hq, D), dtype=q.dtype, device=q.device)

    grid = (B * Hq,)
    _flash_decode_fwd[grid](
        q, k, v, out, kv_len, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        Hq, KV_GROUP,
        BLOCK_N=_DECODE_BLOCK_N, D=D,
        num_warps=_DECODE_NUM_WARPS, num_stages=_DECODE_NUM_STAGES,
    )
    return out
