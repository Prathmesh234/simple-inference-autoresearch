"""
kernels/rope_kernel.py — Section 14c

Triton fused RoPE kernel — model-agnostic.

Why this is faster than PyTorch
--------------------------------
The PyTorch reference (ops/rope.py) for each of Q and K does:
    rotated = rotate_half(x)         # read x, write rotated (with -x slice copy)
    out     = x * cos                # read x + cos, write tmp1
    out    += rotated * sin          # read tmp1 + rotated + sin, write out

That's ~5 reads + 3 writes of (B, T, n_heads, head_dim) bf16 traffic per
tensor, plus an extra rotate_half() allocation. RoPE is memory-bound:
6 FLOPs per element on a GPU with 1.56 FLOPs/byte break-even.

The fused kernel collapses all of that into:
    read x once, read cos+sin once (shared across all heads of a token),
    write out once.
That is the theoretical minimum.

Generality across model families
---------------------------------
RoPE varies along three axes between model families:

1. Pair layout
   - **NEOX / Llama / Mistral / Qwen / Yi / DeepSeek / Phi-3**:
       pairs are (x_i, x_{i+HALF})         (split-half)
       cos/sin in HF's checkpoint are stored DUPLICATED to width head_dim
   - **GPT-J / GPT-NeoX-original / ChatGLM**:
       pairs are (x_{2i}, x_{2i+1})         (interleaved / adjacent)
       cos/sin are width head_dim/2

2. head_dim — anything even (64, 96, 128, 256, ...)

3. cos/sin storage width — either head_dim (duplicated) or head_dim/2 (raw)

We handle (1) via the INTERLEAVED constexpr branch, (2) via constexpr
autotuning, and (3) in the Python wrapper which slices the duplicated layout
down to the half-width the kernel consumes.

What we do NOT handle (yet)
---------------------------
- Partial RoPE (rotate only first rot_dim of head_dim, pass the rest through).
  Used by GPT-J/Phi-2. Would need an extra copy for the tail; easy to add
  by exposing a ROT_DIM constexpr and storing x as-is for cols ≥ ROT_DIM.
- Position IDs that differ per sequence in a batch (variable start_pos).
  The kernel infers position from row_pid % seq_len, which assumes every
  batch element shares the [0, T) positional range of the passed cos/sin.

Kernel design
-------------
Launch grid: (n_tokens, n_heads)
- One program handles one (token, head) and computes BOTH halves of the
  rotation in a single pass — no `rotate_half` materialization.
- BLOCK_SIZE = next_power_of_2(HALF). For head_dim=128 → HALF=64.
- INTERLEAVED constexpr selects the pair layout at compile time, so there's
  no runtime branch.

A multi-head-per-program variant was tried (BLOCK_H tile to amortize cos/sin
loads). It regressed at T=128–2048 because cos/sin is only ~5–10% of total
traffic and the 2D indexing overhead outweighed the BW saving.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=["HALF", "INTERLEAVED"],
)
@triton.jit
def _rope_qk_fwd(
    q_ptr,        # input  Q (n_tokens, n_heads_q,  head_dim), rows q_in_rs apart
    k_ptr,        # input  K (n_tokens, n_heads_kv, head_dim), rows k_in_rs apart
    cos_ptr,      # cos    (seq_len, cos_row_stride) — only first HALF used
    sin_ptr,      # sin    same layout as cos
    q_out_ptr,    # output Q (n_tokens, n_heads_q,  head_dim) contiguous
    k_out_ptr,    # output K (n_tokens, n_heads_kv, head_dim) contiguous
    q_in_rs,      # input row stride for Q (== n_heads_q*head_dim if contiguous)
    k_in_rs,      # input row stride for K (== n_heads_kv*head_dim if contiguous)
    n_heads_q: tl.constexpr,
    n_heads_kv: tl.constexpr,
    seq_len,
    cos_row_stride,  # stride between cos rows (HALF for raw, head_dim for HF duplicated)
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,        # next_power_of_2(HALF)
    INTERLEAVED: tl.constexpr,       # False = Llama/NEOX, True = GPT-J style
):
    """
    One program per (token, head) across BOTH Q and K in a single launch.

    The grid second dim ranges over n_heads_q + n_heads_kv: program columns
    [0, n_heads_q) rotate Q heads, columns [n_heads_q, n_heads_q+n_heads_kv)
    rotate K heads. This fuses what used to be two separate kernel launches
    (one for Q, one for K) into one — halving RoPE launches per decode step.

    Input is read with an explicit per-row stride (q_in_rs / k_in_rs) so the
    Q/K slices of the fused-qkv GEMM output — which are NON-contiguous (their row
    stride is the full qkv width, e.g. 6144, not n_heads*head_dim) — can be rotated
    in place WITHOUT a prior `.contiguous()` copy (2 copy kernels/layer eliminated
    from every decode step). Within a row the heads are head_dim-contiguous and the
    head_dim itself is unit-stride for both the contiguous and the sliced layouts,
    so only the row base differs between input and (contiguous) output. The rotation
    math is identical (float32 compute, store in the input dtype) → bit-identical.
    """
    row_pid = tl.program_id(0)       # one row = one token (B*T flattened)
    col_pid = tl.program_id(1)       # head index across Q heads then K heads

    # Position within the passed cos/sin window. Assumes every batch element
    # shares the same positional range (true for prefill and for decode T=1).
    seq_pos = row_pid % seq_len
    cs_base = seq_pos * cos_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF

    cos = tl.load(cos_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)

    out_dtype = q_ptr.dtype.element_ty  # q and k share dtype; hoist out of branches

    # Within-head offsets of the rotation pair — identical for the (possibly
    # strided) input base and the contiguous output base.
    if INTERLEAVED:
        da = 2 * cols
        db = da + 1
    else:
        da = cols
        db = da + HALF

    if col_pid < n_heads_q:
        in_base = row_pid * q_in_rs + col_pid * HEAD_DIM
        out_base = (row_pid * n_heads_q + col_pid) * HEAD_DIM
        x_a = tl.load(q_ptr + in_base + da, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(q_ptr + in_base + db, mask=mask, other=0.0).to(tl.float32)
        out_a = x_a * cos - x_b * sin
        out_b = x_b * cos + x_a * sin
        tl.store(q_out_ptr + out_base + da, out_a.to(out_dtype), mask=mask)
        tl.store(q_out_ptr + out_base + db, out_b.to(out_dtype), mask=mask)
    else:
        head = col_pid - n_heads_q
        in_base = row_pid * k_in_rs + head * HEAD_DIM
        out_base = (row_pid * n_heads_kv + head) * HEAD_DIM
        x_a = tl.load(k_ptr + in_base + da, mask=mask, other=0.0).to(tl.float32)
        x_b = tl.load(k_ptr + in_base + db, mask=mask, other=0.0).to(tl.float32)
        out_a = x_a * cos - x_b * sin
        out_b = x_b * cos + x_a * sin
        tl.store(k_out_ptr + out_base + da, out_a.to(out_dtype), mask=mask)
        tl.store(k_out_ptr + out_base + db, out_b.to(out_dtype), mask=mask)


def _check_cos_sin(cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> int:
    """
    Validate cos/sin shape and return the row stride to use.

    Accepts either layout the calling code might produce:
      - HF / Llama duplicated: (T, head_dim)   → row stride = head_dim
      - Raw half-width:        (T, head_dim/2) → row stride = head_dim/2

    The kernel only ever reads the first head_dim/2 elements of each row,
    so we just pass the stride and avoid any copy.
    """
    half = head_dim // 2
    if cos.shape[-1] not in (head_dim, half):
        raise ValueError(
            f"cos/sin last dim must be head_dim ({head_dim}) or head_dim/2 ({half}); "
            f"got {cos.shape[-1]}"
        )
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have identical shape; got {cos.shape} vs {sin.shape}")
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("cos/sin must be unit-stride along the last dim")
    if cos.stride(0) != sin.stride(0):
        raise ValueError("cos and sin must share the same row stride")
    return cos.stride(0)


def _apply_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_row_stride: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K in a single fused kernel launch."""
    B, T, n_heads_q, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert k.shape[0] == B and k.shape[1] == T and k.shape[3] == head_dim
    n_heads_kv = k.shape[2]
    seq_len = cos.shape[0]
    assert sin.shape[0] == seq_len

    # The kernel reads Q/K with an explicit per-(B*T)-row stride, so the
    # NON-contiguous Q/K slices of the fused-qkv GEMM output are rotated in place
    # without a `.contiguous()` copy. The required layout is: head_dim unit-stride
    # and heads head_dim-apart within a row (true for both a contiguous tensor and
    # a last-dim slice of the wider qkv buffer), and a uniform flattened-row stride
    # == stride(1) (holds when stride(0) == T*stride(1)). Fall back to a copy for
    # any exotic layout so correctness never depends on the fast-path assumptions.
    def _row_stride(x):
        if (x.stride(-1) == 1 and x.stride(2) == head_dim
                and x.stride(0) == T * x.stride(1)):
            return x.stride(1)
        return None

    q_rs, k_rs = _row_stride(q), _row_stride(k)
    if q_rs is None:
        q = q.contiguous(); q_rs = n_heads_q * head_dim
    if k_rs is None:
        k = k.contiguous(); k_rs = n_heads_kv * head_dim

    q_out = torch.empty(B, T, n_heads_q, head_dim, dtype=q.dtype, device=q.device)
    k_out = torch.empty(B, T, n_heads_kv, head_dim, dtype=k.dtype, device=k.device)

    n_tokens   = B * T
    HALF       = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(HALF)

    # Grid covers every (token, head) across BOTH Q and K heads in one launch.
    grid = (n_tokens, n_heads_q + n_heads_kv)
    _rope_qk_fwd[grid](
        q, k, cos, sin, q_out, k_out,
        q_rs, k_rs,
        n_heads_q,
        n_heads_kv,
        seq_len,
        cos_row_stride,
        HEAD_DIM=head_dim,
        HALF=HALF,
        BLOCK_SIZE=BLOCK_SIZE,
        INTERLEAVED=interleaved,
    )
    return q_out, k_out


def rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for ops.rope.apply_rope.

    Args:
        q:           (B, T, n_heads_q,  head_dim)
        k:           (B, T, n_heads_kv, head_dim)
        cos:         (T, head_dim) duplicated  OR  (T, head_dim/2) raw.
                     Only the first head_dim/2 elements of each row are read.
        sin:         same shape as cos
        interleaved: False (default, NEOX/Llama split-half pairs) or
                     True (GPT-J style adjacent pairs)

    Returns:
        (q_rot, k_rot) — same shapes and dtypes as inputs.
    """
    head_dim = q.shape[-1]
    cos_row_stride = _check_cos_sin(cos, sin, head_dim)
    return _apply_qk(q, k, cos, sin, cos_row_stride, interleaved)
