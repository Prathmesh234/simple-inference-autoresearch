"""
W8A8 int8 tensor-core GEMM for the decode hot path.

Why W8A8 on top of the existing weight-only W8A16 path?
-------------------------------------------------------
The W8A16 kernel (kernels/w8a16_gemm_kernel.py) stores int8 weights but upcasts
them to bf16 *inside* the kernel and runs a bf16 ``tl.dot``. At the b128 decode
shape the dominant gate_up GEMM (N=28672, K=4096, M=128) then sits at ~113
TFLOP/s and ~442 GB/s — saturating neither the bf16 compute roof (~360 TFLOP/s)
nor HBM (~960 GB/s). The bottleneck is the int8->bf16 upcast plus the M=128 tile
occupancy.

W8A8 instead quantizes the *activations* to int8 as well (per-token / per-row
dynamic symmetric scale) and runs a true int8 ``tl.dot`` accumulating into int32.
Ada's int8 tensor cores run at ~2x the bf16 rate and there is no upcast, so the
gate_up GEMM drops from ~268us to ~160us (1.67x) at M=128, INCLUDING the fused
activation-quant kernel below.

Scope (deliberately narrow — validated by cheap standalone + real-activation benches):
  * Used for the MLP **gate_up** projection only. Its input is the post-rmsnorm
    residual stream; per-token dynamic scaling absorbs the activation outliers so
    the end-to-end rel_err matches the already-accepted W8A16 weight-only int8
    (~0.06 on real decode activations).
  * NOT used for qkv (its GEMM is too small for the quant overhead to pay off:
    0.70x), nor for down (post-SwiGLU activations have per-layer outliers up to
    rel_err 0.19), nor o_proj / lm_head.
  * For M<=16 (b1 decode, bandwidth-bound, no int8-MMA win) and M>256 (prefill)
    we fall back to the W8A16 path so latency/TTFT are never regressed.

Graph safety: like w8a16, the tiles are hardcoded (no autotune) and every op has
static shapes at a fixed batch, so the whole path replays inside the captured
CUDA decode graph. The weights are the SAME int8 buffers + per-channel scale used
by W8A16 (quantize_int8_per_channel), so this is a drop-in for the activation side
only — no extra weight storage.
"""

import torch
import triton
import triton.language as tl

from kernels.w8a16_gemm_kernel import w8a16_linear_triton


@triton.jit
def _quant_per_token(x_ptr, xi_ptr, xs_ptr, M, K, sxm, sxk,
                     BLOCK_K: tl.constexpr):
    """One program per row: symmetric per-token int8 quant.

    xs[row] = max(|x[row]|, 1e-8) / 127 ;  xi[row] = round(x[row] / xs[row]).
    BLOCK_K must be >= K (a power of two) so each row is handled in one block.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * sxm + offs * sxk, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-8)
    scale = amax / 127.0
    xi = tl.extra.cuda.libdevice.round(x / scale)
    xi = tl.minimum(tl.maximum(xi, -127.0), 127.0).to(tl.int8)
    tl.store(xi_ptr + row * K + offs, xi, mask=mask)
    tl.store(xs_ptr + row, scale)


@triton.jit
def _w8a8_gemm(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
               sxm, sxk, swn, swk, sym, syn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """y = (x_int8 @ w_int8.T) * act_scale[:,None] * w_scale[None,:] -> bf16.

    int32 accumulation is exact: K=4096 terms each <= 127*127 -> max 6.6e7 < 2^31.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        acc += tl.dot(a, b.T, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    xs = tl.load(xs_ptr + offs_m)[:, None].to(tl.float32)
    ws = tl.load(ws_ptr + offs_n)[None, :].to(tl.float32)
    y = acc.to(tl.float32) * xs * ws

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn
    tl.store(y_ptrs, y.to(tl.bfloat16),
             mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))


def w8a8_linear_triton(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """int8-activation x int8-weight linear: y ≈ x @ (w_int8 * scale).T.

    Drop-in for w8a16_linear_triton (same weight buffers). Falls back to the
    W8A16 path for M<=16 (b1 decode) and M>256 (prefill), where int8 MMA gives
    no win and the activation-quant overhead would only hurt.

    Args:
        x:      (..., K) bf16/fp16 activations.
        w_int8: (N, K) int8 weights (out_features, in_features).
        scale:  (N,) per-output-channel scale.
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

    # Bandwidth-bound (b1) or prefill: int8 MMA gives no benefit -> W8A16.
    if M <= 16 or M > 256:
        return w8a16_linear_triton(x, w_int8, scale)

    x2d = x2d.contiguous()
    xi = torch.empty((M, K), dtype=torch.int8, device=x.device)
    xs = torch.empty((M,), dtype=torch.float32, device=x.device)
    _quant_per_token[(M,)](
        x2d, xi, xs, M, K, x2d.stride(0), x2d.stride(1),
        BLOCK_K=triton.next_power_of_2(K), num_warps=4,
    )

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    num_stages, num_warps = 2, 4
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a8_gemm[grid](
        xi, w_int8, xs, scale, y, M, N, K,
        xi.stride(0), xi.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N)


@triton.jit
def _w8a8_swiglu_fwd(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, I, K,
                     sxm, sxk, swn, swk, sym, syn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """Fused gate_up W8A8 GEMM + SwiGLU epilogue.

    y[m, n] = silu(gate) * up   where
        gate = (x_int8 @ Wg.T) * act_scale[m] * w_scale[n]
        up   = (x_int8 @ Wu.T) * act_scale[m] * w_scale[I + n]
    Wg = w rows [0, I)   (gate),   Wu = w rows [I, 2I)  (up).
    Two int32 accumulators are accumulated in the K-loop so the (M, 2I)
    intermediate is never materialised and the standalone swiglu launch is
    removed. Output is (M, I) bf16, consumed directly by the down projection.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(I, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % I
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    wg_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
    wu_ptrs = w_ptr + (I + offs_n)[:, None] * swn + offs_k[None, :] * swk

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        bg = tl.load(wg_ptrs)
        bu = tl.load(wu_ptrs)
        acc_g += tl.dot(a, bg.T, out_dtype=tl.int32)
        acc_u += tl.dot(a, bu.T, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * sxk
        wg_ptrs += BLOCK_K * swk
        wu_ptrs += BLOCK_K * swk

    xsc = tl.load(xs_ptr + offs_m)[:, None].to(tl.float32)
    wsg = tl.load(ws_ptr + offs_n)[None, :].to(tl.float32)
    wsu = tl.load(ws_ptr + I + offs_n)[None, :].to(tl.float32)
    gate = acc_g.to(tl.float32) * xsc * wsg
    up = acc_u.to(tl.float32) * xsc * wsu
    out = (gate * tl.sigmoid(gate)) * up

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn
    tl.store(y_ptrs, out.to(tl.bfloat16),
             mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < I))


def w8a8_swiglu_prequant(
    x_int8: torch.Tensor,
    act_scale: torch.Tensor,
    w_gate_up_int8: torch.Tensor,
    w_gate_up_scale: torch.Tensor,
    orig_shape,
) -> torch.Tensor:
    """Fused gate_up W8A8 GEMM + SwiGLU when the activation is already int8.

    Replaces (w8a8_linear_prequant -> chunk -> swiglu_triton) for the decode
    bucket: produces silu(gate)*up of shape (..., I) in one kernel, skipping the
    (M, 2I) intermediate write and the separate swiglu launch (EXP-E).

    BLOCK_N is pinned to 64: the two int32 accumulators spill registers at
    BLOCK_N>=128 (catastrophic ~17x slowdown), while 64 is the stable optimum.
    """
    M, K = x_int8.shape
    two_I = w_gate_up_int8.shape[0]
    I = two_I // 2
    y = torch.empty((M, I), dtype=torch.bfloat16, device=x_int8.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 128
    num_stages, num_warps = 2, 4
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(I, BLOCK_N),)
    _w8a8_swiglu_fwd[grid](
        x_int8, w_gate_up_int8, act_scale, w_gate_up_scale, y, M, I, K,
        x_int8.stride(0), x_int8.stride(1),
        w_gate_up_int8.stride(0), w_gate_up_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], I)


def w8a8_linear_prequant(
    x_int8: torch.Tensor,
    act_scale: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    orig_shape,
) -> torch.Tensor:
    """W8A8 GEMM when the activation is ALREADY int8-quantized (per-token).

    Used by the EXP-D path where the preceding fused add_rmsnorm emits the int8
    normed output + per-row scale, so the standalone _quant_per_token launch is
    skipped. x_int8 is (M, K) int8, act_scale is (M,) fp32.
    """
    M, K = x_int8.shape
    N = w_int8.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x_int8.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    num_stages, num_warps = 2, 4
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a8_gemm[grid](
        x_int8, w_int8, act_scale, w_scale, y, M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N)


def w8a8_qkv_prequant(
    x_int8: torch.Tensor,
    act_scale: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
    orig_shape,
) -> torch.Tensor:
    """W8A8 fused-qkv GEMM with the activation ALREADY int8-quantized per-token.

    The decode qkv projection (fused q+k+v, N=6144, K=4096) was previously left on
    the weight-only W8A16 path because a *self-quantizing* W8A8 lost (the standalone
    _quant_per_token launch costs more than the GEMM saves). Here the int8 normed
    activation + per-row scale come for free from the preceding fused
    ``attn_norm.add_norm_quant`` (the EXP-D trick), so only the GEMM remains.

    At the b128 decode shape (M=128) the int8 GEMM needs its OWN tile: BLOCK_N=64
    (num_stages=3, num_warps=8) runs at 22us vs 49us for the EXP-G-tuned W8A16
    (2.2x). The default (128,128,128) tile — what the earlier "qkv W8A8 is dead"
    conclusion used — is ~2x slower at this N=6144 shape (same "wrong tile" lesson
    as EXP-G). Tile is hardcoded (no autotune) so the static-shape decode path
    replays inside the captured CUDA graph.

    Caller (ops/attention._project_qkv) owns the M-bucket dispatch: this is only
    invoked for 16 < M <= 256, with the bf16 W8A16 fallback used outside that range.
    """
    M, K = x_int8.shape
    N = w_int8.shape[0]
    assert x_int8.is_contiguous(), "qkv W8A8 expects contiguous int8 activations"
    assert K % 128 == 0 and N % 64 == 0, f"qkv W8A8 tile assumes K%128==0, N%64==0 (got K={K}, N={N})"
    assert act_scale.shape == (M,), f"act_scale must be (M,), got {tuple(act_scale.shape)}"
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x_int8.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 128
    num_stages, num_warps = 3, 8
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a8_gemm[grid](
        x_int8, w_int8, act_scale, w_scale, y, M, N, K,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N)


def w8a8_down_linear(
    x: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Self-quantizing W8A8 GEMM tuned for the MLP **down** projection.

    down is the dominant decode GEMM (M=128, K=14336 intermediate, N=4096 hidden).
    Its 58.7 MB int8 weight fits Ada's 96 MB L2, so the GEMM is compute/L2-bound
    rather than HBM-bound, and the int8 tensor cores (~2x bf16) pay off: with the
    down-specific tile below the int8 GEMM runs ~50us vs ~95us for the W8A16
    (weight-int8, bf16-MMA) path — a 1.9x GEMM speedup.

    Unlike gate_up/qkv, down's input is the SwiGLU output, which is NOT produced by
    a per-row pass, so there is no free activation int8 quant: we pay a standalone
    per-token _quant_per_token launch (~15us at K=14336). Net standalone is still
    ~1.2-1.45x over W8A16. Tile is hardcoded (no autotune) and shapes are static at
    a fixed batch, so the whole path replays inside the captured CUDA decode graph.

    Caller (ops/mlp.SwiGLUMLP.forward) owns the M-bucket dispatch: only invoked for
    16 < M <= 256; b1 (M<=16, bandwidth-bound) and prefill (M>256) keep W8A16.

    Args:
        x:      (..., K) bf16 SwiGLU output (down input).
        w_int8: (N, K) int8 down weight (out_features=hidden, in_features=I).
        w_scale: (N,) per-output-channel fp32 scale.
    Returns:
        (..., N) bf16.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K)
    if x2d.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        x2d = x2d.to(torch.bfloat16)
    x2d = x2d.contiguous()
    M = x2d.shape[0]
    N = w_int8.shape[0]

    xi = torch.empty((M, K), dtype=torch.int8, device=x.device)
    xs = torch.empty((M,), dtype=torch.float32, device=x.device)
    _quant_per_token[(M,)](
        x2d, xi, xs, M, K, x2d.stride(0), x2d.stride(1),
        BLOCK_K=triton.next_power_of_2(K), num_warps=4,
    )

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 256
    num_stages, num_warps = 4, 4
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a8_gemm[grid](
        xi, w_int8, xs, w_scale, y, M, N, K,
        xi.stride(0), xi.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_stages=num_stages, num_warps=num_warps,
    )
    return y.reshape(*orig_shape[:-1], N)
