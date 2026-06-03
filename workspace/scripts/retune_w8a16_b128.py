"""Targeted retune of the W8A16 down-proj GEMM at the b128 decode shape.

The dominant decode op is the MLP w8a16 GEMM (~60% of the profiled b128 step).
The hardcoded config picks BLOCK_M=32 for the down-proj (N=4096) at M=128, which
splits M=128 into 4 row-blocks that each re-stream the full int8 weight from HBM
(4x weight traffic on a weight-bandwidth-bound op). This sweeps BLOCK_M/N/K and
warps/stages for BOTH MLP shapes at M=128 to check whether a larger BLOCK_M
(less redundant weight traffic) is actually faster.
"""
import itertools
import torch
import triton

from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16

# (name, N, K)  — gate_up and down for Llama-3.1-8B
SHAPES = [("gate_up", 28672, 4096), ("down", 4096, 14336)]
M = 128


def run_cfg(x, w_int8, scale, N, K, BM, BN, BK, ns, nw):
    y = torch.empty((M, N), dtype=x.dtype, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm[grid](
        x, w_int8, scale, y, M, N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
        num_stages=ns, num_warps=nw,
    )
    return y


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


for name, N, K in SHAPES:
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt) * 0.02
    w_int8, scale = quantize_int8_per_channel(w)
    w_int8 = w_int8.to(dev); scale = scale.to(dev)
    y_ref = (x.float() @ (w_int8.float().T)) * scale.float()

    results = []
    for BM, BN, BK, ns, nw in itertools.product(
        (16, 32, 64, 128), (64, 128, 256), (64, 128), (2, 3, 4), (4, 8)
    ):
        if BM * BN > 128 * 256:
            continue
        try:
            y = run_cfg(x, w_int8, scale, N, K, BM, BN, BK, ns, nw)
            err = (y.float() - y_ref).abs().max().item()
            if err > 1.0:
                continue
            t = bench(lambda: run_cfg(x, w_int8, scale, N, K, BM, BN, BK, ns, nw))
            results.append((t, BM, BN, BK, ns, nw, err))
        except Exception:
            continue
    results.sort()
    print(f"\n=== {name} (N={N}, K={K}) M={M} — top 8 ===")
    for t, BM, BN, BK, ns, nw, err in results[:8]:
        tag = " <-- current" if (name == "down" and (BM, BN, BK, ns, nw) == (32, 128, 128, 3, 8)) \
            or (name == "gate_up" and (BM, BN, BK, ns, nw) == (128, 128, 128, 2, 4)) else ""
        print(f"  {t:7.1f}us  BM={BM:3d} BN={BN:3d} BK={BK:3d} ns={ns} nw={nw} err={err:.3f}{tag}")
    # also print the current config's time explicitly
    cur = (32, 128, 128, 3, 8) if name == "down" else (128, 128, 128, 2, 4)
    for t, BM, BN, BK, ns, nw, err in results:
        if (BM, BN, BK, ns, nw) == cur:
            print(f"  current config time: {t:.1f}us")
            break
