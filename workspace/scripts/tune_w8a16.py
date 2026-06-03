"""Offline config search for the W8A16 GEMM at the two MLP weight shapes.

Eager-only, fully controlled (no triton.autotune do_bench), so it is safe.
Tries a curated set of safe tensor-core tiles, checks correctness vs F.linear,
times each, and prints the best config per (N, K) at the decode M and prefill M.
"""
import os, sys, time, itertools
sys.path.insert(0, os.getcwd())
import torch, triton
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel

DEV, DT = "cuda", torch.bfloat16
torch.manual_seed(0)

SHAPES = [("mlp.gate_up", 2 * 14336, 4096), ("mlp.down", 4096, 14336)]
MS = [128, 1, 2048]

BMS, BNS, BKS = (16, 32, 64, 128), (64, 128, 256), (32, 64, 128)
STAGES, WARPS = (2, 3, 4), (4, 8)


def run_cfg(x, w_int8, scale, N, K, M, bm, bn, bk, ns, nw):
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _w8a16_gemm[grid](
        x, w_int8, scale, y, M, N, K,
        x.stride(0), x.stride(1), w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_stages=ns, num_warps=nw,
    )
    return y


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


for name, N, K in SHAPES:
    w = torch.randn(N, K, device=DEV, dtype=DT) * 0.02
    w_int8, scale = quantize_int8_per_channel(w)
    for M in MS:
        x = torch.randn(M, K, device=DEV, dtype=DT)
        y_ref = torch.nn.functional.linear(x, w)
        t_bf16 = bench(lambda: torch.nn.functional.linear(x, w))
        best = None
        for bm, bn, bk, ns, nw in itertools.product(BMS, BNS, BKS, STAGES, WARPS):
            try:
                y = run_cfg(x, w_int8, scale, N, K, M, bm, bn, bk, ns, nw)
                torch.cuda.synchronize()
            except Exception:
                continue
            rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-9)
            if rel >= 2e-2:
                continue
            t = bench(lambda: run_cfg(x, w_int8, scale, N, K, M, bm, bn, bk, ns, nw))
            if best is None or t < best[0]:
                best = (t, (bm, bn, bk, ns, nw), rel.item())
        if best:
            t, cfg, rel = best
            print(f"{name:<12} M={M:<5} best={cfg} int8={t:.3f}ms bf16={t_bf16:.3f}ms "
                  f"speedup={t_bf16/t:.2f}x rel={rel:.2e}")
        else:
            print(f"{name:<12} M={M:<5} NO VALID CONFIG")
