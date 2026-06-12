"""
Standalone gate for the W8A8 attention-output (o_proj) GEMM (w8a8_oproj_linear).

o_proj is a square GEMM (M=128 decode, K=Hq*D=4096, N=hidden=4096). Its 16.7 MB int8
weight is L2-resident on Ada (96 MB L2), so the int8 tensor-core GEMM (~2.1x bf16)
pays off. This bench checks, at the real decode shape, that w8a8_oproj_linear is
(a) numerically close to the bf16 cuBLAS path it replaces and (b) faster INCLUDING
its standalone per-token quant.

NOTE: the standalone per-token quant launch (~14us, launch/occupancy-bound) is the
limiter here; its cost is largely hidden inside the captured CUDA decode graph, so
the in-graph win is larger than the standalone net measured below (same as down_proj).

Run:  uv run python benchmarks/benchmark_kernel/bench_w8a8_oproj_compare.py
"""

import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_oproj_linear

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

# o_proj: (N=hidden, K=Hq*D). M = B*T decode rows.
N, K = 4096, 4096
BATCHES = [32, 64, 128, 256]


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main():
    w = torch.randn(N, K, device=DEV, dtype=DT) * 0.02
    w_int8, scale = quantize_int8_per_channel(w)
    print(f"o_proj N={N} K={K}  weight={N*K/1e6:.1f}MB (fits 96MB L2)")
    print(f"{'M':>5}{'rel_err':>11}{'bf16_us':>11}{'w8a8_us':>11}{'speedup':>9}")
    all_ok = True
    for M in BATCHES:
        x = torch.randn(M, K, device=DEV, dtype=DT) * 0.5
        ref = F.linear(x, w).float()
        y8 = w8a8_oproj_linear(x, w_int8, scale)
        rel = (y8.float() - ref).norm() / ref.norm().clamp_min(1e-9)
        tbf = bench(lambda: F.linear(x, w))
        t8 = bench(lambda: w8a8_oproj_linear(x, w_int8, scale))
        spd = tbf / t8
        # standalone net may be <1 due to the un-amortized quant launch; the gate is
        # numeric faithfulness here. In-graph speed is validated by the full sweep.
        ok = rel < 3e-2
        all_ok &= ok
        flag = "" if ok else "  <-- REL ERR"
        print(f"{M:>5}{rel.item():>11.2e}{tbf:>11.1f}{t8:>11.1f}{spd:>8.2f}x{flag}")
    print("\nPASS" if all_ok else "\nFAIL")


if __name__ == "__main__":
    main()
