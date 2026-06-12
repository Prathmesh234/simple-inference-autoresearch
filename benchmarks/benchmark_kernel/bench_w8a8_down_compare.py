"""
Standalone gate for the W8A8 down-projection GEMM kernel (w8a8_down_linear).

down is the dominant decode GEMM (M=128, K=14336 intermediate, N=4096 hidden).
Its 58.7 MB int8 weight fits Ada's 96 MB L2, so the GEMM is compute/L2-bound and
int8 tensor cores (~2x bf16) pay off. This bench checks, at the real decode shape,
that w8a8_down_linear is (a) numerically faithful to the W8A16 weight-only path it
replaces and (b) faster than W8A16 INCLUDING its standalone per-token quant.

Run:  uv run python benchmarks/benchmark_kernel/bench_w8a8_down_compare.py
"""

import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_down_linear

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

# down: (N=hidden, K=intermediate). M = B*T decode rows.
N, K = 4096, 14336
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
    ref_full = (w_int8.float() * scale[:, None])  # the int8-weight reference
    print(f"down N={N} K={K}  weight={N*K/1e6:.1f}MB (fits 96MB L2)")
    print(f"{'M':>5}{'rel_err':>11}{'w8a16_us':>11}{'w8a8_us':>11}{'speedup':>9}")
    all_ok = True
    for M in BATCHES:
        x = torch.randn(M, K, device=DEV, dtype=DT) * 0.5
        ref = x.float() @ ref_full.T
        y8 = w8a8_down_linear(x, w_int8, scale)
        rel = (y8.float() - ref).norm() / ref.norm().clamp_min(1e-9)
        t16 = bench(lambda: w8a16_linear_triton(x, w_int8, scale))
        t8 = bench(lambda: w8a8_down_linear(x, w_int8, scale))
        spd = t16 / t8
        ok = (rel < 3e-2) and (spd > 1.0)
        all_ok &= ok
        flag = "" if ok else ("  <-- REL ERR" if rel >= 3e-2 else "  <-- SLOWER")
        print(f"{M:>5}{rel.item():>11.2e}{t16:>11.1f}{t8:>11.1f}{spd:>8.2f}x{flag}")
    print("\nPASS" if all_ok else "\nFAIL")


if __name__ == "__main__":
    main()
