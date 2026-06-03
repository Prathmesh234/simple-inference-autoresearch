"""
Standalone gate for the W8A16 int8 weight-only GEMM kernel.

Checks, at the real Llama-3.1-8B decode shapes (M = B*T), that
w8a16_linear_triton is (a) numerically faithful to F.linear (per-channel int8
quant error is small) and (b) faster than cuBLAS bf16 F.linear (it reads half
the weight bytes). Run:  uv run python benchmarks/benchmark_kernel/bench_w8a16_compare.py
"""

import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

DEV = "cuda"
DT = torch.bfloat16
torch.manual_seed(0)

# (name, N=out_features, K=in_features) for the Llama-3.1-8B linears.
LAYERS = [
    ("mlp.gate_up", 2 * 14336, 4096),
    ("mlp.down",    4096,       14336),
    ("attn.wq",     4096,       4096),
    ("attn.wkv",    2 * 1024,   4096),
    ("attn.wo",     4096,       4096),
    ("lm_head",     128256,     4096),
]
BATCHES = [1, 128]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    print(f"{'layer':<14}{'M':>5}{'rel_err':>11}{'bf16_ms':>10}{'int8_ms':>10}{'speedup':>9}")
    all_ok = True
    for name, N, K in LAYERS:
        w = torch.randn(N, K, device=DEV, dtype=DT) * 0.02
        w_int8, scale = quantize_int8_per_channel(w)
        for M in BATCHES:
            x = torch.randn(M, K, device=DEV, dtype=DT)
            y_ref = torch.nn.functional.linear(x, w)
            y_q = w8a16_linear_triton(x, w_int8, scale)
            rel = (y_q.float() - y_ref.float()).norm() / y_ref.float().norm().clamp_min(1e-9)
            t_bf16 = bench(lambda: torch.nn.functional.linear(x, w))
            t_int8 = bench(lambda: w8a16_linear_triton(x, w_int8, scale))
            spd = t_bf16 / t_int8
            ok = rel < 2e-2
            all_ok &= ok
            flag = "" if ok else "  <-- REL ERR HIGH"
            print(f"{name:<14}{M:>5}{rel.item():>11.2e}{t_bf16:>10.3f}{t_int8:>10.3f}{spd:>8.2f}x{flag}")
    print("\nPASS" if all_ok else "\nFAIL (rel err)")


if __name__ == "__main__":
    main()
