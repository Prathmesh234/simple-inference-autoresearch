"""Roofline check: does total decode GEMM time grow sub-linearly with batch M?
If so, agg = M*1000/decode_ms climbs -> higher batch is a frontier win."""
import torch
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)

# Per-layer int8 GEMMs that stream weights once (the bandwidth-bound part of decode).
# gate_up: N=28672 K=4096 ; down: N=4096 K=14336 ; qkv(EXP-B): N=6144 K=4096 ; o: bf16(skip)
SHAPES = {"gate_up": (28672, 4096), "down": (4096, 14336), "qkv": (6144, 4096)}
NLAYERS = 32

W = {}
for name, (N, K) in SHAPES.items():
    w = (torch.randn(N, K, device=dev, dtype=dt) * 0.02)
    wi, sc = quantize_int8_per_channel(w)
    W[name] = (wi.to(dev), sc.to(dev), K, N)

def bench_M(M, iters=100, warmup=30):
    xs = {n: torch.randn(M, K, device=dev, dtype=dt) for n, (_, _, K, _) in W.items()}
    def step():
        for n, (wi, sc, K, N) in W.items():
            w8a16_linear_triton(xs[n], wi, sc)
    for _ in range(warmup): step()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): step()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us for ONE layer's 3 GEMMs

print(f"{'M':>5} {'us/layer':>9} {'us*32layers':>11} {'rel_time':>8} {'rel_M':>6} {'~agg_proxy':>10}")
base = None
for M in [128, 160, 192, 224, 256]:
    t = sorted(bench_M(M) for _ in range(7))[3]  # median
    if base is None: base = (t, M)
    rel_t = t / base[0]; rel_M = M / base[1]
    agg_proxy = rel_M / rel_t  # how much agg scales vs b128 (if GEMM-dominated)
    print(f"{M:>5} {t:9.1f} {t*NLAYERS/1000:11.3f} {rel_t:8.3f} {rel_M:6.3f} {agg_proxy:10.3f}")
