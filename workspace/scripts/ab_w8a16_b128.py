"""Focused A/B: current vs candidate W8A16 configs at M=128 (pre-allocated, median)."""
import torch
import triton
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"; dt = torch.bfloat16
M = 128

CASES = {
    "gate_up": dict(N=28672, K=4096,
                    current=(128, 128, 128, 2, 4),
                    cands=[(128, 256, 128, 2, 8), (64, 256, 128, 2, 4), (128, 256, 128, 2, 4)]),
    "down": dict(N=4096, K=14336,
                 current=(32, 128, 128, 3, 8),
                 cands=[(32, 128, 128, 3, 4), (64, 64, 128, 3, 4)]),
}


def make(N, K):
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt) * 0.02
    w_int8, scale = quantize_int8_per_channel(w)
    return x, w_int8.to(dev), scale.to(dev), torch.empty((M, N), dtype=dt, device=dev)


def run(x, w_int8, scale, y, N, K, cfg):
    BM, BN, BK, ns, nw = cfg
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm[grid](x, w_int8, scale, y, M, N, K,
                      x.stride(0), x.stride(1), w_int8.stride(0), w_int8.stride(1),
                      y.stride(0), y.stride(1),
                      BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
                      num_stages=ns, num_warps=nw)


def bench(x, w_int8, scale, y, N, K, cfg, iters=300, warmup=80):
    for _ in range(warmup):
        run(x, w_int8, scale, y, N, K, cfg)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run(x, w_int8, scale, y, N, K, cfg)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


for name, c in CASES.items():
    N, K = c["N"], c["K"]
    x, w_int8, scale, y = make(N, K)
    print(f"\n=== {name} (N={N},K={K}) M={M} ===")
    for label, cfg in [("current", c["current"])] + [(f"cand{i}", cg) for i, cg in enumerate(c["cands"])]:
        ts = sorted(bench(x, w_int8, scale, y, N, K, cfg) for _ in range(7))
        print(f"  {label:8s} {str(cfg):28s} median={ts[3]:7.1f}us  min={ts[0]:7.1f}us")
