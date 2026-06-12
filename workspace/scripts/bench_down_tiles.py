"""Standalone microbench: down_proj W8A16 GEMM tile sweep.

down shape at b128 decode: M=128, N=4096, K=14336, int8 weight (per-out-channel
scale) x bf16 activation -> bf16. Current production tile is (BLOCK_M=32,
BLOCK_N=128, BLOCK_K=256). BLOCK_M=32 re-reads the 58MB int8 weight M/32=4x;
under the full decode graph that 4x hits DRAM (L2 contention). BLOCK_M=128 reads
it 1x but needs enough N-blocks (small BLOCK_N) for SM occupancy.

MIN-of-many timing (clock-noise-free). Standalone CANNOT reproduce the in-graph
L2 contention, so this only rules out compute-broken tiles; the real test is the
full profiler.
"""
import torch, triton
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"
M, N, K = 128, 4096, 14336
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
w_int8, scale = quantize_int8_per_channel(w)
y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

ref = (x.float() @ (w_int8.float() * scale[:, None]).T)


def run(BM, BN, BK, ns, nw, GM=8):
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm[grid](
        x, w_int8, scale, y, M, N, K,
        x.stride(0), x.stride(1), w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        num_stages=ns, num_warps=nw,
    )


def bench(BM, BN, BK, ns, nw, iters=200, reps=5):
    try:
        run(BM, BN, BK, ns, nw)
    except Exception as e:
        return None, str(e)[:60]
    err = (y.float() - ref).abs().max().item()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            run(BM, BN, BK, ns, nw)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000, err  # us


configs = [
    # production
    (32, 128, 256, 3, 8),
    (32, 128, 128, 3, 8),
    (64, 64, 128, 3, 8),
    # BLOCK_M=128 -> 1x weight read, vary BLOCK_N for occupancy (N/BN programs)
    (128, 32, 128, 3, 8),
    (128, 32, 256, 3, 8),
    (128, 32, 128, 4, 8),
    (128, 64, 128, 3, 8),
    (128, 64, 256, 3, 8),
    (128, 64, 128, 4, 8),
    (128, 32, 128, 2, 4),
    (128, 64, 128, 2, 4),
    (128, 32, 64, 4, 8),
    (64, 32, 128, 3, 8),  # 2x weight, 256 progs
    (64, 32, 256, 3, 8),
]
print(f"{'BM':>4} {'BN':>4} {'BK':>4} {'st':>3} {'nw':>3}  {'progs':>5}  {'us':>8}  maxerr")
for BM, BN, BK, ns, nw in configs:
    progs = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    t, err = bench(BM, BN, BK, ns, nw)
    if t is None:
        print(f"{BM:>4} {BN:>4} {BK:>4} {ns:>3} {nw:>3}  {progs:>5}  FAILED: {err}")
    else:
        print(f"{BM:>4} {BN:>4} {BK:>4} {ns:>3} {nw:>3}  {progs:>5}  {t:>8.2f}  {err:.4f}")
