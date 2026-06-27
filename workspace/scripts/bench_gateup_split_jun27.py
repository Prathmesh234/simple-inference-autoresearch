"""jun27: is the FUSED gate_up+swiglu (1 kernel, 2 int32 accumulators, BLOCK_N
pinned 64) actually optimal, or do 2 single-accumulator GEMMs (BLOCK_N free to
128) + a separate swiglu win despite the extra intermediate traffic?

gate_up is HBM-bound (117MB int8 weight); both variants read the same 117MB. The
question is tile efficiency (BN=64 double-acc vs BN=128 single-acc) vs the extra
~18MB intermediate traffic the split pays. Decode shape M=128, K=4096, I=14336.
MIN-of-many.
"""
import torch, triton, time
import kernels.w8a8_gemm_kernel as KK
from kernels.swiglu_kernel import swiglu_triton

torch.manual_seed(0)
dev = "cuda"
M, K, I = 128, 4096, 14336
two_I = 2 * I

xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
wgu = torch.randint(-127, 128, (two_I, K), dtype=torch.int8, device=dev)   # gate rows[0:I], up rows[I:2I]
wgu_s = (torch.rand(two_I, device=dev) * 0.001 + 1e-4).float()

gemm = KK._w8a8_gemm


def single_gemm(w, ws, N, BM, BN, BK, nw, ns, GM=8):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    gemm[grid](xi, w, xs, ws, y, M, N, K,
               xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
               y.stride(0), y.stride(1),
               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
               num_stages=ns, num_warps=nw)
    return y


def split_path(cfg):
    gate = single_gemm(wgu[:I], wgu_s[:I], I, *cfg)
    up = single_gemm(wgu[I:], wgu_s[I:], I, *cfg)
    return swiglu_triton(gate, up)


def fused_path():
    # the shipped production config (BLOCK_K=256, nw=8)
    return KK.w8a8_swiglu_prequant(xi, xs, wgu, wgu_s, (M, K))


def bench(fn, iters=300, reps=6):
    try:
        fn()
    except Exception:
        return None
    best = 1e9
    for _ in range(reps):
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


tf = bench(fused_path)
print(f"FUSED (shipped BK256/nw8):  {tf:.1f}us")
print("split = 2x single-acc GEMM + swiglu_triton:")
for cfg in [(128, 128, 128, 8, 3), (128, 128, 256, 8, 2), (128, 256, 128, 8, 3),
            (128, 128, 128, 4, 3), (128, 256, 256, 8, 2), (128, 128, 256, 4, 2),
            (64, 128, 256, 8, 2)]:
    t = bench(lambda: split_path(cfg))
    if t:
        print(f"  split {str(cfg):<22} {t:>7.1f}us  fused/split={tf/t:.3f}x")
