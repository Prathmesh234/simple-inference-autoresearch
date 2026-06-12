"""Config sweep for the fused gate_up W8A8 SwiGLU kernel at the b128 decode shape.

Shape: M=128, K=4096, I=14336 (gate_up weight is (2I, K) = (28672, 4096)).
BLOCK_N is constrained <=64 (two int32 accumulators spill registers at >=128 per
the kernel docstring). Sweep BLOCK_M/BLOCK_K/num_warps/num_stages around that.
Current production config: BM=128, BN=64, BK=128, ns=2, nw=4. MIN-of-many timing.
"""
import torch, triton
import kernels.w8a8_gemm_kernel as wk

torch.manual_seed(0)
dev = "cuda"
M, K, I = 128, 4096, 14336
two_I = 2 * I

xi = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
xs = torch.rand(M, device=dev, dtype=torch.float32) * 0.01 + 0.001
w = torch.randint(-127, 127, (two_I, K), device=dev, dtype=torch.int8)
ws = torch.rand(two_I, device=dev, dtype=torch.float32) * 0.01 + 0.001
y = torch.empty((M, I), device=dev, dtype=torch.bfloat16)


def run(BM, BN, BK, nw, ns, GM=8):
    grid = (triton.cdiv(M, BM) * triton.cdiv(I, BN),)
    wk._w8a8_swiglu_fwd[grid](
        xi, w, xs, ws, y, M, I, K,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        num_warps=nw, num_stages=ns,
    )


def bench(BM, BN, BK, nw, ns, iters=200, reps=6):
    try:
        run(BM, BN, BK, nw, ns)
    except Exception:
        return None
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            run(BM, BN, BK, nw, ns)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000


base = bench(128, 64, 128, 4, 2)
print(f"PRODUCTION BM=128 BN=64 BK=128 nw=4 ns=2:  {base:.1f}us\n")
results = []
for BM in (32, 64, 128):
    for BN in (32, 64):
        for BK in (64, 128, 256):
            for nw in (2, 4, 8):
                for ns in (2, 3, 4):
                    u = bench(BM, BN, BK, nw, ns)
                    if u is not None:
                        results.append((u, BM, BN, BK, nw, ns))
print("top 12:")
for u, BM, BN, BK, nw, ns in sorted(results)[:12]:
    tag = "  <== prod" if (BM, BN, BK, nw, ns) == (128, 64, 128, 4, 2) else ""
    print(f"  BM={BM:>3} BN={BN:>2} BK={BK:>3} nw={nw} ns={ns}  {u:>7.1f}us{tag}")
