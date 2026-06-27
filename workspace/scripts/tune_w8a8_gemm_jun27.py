"""jun27 EXP-3: re-sweep the _w8a8_gemm tiles for qkv / down / o_proj at their
real instruct-b128 decode shapes, hunting the same BLOCK_K/num_stages headroom
that won EXP-1 on the swiglu kernel.

Benches the PRODUCTION _w8a8_gemm kernel directly (GEMM only, int8 inputs — the
tile only affects the GEMM, not the per-token quant). MIN-of-many vs clock drift.
Current production tiles:
  qkv    : M=128 N=6144  K=4096  -> (BM128, BN64, BK128, nw8, ns3)
  down   : M=128 N=4096  K=14336 -> (BM64,  BN64, BK256, nw4, ns4)
  o_proj : M=128 N=4096  K=4096  -> (BM32,  BN64, BK256, nw4, ns2)
"""
import torch, triton, time
import kernels.w8a8_gemm_kernel as KK

torch.manual_seed(0)
dev = "cuda"
kern = KK._w8a8_gemm


def make(M, N, K):
    xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()
    return xi, xs, w, ws


def run(xi, xs, w, ws, M, N, K, BM, BN, BK, nw, ns, GM=8):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kern[grid](
        xi, w, xs, ws, y, M, N, K,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        num_stages=ns, num_warps=nw,
    )
    return y


def bench(args, cfg, iters=300, reps=5):
    xi, xs, w, ws, M, N, K = args
    try:
        run(xi, xs, w, ws, M, N, K, *cfg)
    except Exception as e:
        return None
    best = 1e9
    for _ in range(reps):
        for _ in range(15):
            run(xi, xs, w, ws, M, N, K, *cfg)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            run(xi, xs, w, ws, M, N, K, *cfg)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


SHAPES = {
    "qkv   (N=6144,K=4096) ": (128, 6144, 4096, (128, 64, 128, 8, 3)),
    "down  (N=4096,K=14336)": (128, 4096, 14336, (64, 64, 256, 4, 4)),
    "oproj (N=4096,K=4096) ": (128, 4096, 4096, (32, 64, 256, 4, 2)),
}

# candidate tiles to try on each shape (BM,BN,BK,nw,ns)
CANDS = [
    (128, 64, 128, 4, 3), (128, 64, 128, 8, 3), (128, 64, 256, 8, 2),
    (128, 64, 256, 4, 2), (64, 64, 256, 4, 4), (64, 64, 256, 8, 3),
    (64, 64, 128, 4, 4), (64, 64, 128, 8, 3), (32, 64, 256, 4, 2),
    (32, 64, 256, 8, 2), (32, 64, 128, 4, 4), (128, 32, 256, 8, 2),
    (64, 128, 128, 8, 3), (128, 128, 128, 8, 2), (64, 32, 256, 4, 3),
]

for name, (M, N, K, cur) in SHAPES.items():
    args = (*make(M, N, K), M, N, K)
    cur_t = bench(args, cur)
    cands = CANDS if cur in CANDS else [cur] + CANDS
    res = []
    for c in cands:
        t = bench(args, c)
        if t is not None:
            res.append((t, c))
    res.sort()
    print(f"\n=== {name}  current {cur} = {cur_t:.1f}us ===")
    for t, c in res[:5]:
        tag = "  <-- current" if c == cur else f"  {cur_t/t:.3f}x"
        print(f"  {str(c):<26} {t:>7.1f}us{tag}")
