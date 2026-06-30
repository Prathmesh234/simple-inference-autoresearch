"""jun30: re-tune the W8A16 GEMM at M=1 (b1 decode) — 82.5% of b1 decode, at 1.20x
its weight-read floor. The b1 tile (16,128,128,4,8) was tuned offline; probe whether
a better tile gets closer to the read floor. Per-shape (gate_up/down/qkv/lm_head),
MIN-of-many, bit-identical output required. Calls the production _w8a16_gemm."""
import torch, triton, time
import kernels.w8a16_gemm_kernel as W

torch.manual_seed(0); dev = "cuda"
kern = W._w8a16_gemm
SHAPES = [("gate_up", 1, 28672, 4096), ("down", 1, 4096, 14336),
          ("qkv", 1, 6144, 4096), ("lm_head", 1, 128256, 4096)]


def run(x, w, s, M, N, K, BM, BN, BK, nw, ns, GM=8):
    y = torch.empty((M, N), dtype=x.dtype, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    kern[grid](x, w, s, y, M, N, K, x.stride(0), x.stride(1),
               w.stride(0), w.stride(1), y.stride(0), y.stride(1),
               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, num_warps=nw, num_stages=ns)
    return y


def bench(fn, iters=400, reps=6):
    try: fn()
    except Exception as e: return None, str(e)[:40]
    best = 1e9
    for _ in range(reps):
        for _ in range(20): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter()-t)/iters*1e6)
    return best, None


# (BM, BN, BK, nw, ns) — production b1 is nw=8, ns=4
CUR = {"gate_up": (16,128,128,8,4), "down": (16,64,128,8,4),
       "qkv": (16,128,128,8,4), "lm_head": (16,128,128,8,4)}
cands = [
    (16,128,128,8,4),(16,256,128,8,4),(16,512,128,8,4),(16,128,256,8,3),
    (16,256,256,8,2),(16,128,128,8,3),(16,128,128,4,4),(16,256,128,4,4),
    (16,512,64,8,4),(16,1024,64,8,4),(16,256,64,8,4),(16,64,128,8,4),
    (16,128,256,8,2),(16,512,128,8,3),(16,256,128,16,4),
]
for name, M, N, K in SHAPES:
    floor = N * K / 960e9 * 1e6  # int8 weight read floor (us)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    s = (torch.rand(N, device=dev) * 0.01 + 1e-3).float()
    cur_cfg = CUR[name]
    ref = run(x, w, s, M, N, K, *cur_cfg).float()
    cur_t, _ = bench(lambda: run(x, w, s, M, N, K, *cur_cfg))
    res = []
    for c in cands:
        t, err = bench(lambda: run(x, w, s, M, N, K, *c))
        if t: res.append((t, c))
    res.sort()
    print(f"\n=== {name} M={M} N={N} K={K}  floor~{floor:.0f}us  cur={cur_cfg}={cur_t:.1f}us ({cur_t/floor:.2f}x) ===")
    for t, c in res[:4]:
        tag = " <-cur" if c == cur_cfg else f"  {cur_t/t:.3f}x"
        print(f"   {str(c):<22} {t:>7.1f}us{tag}")
