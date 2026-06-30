"""jun30: re-sweep the gate_up W8A8+SwiGLU fused kernel on THIS box, with the
ACTUAL production tile (128,64,256,8,2) as the baseline. gate_up is 38% of decode
(153.7us/call here) and HBM-bound (floor ~122us). Confirm the production tile is
still optimal here and probe a few untried configs (BK=512, GROUP_M, BM=64).
MIN-of-many to defeat clock drift; bit-identical output required."""
import torch, triton, time
import kernels.w8a8_gemm_kernel as K

torch.manual_seed(0)
dev = "cuda"
M, KK, I = 128, 4096, 14336
N = 2 * I
xi = torch.randint(-127, 128, (M, KK), dtype=torch.int8, device=dev)
xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
w = torch.randint(-127, 128, (N, KK), dtype=torch.int8, device=dev)
ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()
kern = K._w8a8_swiglu_fwd


def run(BM, BN, BK, nw, ns, GROUP_M=8):
    y = torch.empty((M, I), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(I, BN),)
    kern[grid](xi, w, xs, ws, y, M, I, KK,
               xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
               y.stride(0), y.stride(1),
               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GROUP_M,
               num_warps=nw, num_stages=ns)
    return y


def bench(cfg, iters=300, reps=6):
    try:
        run(*cfg)
    except Exception as e:
        return None, str(e)[:50]
    best = 1e9
    for _ in range(reps):
        for _ in range(15):
            run(*cfg)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            run(*cfg)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best, None


CUR = (128, 64, 256, 8, 2)  # ACTUAL production config (EXP-1 jun27)
cands = [
    CUR,
    (128, 64, 128, 8, 2), (128, 64, 128, 8, 3), (128, 64, 128, 8, 4),
    (128, 64, 256, 4, 2), (128, 64, 512, 8, 2), (128, 64, 512, 4, 2),
    (128, 32, 256, 8, 2), (128, 32, 256, 8, 3), (128, 32, 512, 8, 2),
    (64, 64, 256, 8, 2), (64, 64, 256, 4, 4), (64, 64, 512, 4, 2),
    (128, 64, 256, 8, 2, 4), (128, 64, 256, 8, 2, 16),
    (256, 64, 256, 8, 2),
]
ref = run(*CUR).float()
print(f"HBM floor ~122us. baseline CUR={CUR}\n{'config':<30} {'us':>8} {'vs cur':>8}  rel_err")
results = []
cur_t = None
for c in cands:
    t, err = bench(c)
    if t is None:
        print(f"{str(c):<30} {'FAIL':>8}  {err}"); continue
    rel = ((run(*c).float() - ref).abs() / (ref.abs() + 1e-4)).mean().item()
    if c == CUR:
        cur_t = t
    results.append((t, c, rel))
    sp = f"{cur_t/t:.3f}x" if cur_t else "--"
    print(f"{str(c):<30} {t:>8.1f} {sp:>8}  {rel:.5f}")
results.sort()
print("\n--- best 5 ---")
for t, c, rel in results[:5]:
    print(f"{str(c):<30} {t:>8.1f}us  {(cur_t/t if cur_t else 0):.3f}x  rel={rel:.5f}")
