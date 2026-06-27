"""jun27: thorough config sweep of the gate_up W8A8+SwiGLU fused kernel at the
real instruct-b128 decode shape (M=128, K=4096, I=14336, N=2I=28672).

gate_up is the single biggest decode op (167.5us/call in-graph, 39% of decode).
Its 117MB int8 weight exceeds the 96MB L2 -> HBM-bound, floor = 117MB/960GBs =
~122us. We're ~45us above floor => un-hidden compute / pipeline bubbles. Prior
sessions swept BLOCK_N/num_warps ("sub-noise, BLOCK_N pinned 64") but num_stages /
BLOCK_K look under-explored. Hypothesis: deeper pipelining hides the int8 MMA
under the weight HBM load and approaches the 122us floor.

Uses the PRODUCTION kernel (import, not a copy) so the result is directly valid.
MIN-of-many to defeat clock drift. Correctness vs current config checked.
"""
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


def run(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M=8):
    y = torch.empty((M, I), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(I, BLOCK_N),)
    kern[grid](
        xi, w, xs, ws, y, M, I, KK,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def bench(cfg, iters=300, reps=5):
    try:
        run(*cfg)
    except Exception as e:
        return None, str(e)[:60]
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


# (BM, BN, BK, nw, ns)
CUR = (128, 64, 128, 4, 2)  # current production config
cands = [
    CUR,
    (128, 64, 128, 4, 3),
    (128, 64, 128, 4, 4),
    (128, 64, 128, 8, 2),
    (128, 64, 128, 8, 3),
    (128, 64, 128, 8, 4),
    (128, 64, 256, 4, 2),
    (128, 64, 256, 4, 3),
    (128, 64, 256, 8, 2),
    (128, 64, 256, 8, 3),
    (128, 32, 128, 4, 3),
    (128, 32, 128, 4, 4),
    (128, 32, 256, 8, 3),
    (64, 64, 128, 4, 3),
    (64, 64, 128, 4, 4),
    (64, 64, 256, 4, 4),
    (128, 128, 128, 8, 2),  # known to spill, confirm
]

# correctness ref from current config
ref = run(*CUR).float()
print(f"HBM floor (117MB/960GBps) ~= 122 us\n")
print(f"{'config (BM,BN,BK,nw,ns)':<28} {'us':>8}  {'vs cur':>7}  rel_err")
cur_t = None
results = []
for c in cands:
    t, err = bench(c)
    if t is None:
        print(f"{str(c):<28} {'FAIL':>8}  {err}")
        continue
    y = run(*c).float()
    rel = ((y - ref).abs() / (ref.abs() + 1e-4)).mean().item()
    if c == CUR:
        cur_t = t
    results.append((t, c, rel))
    print(f"{str(c):<28} {t:>8.1f}", end="")
    if cur_t:
        print(f"  {cur_t/t:>6.3f}x  {rel:.5f}")
    else:
        print(f"  {'--':>7}  {rel:.5f}")

results.sort()
print("\n--- best 5 ---")
for t, c, rel in results[:5]:
    sp = (cur_t / t) if cur_t else 0
    print(f"{str(c):<28} {t:>8.1f} us  {sp:.3f}x  rel_err={rel:.5f}")
