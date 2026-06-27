"""jun27 EXP-6: split-K for the W8A8 decode GEMMs (o_proj, down).

o_proj (M=128,N=4096,K=4096) and down (M=128,N=4096,K=14336) are W8A8 and sit far
above their int8 compute floors (o_proj 32us vs ~6us floor; down 48us vs ~20us):
they are occupancy/latency-bound — with BLOCK_M=64,BLOCK_N=64 only ~128 programs
launch on 142 SMs (~1 wave), so the serial K-loop latency isn't hidden. int8 MMA
(2x faster compute than the old W8A16 these were split-K-tested on) makes the K-loop
latency an even bigger fraction => split-K may finally pay. Split the K reduction
across SPLIT_K programs, each accumulating int32 over its K-chunk and atomic-adding
the SCALED fp32 partial (scale is constant across splits, so sum-of-scaled-partials
== scaled-total). MIN-of-many, correctness vs the production kernel.
"""
import torch, triton, triton.language as tl, time
import kernels.w8a8_gemm_kernel as KK

torch.manual_seed(0)
dev = "cuda"


@triton.jit
def _w8a8_gemm_splitk(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
                      sxm, sxk, swn, swk, sym, syn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
    """Split-K W8A8: grid (num_pid_m*num_pid_n*SPLIT_K,). Each program reduces
    K-chunk [sk*KS,(sk+1)*KS) and atomic-adds the scaled fp32 partial to y (fp32)."""
    pid = tl.program_id(0)
    sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    KS = tl.cdiv(K, SPLIT_K)
    k_start = sk * KS
    offs_k = k_start + tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for kk in range(0, tl.cdiv(KS, BLOCK_K)):
        kmask = (k_start + kk * BLOCK_K + tl.arange(0, BLOCK_K)) < K
        a = tl.load(x_ptrs, mask=kmask[None, :], other=0)
        b = tl.load(w_ptrs, mask=kmask[None, :], other=0)
        acc += tl.dot(a, b.T, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    xs = tl.load(xs_ptr + offs_m)[:, None].to(tl.float32)
    ws = tl.load(ws_ptr + offs_n)[None, :].to(tl.float32)
    y = acc.to(tl.float32) * xs * ws
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn
    tl.atomic_add(y_ptrs, y, mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))


def splitk(xi, w, xs, ws, M, N, K, BM, BN, BK, SK, nw, ns, GM=8):
    y = torch.zeros((M, N), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN) * SK,)
    _w8a8_gemm_splitk[grid](xi, w, xs, ws, y, M, N, K,
                            xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
                            y.stride(0), y.stride(1),
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK, GROUP_M=GM,
                            num_stages=ns, num_warps=nw)
    return y.to(torch.bfloat16)


def cur(xi, w, xs, ws, M, N, K, BM, BN, BK, nw, ns, GM=8):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    KK._w8a8_gemm[grid](xi, w, xs, ws, y, M, N, K,
                        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
                        y.stride(0), y.stride(1),
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                        num_stages=ns, num_warps=nw)
    return y


def bench(fn, iters=300, reps=6):
    try:
        fn()
    except Exception as e:
        return None, str(e)[:50]
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
    return best, None


SHAPES = {
    "o_proj (N=4096,K=4096) ": (128, 4096, 4096, (32, 64, 256, 4, 2)),
    "down   (N=4096,K=14336)": (128, 4096, 14336, (64, 64, 256, 4, 4)),
}
for name, (M, N, K, curcfg) in SHAPES.items():
    xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
    w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()
    ref = cur(xi, w, xs, ws, M, N, K, *curcfg).float()
    cur_t, _ = bench(lambda: cur(xi, w, xs, ws, M, N, K, *curcfg))
    print(f"\n=== {name} current {curcfg} = {cur_t:.1f}us ===")
    res = []
    for BM, BN, BK, SK, nw, ns in [
        (64, 64, 128, 2, 4, 3), (64, 64, 128, 4, 4, 3), (64, 64, 256, 2, 4, 3),
        (64, 64, 256, 4, 4, 3), (32, 64, 128, 4, 4, 2), (32, 64, 256, 2, 4, 2),
        (128, 64, 128, 4, 8, 3), (64, 128, 128, 2, 8, 3), (64, 64, 128, 8, 4, 2),
    ]:
        t, err = bench(lambda: splitk(xi, w, xs, ws, M, N, K, BM, BN, BK, SK, nw, ns))
        if t is None:
            continue
        y = splitk(xi, w, xs, ws, M, N, K, BM, BN, BK, SK, nw, ns).float()
        rel = ((y - ref).abs() / (ref.abs() + 1e-3)).mean().item()
        res.append((t, (BM, BN, BK, SK, nw, ns), rel))
    res.sort()
    for t, c, rel in res[:5]:
        print(f"  splitk {str(c):<24} {t:>7.1f}us  {cur_t/t:.3f}x  rel={rel:.5f}")
