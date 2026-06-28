"""jun28 EXP-13 probe: grouped-K int8 GEMM at the down shape (M=128,N=4096,K=14336)
with a per-(row, K-group) activation scale (group=64, aligned to the swiglu BLOCK_N).

If this is within ~1.2x of the per-row down GEMM (48us), then fusing the down
activation quant INTO the swiglu epilogue (each swiglu BLOCK_N=64 tile emits its
group's int8 + scale) is net positive: it removes the swiglu bf16 write (3.67MB) +
the standalone _quant_per_token (read 3.67MB + write 1.8MB + launch) per layer
(~234MB/step). Grouped int8 rescale is cheap (no int4 unpack — the EXP-O blocker).
Per-group quant is also FINER than per-row -> at least as faithful.
"""
import torch, triton, triton.language as tl, time
import kernels.w8a8_gemm_kernel as KK

torch.manual_seed(0); dev = "cuda"
M, N, K = 128, 4096, 14336
GROUP = 64
NG = K // GROUP  # 224 groups


@triton.jit
def _w8a8_gemm_groupk(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
                      sxm, sxk, swn, swk, sxsm, sxsg, sym, syn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_K: tl.constexpr,
                      GROUP_M: tl.constexpr):
    """y = sum_g (x_g @ w_g.T)*act_scale[m,g] * w_scale[n].  BLOCK_K==GROUP_K
    (one K-group per iteration), so each iter rescales its int32 partial by the
    group's per-row act scale and accumulates into fp32."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_groups = tl.cdiv(K, BLOCK_K)
    for g in range(0, n_groups):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        p = tl.dot(a, b.T, out_dtype=tl.int32)
        xs = tl.load(xs_ptr + offs_m * sxsm + g * sxsg)[:, None].to(tl.float32)
        acc += p.to(tl.float32) * xs
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    ws = tl.load(ws_ptr + offs_n)[None, :].to(tl.float32)
    y = acc * ws
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn
    tl.store(y_ptrs, y.to(tl.bfloat16),
             mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))


xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
xsg = (torch.rand(M, NG, device=dev) * 0.01 + 1e-3).float()   # per-(row,group) scale
w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()
# per-row scale for the current kernel comparison
xs_row = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()


def grouped(BM, BN, nw, ns):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a8_gemm_groupk[grid](xi, w, xsg, ws, y, M, N, K,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        xsg.stride(0), xsg.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=GROUP, GROUP_K=GROUP, GROUP_M=8,
        num_warps=nw, num_stages=ns)
    return y


def perrow(BM, BN, BK, nw, ns):  # current down kernel (per-row scale)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    KK._w8a8_gemm[grid](xi, w, xs_row, ws, y, M, N, K,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, num_warps=nw, num_stages=ns)
    return y


def bench(fn, iters=300, reps=5):
    try: fn()
    except Exception as e: return None
    best=1e9
    for _ in range(reps):
        for _ in range(15): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/iters*1e6)
    return best

print(f"per-row down (current 64,64,256,4,4): {bench(lambda: perrow(64,64,256,4,4)):.1f} us")
print("grouped-K (group=64, BLOCK_K=64):")
for BM,BN,nw,ns in [(64,64,4,4),(64,64,4,3),(64,128,4,3),(128,64,4,3),(64,64,8,3),(32,64,4,4),(64,128,8,2)]:
    t=bench(lambda: grouped(BM,BN,nw,ns))
    if t: print(f"  BM={BM} BN={BN} nw={nw} ns={ns}: {t:.1f} us  ({bench(lambda: perrow(64,64,256,4,4))/t:.2f}x vs per-row)")
