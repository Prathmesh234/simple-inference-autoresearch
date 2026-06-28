"""jun28 EXP-14 probe: can a W4A8 int4 gate GEMM beat W8A8 at the gate shape, using
BLOCK_K=256 that handles 4 groups (g=64) per iteration (group=64 otherwise forces
BLOCK_K=64 -> the EXP-13 inefficiency wall that likely sank EXP-O's 175us)?

Shape M=128, N=14336, K=4096, group=64 along K. int4 weight packs 2 nibbles/byte
(29MB vs int8 58.7MB). Activation int8 per-row. Correctness vs an int4-dequant ref +
speed vs the production int8 _w8a8_gemm. int4 < int8 here => build fused gate_up int4.
"""
import torch, triton, triton.language as tl, time
import kernels.w8a8_gemm_kernel as KK

torch.manual_seed(0); dev = "cuda"
M, N, K = 128, 14336, 4096
GSZ = 64
NG = K // GSZ


@triton.jit
def _w4a8_gemm(x_ptr, wp_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
               sxm, sxk, swn, swk, swsn, swsg, sym, syn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GSZ: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    npg = GROUP_M * num_pid_n; gid = pid // npg
    fpm = gid * GROUP_M; gsm = min(num_pid_m - fpm, GROUP_M)
    pid_m = fpm + ((pid % npg) % gsm); pid_n = (pid % npg) // gsm
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    GPK: tl.constexpr = BLOCK_K // GSZ
    HG: tl.constexpr = GSZ // 2
    offs_g = tl.arange(0, GSZ)
    offs_hg = tl.arange(0, HG)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_groups = K // GSZ
    for g_idx in range(0, n_groups):
        kbase = g_idx * GSZ
        asub = tl.load(x_ptr + offs_m[:, None] * sxm + (kbase + offs_g)[None, :] * sxk)  # [BM,GSZ] int8
        wp = tl.load(wp_ptr + offs_n[:, None] * swn + (kbase // 2 + offs_hg)[None, :] * swk)  # [BN,HG] uint8
        lo = ((wp << 4).to(tl.int8) >> 4)
        hi = (wp.to(tl.int8) >> 4)
        wsub = tl.interleave(lo, hi)  # [BN, GSZ] int8
        p = tl.dot(asub, wsub.T, out_dtype=tl.int32)
        wsg = tl.load(ws_ptr + offs_n * swsn + g_idx * swsg)[None, :].to(tl.float32)
        acc += p.to(tl.float32) * wsg
    xs = tl.load(xs_ptr + offs_m)[:, None].to(tl.float32)
    y = acc * xs
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn, y.to(tl.bfloat16),
             mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))


xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
xsc = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
wq = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
wsg = (torch.rand(N, NG, device=dev) * 0.001 + 1e-4).float()
wpk = ((wq[:, 0::2] & 0xF) | ((wq[:, 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()
w8w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
ws8 = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()


def run4(BM, BN, BK, nw, ns):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w4a8_gemm[grid](xi, wpk, xsc, wsg, y, M, N, K,
        xi.stride(0), xi.stride(1), wpk.stride(0), wpk.stride(1),
        wsg.stride(0), wsg.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GSZ=GSZ, GROUP_M=8, num_warps=nw, num_stages=ns)
    return y


def run8(BM, BN, BK, nw, ns):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    KK._w8a8_gemm[grid](xi, w8w, xsc, ws8, y, M, N, K,
        xi.stride(0), xi.stride(1), w8w.stride(0), w8w.stride(1),
        y.stride(0), y.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
        num_warps=nw, num_stages=ns)
    return y


def bench(fn, iters=300, reps=5):
    try: fn()
    except Exception as e: return None, str(e)[:90]
    best = 1e9
    for _ in range(reps):
        for _ in range(15): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter()-t)/iters*1e6)
    return best, None


# correctness vs int4-dequant reference
wsc_full = torch.repeat_interleave(wsg, GSZ, dim=1)        # [N,K]
wdq = wq.float() * wsc_full
yr = (xi.float() * xsc[:, None]) @ wdq.T
y4 = run4(128, 64, 256, 8, 3)
rel = ((y4.float() - yr).abs() / (yr.abs() + 1e-2)).mean().item()
print(f"int4 kernel rel_err vs dequant ref: {rel:.4f}  (should be ~0)")

tb, _ = bench(lambda: run8(64, 64, 256, 4, 4))
print(f"int8 W8A8 (per-row 64/64/256/4/4): {tb:.1f} us  (HBM floor ~61us, int4 floor ~31us)")
print("int4 W4A8 (group=64, BLOCK_K spans BLOCK_K/64 groups):")
for BM, BN, BK, nw, ns in [(128,64,256,8,3),(128,64,256,4,3),(64,64,256,4,4),
                           (128,64,128,8,3),(128,128,256,8,2),(64,128,256,8,3),(128,64,512,8,2)]:
    t, err = bench(lambda: run4(BM, BN, BK, nw, ns))
    print(f"  {BM}/{BN}/{BK}/{nw}/{ns}: " + (f"{t:.1f} us  ({tb/t:.2f}x vs int8)" if t else f"FAIL {err}"))
