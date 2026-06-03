"""Cheap standalone bench: fused gate_up-W8A8 + swiglu epilogue vs current
(w8a8 gate_up GEMM -> chunk -> separate swiglu kernel).

Decode shape: M=128, K=4096, I=14336 (N_combined=2I=28672). Input already int8
(EXP-D path). Measures GEMM+swiglu fused vs separate, plus correctness rel_err.
"""
import torch, triton, triton.language as tl, time

torch.manual_seed(0)
dev = "cuda"
M, K, I = 128, 4096, 14336
N = 2 * I

# random int8 activation + per-row scale, int8 weight (2I,K) + per-channel scale
xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()

from kernels.w8a8_gemm_kernel import w8a8_linear_prequant
from kernels.swiglu_kernel import swiglu_triton


@triton.jit
def _w8a8_swiglu_fwd(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, I, K,
                     sxm, sxk, swn, swk, sym, syn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """y[:, :I] = silu(gate)*up where gate=x@Wg.T, up=x@Wu.T.
    Weight rows [0,I)=gate, [I,2I)=up. Output is (M, I)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(I, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % I
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    wg_ptrs = w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk
    wu_ptrs = w_ptr + (I + offs_n)[:, None] * swn + offs_k[None, :] * swk

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(x_ptrs)
        bg = tl.load(wg_ptrs)
        bu = tl.load(wu_ptrs)
        acc_g += tl.dot(a, bg.T, out_dtype=tl.int32)
        acc_u += tl.dot(a, bu.T, out_dtype=tl.int32)
        x_ptrs += BLOCK_K * sxk
        wg_ptrs += BLOCK_K * swk
        wu_ptrs += BLOCK_K * swk

    xsc = tl.load(xs_ptr + offs_m)[:, None].to(tl.float32)
    wsg = tl.load(ws_ptr + offs_n)[None, :].to(tl.float32)
    wsu = tl.load(ws_ptr + I + offs_n)[None, :].to(tl.float32)
    gate = acc_g.to(tl.float32) * xsc * wsg
    up = acc_u.to(tl.float32) * xsc * wsu
    out = (gate * tl.sigmoid(gate)) * up

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * sym + offs_yn[None, :] * syn
    tl.store(y_ptrs, out.to(tl.bfloat16),
             mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < I))


def fused(xi, w, xs, ws, BLOCK_N=128, num_warps=4, num_stages=2):
    y = torch.empty((M, I), dtype=torch.bfloat16, device=dev)
    BLOCK_M, BLOCK_K, GROUP_M = 128, 128, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(I, BLOCK_N),)
    _w8a8_swiglu_fwd[grid](
        xi, w, xs, ws, y, M, I, K,
        xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y


def current(xi, w, xs, ws):
    comb = w8a8_linear_prequant(xi, xs, w, ws, (M, K))
    gate, up = comb.chunk(2, dim=-1)
    return swiglu_triton(gate, up)


# correctness
yf = fused(xi, w, xs, ws)
yc = current(xi, w, xs, ws)
rel = (yf.float() - yc.float()).abs() / (yc.float().abs() + 1e-4)
print(f"rel_err mean={rel.mean():.5f} max={rel.max():.5f}")


def bench(fn, *a, iters=200):
    for _ in range(20):
        fn(*a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


print(f"current (gemm+chunk+swiglu): {bench(current, xi, w, xs, ws):.1f} us")
for bn in (64, 128, 256):
    for nw in (4, 8):
        t = bench(lambda: fused(xi, w, xs, ws, BLOCK_N=bn, num_warps=nw))
        print(f"fused BLOCK_N={bn} nw={nw}: {t:.1f} us")
