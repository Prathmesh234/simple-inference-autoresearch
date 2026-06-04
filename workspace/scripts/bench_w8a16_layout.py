"""Test: does a transposed (K,N) int8 weight layout speed up the W8A16 GEMM?

Current kernel stores weights (N,K) and computes tl.dot(x, w.to(bf16).trans()).
The B-operand load (BN,BK) from (N,K) is strided in N. A (K,N) layout loads
(BK,BN) contiguously in N (coalesced) and feeds tl.dot(x, w) with no transpose.
Bench gate_up (N=28672,K=4096) and down (N=4096,K=14336) at M=128.
"""
import torch
import triton
import triton.language as tl
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"; dt = torch.bfloat16
M = 128


@triton.jit
def _w8a16_gemm_tn(
    x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,   # w is (K, N)
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)  # (BK,BN)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)   # (BM,BK)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0)     # (BK,BN) int8
        acc += tl.dot(x, w.to(x.dtype))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale = tl.load(s_ptr + offs_yn, mask=offs_yn < N, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    y_ptrs = (y_ptr + offs_ym[:, None].to(tl.int64) * stride_ym + offs_yn[None, :].to(tl.int64) * stride_yn)
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def run_nk(x, w8, scale, N, K, cfg):
    BM, BN, BK, ns, nw = cfg
    y = torch.empty((M, N), dtype=dt, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm[grid](x, w8, scale, y, M, N, K, x.stride(0), x.stride(1),
                      w8.stride(0), w8.stride(1), y.stride(0), y.stride(1),
                      BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, num_stages=ns, num_warps=nw)
    return y


def run_tn(x, w8t, scale, N, K, cfg):
    BM, BN, BK, ns, nw = cfg
    y = torch.empty((M, N), dtype=dt, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm_tn[grid](x, w8t, scale, y, M, N, K, x.stride(0), x.stride(1),
                         w8t.stride(0), w8t.stride(1), y.stride(0), y.stride(1),
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, num_stages=ns, num_warps=nw)
    return y


def bench(fn, iters=300, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


CASES = [("gate_up", 28672, 4096, (128, 128, 128, 2, 4)),
         ("down",    4096, 14336, (32, 128, 128, 3, 8))]
TN_CFGS = [(128, 128, 128, 2, 4), (128, 256, 128, 2, 8), (64, 128, 128, 3, 4),
           (32, 128, 128, 3, 8), (64, 64, 128, 3, 4)]

for name, N, K, nk_cfg in CASES:
    x = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt) * 0.02
    w8, scale = quantize_int8_per_channel(w)
    w8 = w8.to(dev); scale = scale.to(dev)
    w8t = w8.t().contiguous()  # (K, N)
    y_ref = (x.float() @ w8.float().T) * scale.float()

    t_nk = min(bench(lambda: run_nk(x, w8, scale, N, K, nk_cfg)) for _ in range(5))
    print(f"\n=== {name} (N={N},K={K}) M={M} ===")
    print(f"  (N,K) current cfg {nk_cfg}: {t_nk:.1f}us")
    for cfg in TN_CFGS:
        try:
            y = run_tn(x, w8t, scale, N, K, cfg)
            err = (y.float() - y_ref).abs().max().item()
            if err > 1.0:
                print(f"  (K,N) {cfg}: BAD err={err:.2f}"); continue
            t = min(bench(lambda: run_tn(x, w8t, scale, N, K, cfg)) for _ in range(5))
            print(f"  (K,N) {str(cfg):28s}: {t:7.1f}us  err={err:.3f}  speedup={t_nk/t:.2f}x")
        except Exception as ex:
            print(f"  (K,N) {cfg}: ERR {str(ex)[:60]}")
