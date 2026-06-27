"""jun27 EXP-4 probes:
(a) lm_head W8A8 GEMM tile at the decode shape (M=128, N=128256, K=4096). It is
    HBM-bound (525MB int8) like gate_up, but uses the GENERIC w8a8_linear_triton
    tile (128,128,128,ns2,nw4). Does the gate_up BLOCK_K=256 lever transfer?
(b) rope at the decode shape (n_tokens=128, Hq=32, Hkv=8, D=128). The kernel
    @triton.autotune is keyed only on (HALF, INTERLEAVED) -> the cached config is
    chosen by the FIRST rope call in the process (chat-b1 prefill, 44 tokens), not
    the decode shape. Is a hardcoded decode config faster (the EXP-L pattern)?
MIN-of-many vs clock drift.
"""
import torch, triton, time
import kernels.w8a8_gemm_kernel as KK
import kernels.rope_kernel as RK

torch.manual_seed(0)
dev = "cuda"

# ---------- (a) lm_head GEMM tile ----------
M, N, K = 128, 128256, 4096
xi = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
xs = (torch.rand(M, device=dev) * 0.01 + 1e-3).float()
w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
ws = (torch.rand(N, device=dev) * 0.001 + 1e-4).float()
gemm = KK._w8a8_gemm


def run_gemm(BM, BN, BK, nw, ns, GM=8):
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    gemm[grid](xi, w, xs, ws, y, M, N, K,
               xi.stride(0), xi.stride(1), w.stride(0), w.stride(1),
               y.stride(0), y.stride(1),
               BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
               num_stages=ns, num_warps=nw)
    return y


def bench(fn, iters=200, reps=5):
    try:
        fn()
    except Exception as e:
        return None
    best = 1e9
    for _ in range(reps):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters * 1e6)
    return best


print("=== lm_head GEMM (M=128, N=128256, K=4096), HBM floor ~547us ===")
CUR = (128, 128, 128, 4, 2)
cands = [CUR, (128, 128, 256, 4, 2), (128, 128, 256, 8, 2), (128, 256, 128, 8, 2),
         (128, 256, 256, 8, 2), (128, 64, 256, 8, 2), (64, 128, 256, 8, 2),
         (128, 128, 128, 8, 2), (128, 256, 64, 4, 2)]
cur_t = bench(lambda: run_gemm(*CUR))
res = []
for c in cands:
    t = bench(lambda: run_gemm(*c))
    if t:
        res.append((t, c))
res.sort()
for t, c in res:
    tag = "  <-- current" if c == CUR else f"  {cur_t/t:.3f}x"
    print(f"  {str(c):<24} {t:>7.1f}us{tag}")

# ---------- (b) rope decode shape ----------
print("\n=== rope decode (n_tokens=128, Hq=32, Hkv=8, D=128) ===")
B, T, Hq, Hkv, D = 128, 1, 32, 8, 128
q = torch.randn(B, T, Hq, D, device=dev, dtype=torch.bfloat16).contiguous()
k = torch.randn(B, T, Hkv, D, device=dev, dtype=torch.bfloat16).contiguous()
cos = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
sin = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
HALF = D // 2
BLOCK_SIZE = triton.next_power_of_2(HALF)
n_tokens = B * T
grid = (n_tokens, Hq + Hkv)


def run_rope_cfg(nw, ns):
    q_out = torch.empty_like(q); k_out = torch.empty_like(k)
    RK._rope_qk_fwd.fn[grid](
        q, k, cos, sin, q_out, k_out, Hq, Hkv, T, D,
        HEAD_DIM=D, HALF=HALF, BLOCK_SIZE=BLOCK_SIZE, INTERLEAVED=False,
        num_warps=nw, num_stages=ns,
    )
    return q_out


def run_rope_autotuned():
    return RK.rope_triton(q, k, cos, sin)


t_auto = bench(run_rope_autotuned)
print(f"  autotuned (production)    {t_auto:>7.2f}us")
for nw in (1, 2, 4):
    for ns in (1, 2):
        t = bench(lambda: run_rope_cfg(nw, ns))
        if t:
            print(f"  nw={nw} ns={ns}              {t:>7.2f}us  {t_auto/t:.3f}x")
