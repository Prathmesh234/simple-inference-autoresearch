"""Cheap check: is W8A8 even FASTER than W8A16 for the down-proj decode shape
(M=128, N=4096, K=14336)? down weight is 58MB (fits Ada 96MB L2), so it may be
less HBM-bound than gate_up. If no GEMM win, SmoothQuant faithfulness work is moot.
Also measures W8A8 faithfulness on REAL post-SwiGLU activations + a quick
SmoothQuant (per-input-channel) smoothing test.
"""
import torch, triton, time
torch.manual_seed(0)
dev = "cuda"
M, N, K = 128, 4096, 14336

from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_linear_triton

# synthetic int8 weight stand-in via random bf16 then quantize
w_bf16 = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
w_int8, w_scale = quantize_int8_per_channel(w_bf16)
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.5


def bench(fn, *a, iters=300):
    for _ in range(30):
        fn(*a)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


t16 = bench(lambda: w8a16_linear_triton(x, w_int8, w_scale))
t8 = bench(lambda: w8a8_linear_triton(x, w_int8, w_scale))
print(f"down GEMM  W8A16={t16:.1f}us  W8A8(self-quant)={t8:.1f}us  speedup={t16/t8:.3f}x")
