"""Is gate_up GEMM at M=128 compute- or bandwidth-bound? And does true int8
tensor-core dot (W8A8) beat the current int8->bf16 upcast bf16 dot?"""
import torch, triton, triton.language as tl
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)
M, N, K = 128, 28672, 4096

w = (torch.randn(N, K, device=dev, dtype=dt) * 0.02)
wi, sc = quantize_int8_per_channel(w); wi, sc = wi.to(dev), sc.to(dev)
x = torch.randn(M, K, device=dev, dtype=dt)

def bench(fn, iters=300, warmup=80):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000  # us

# current W8A16 (int8 weight upcast -> bf16 dot)
t_w8a16 = sorted(bench(lambda: w8a16_linear_triton(x, wi, sc)) for _ in range(7))[3]

# cuBLAS bf16 reference
wb = w.contiguous()
t_bf16 = sorted(bench(lambda: torch.mm(x, wb.t())) for _ in range(7))[3]

flops = 2*M*K*N
wbytes_int8 = N*K          # int8 weight
wbytes_bf16 = N*K*2
print(f"gate_up M={M} N={N} K={K}")
print(f"  W8A16(cur)  {t_w8a16:7.1f}us  {flops/(t_w8a16*1e-6)/1e12:6.1f} TFLOP/s  "
      f"{wbytes_int8/(t_w8a16*1e-6)/1e9:6.0f} GB/s(wt)")
print(f"  bf16 cuBLAS {t_bf16:7.1f}us  {flops/(t_bf16*1e-6)/1e12:6.1f} TFLOP/s  "
      f"{wbytes_bf16/(t_bf16*1e-6)/1e9:6.0f} GB/s(wt)")
print(f"  Ada peaks ~ 360 TFLOP/s bf16, ~720 TOPS int8, ~960 GB/s HBM")
