import torch, time
torch.manual_seed(0); dev="cuda"
M,N,K = 128, 128256, 4096
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_linear_triton
w_bf16 = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
w_int8,w_scale = quantize_int8_per_channel(w_bf16)
x = torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.5
def bench(fn,*a,iters=100):
    for _ in range(20): fn(*a)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
t16=bench(lambda:w8a16_linear_triton(x,w_int8,w_scale))
t8=bench(lambda:w8a8_linear_triton(x,w_int8,w_scale))
print(f"lm_head M=128 N=128256 K=4096:  W8A16={t16:.1f}us  W8A8={t8:.1f}us  speedup={t16/t8:.3f}x")
# weight read floor: 525MB/960GBps
print(f"  weight-read floor ~ {525e6/960e9*1e6:.0f}us")
