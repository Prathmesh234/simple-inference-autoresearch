import torch, time
torch.manual_seed(0); dev="cuda"
M,N,K = 128, 4096, 14336  # down: (hidden, intermediate)
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_linear_prequant, _quant_per_token
import triton
w_bf16 = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
w_int8,w_scale = quantize_int8_per_channel(w_bf16)
x = torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.5
# pre-quant x to int8 (simulate fused emit)
xi = torch.empty((M,K),dtype=torch.int8,device=dev); xs=torch.empty((M,),dtype=torch.float32,device=dev)
_quant_per_token[(M,)](x.contiguous(),xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=triton.next_power_of_2(K),num_warps=4)
def bench(fn,*a,iters=300):
    for _ in range(30): fn(*a)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
t16=bench(lambda:w8a16_linear_triton(x,w_int8,w_scale))
t8pq=bench(lambda:w8a8_linear_prequant(xi,xs,w_int8,w_scale,(M,K)))
print(f"down GEMM  W8A16={t16:.1f}us  W8A8(prequant,GEMM-only)={t8pq:.1f}us  speedup={t16/t8pq:.3f}x")
# faithfulness of per-token int8 on this activation
y16 = w8a16_linear_triton(x,w_int8,w_scale).float()
y8 = w8a8_linear_prequant(xi,xs,w_int8,w_scale,(M,K)).float()
rel=(y8-y16).abs()/(y16.abs()+1e-3)
print(f"  GEMM rel_err mean={rel.mean():.4f} max={rel.max():.4f}")
