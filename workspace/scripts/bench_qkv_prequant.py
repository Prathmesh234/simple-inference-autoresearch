import torch, time, triton
torch.manual_seed(0); dev="cuda"
M,N,K = 128, 6144, 4096  # fused qkv: (q+k+v out, hidden)
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
from kernels.w8a8_gemm_kernel import w8a8_linear_prequant, w8a8_linear_triton, _quant_per_token
w_bf16 = torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
w_int8,w_scale = quantize_int8_per_channel(w_bf16)
x = torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.5
xi = torch.empty((M,K),dtype=torch.int8,device=dev); xs=torch.empty((M,),dtype=torch.float32,device=dev)
_quant_per_token[(M,)](x.contiguous(),xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=triton.next_power_of_2(K),num_warps=4)
def bench(fn,*a,iters=400):
    for _ in range(40): fn(*a)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn(*a)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
t16=bench(lambda:w8a16_linear_triton(x,w_int8,w_scale))
t8pq=bench(lambda:w8a8_linear_prequant(xi,xs,w_int8,w_scale,(M,K)))
t8full=bench(lambda:w8a8_linear_triton(x,w_int8,w_scale))
print(f"qkv  W8A16={t16:.1f}us  W8A8(prequant)={t8pq:.1f}us ({t16/t8pq:.3f}x)  W8A8(self-quant)={t8full:.1f}us ({t16/t8full:.3f}x)")
