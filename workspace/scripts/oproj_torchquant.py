import torch, time, triton, torch.nn.functional as F
from kernels.w8a8_gemm_kernel import _w8a8_gemm
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
M,N,K=128,4096,4096
W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
wi,ws=quantize_int8_per_channel(W)
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
BN,BK,st,nw=32,256,3,4; grid=(triton.cdiv(M,128)*triton.cdiv(N,BN),)
def torch_quant():
    xs=(x.abs().amax(1).clamp_min(1e-8)/127.0)
    xi=(x/xs[:,None]).round().clamp_(-127,127).to(torch.int8)
    return xi,xs
def e2e():
    xi,xs=torch_quant()
    _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),BLOCK_M=128,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=st,num_warps=nw)
    return y
def mn(fn,reps=50,inner=200,warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
cub=min(mn(lambda:F.linear(x,W)) for _ in range(2))
tq=min(mn(lambda:torch_quant()) for _ in range(2))
te=min(mn(e2e) for _ in range(2))
print(f"o_proj torch-quant: cuBLAS={cub:.2f}us  torch-quant-only={tq:.2f}us  e2e={te:.2f}us  speedup={cub/te:.3f}x")
