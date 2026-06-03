"""o_proj (N=4096,K=4096,M=128): can a TUNED int8 GEMM beat cuBLAS bf16 (~28.5us)?
Prior '0.56x dead' likely used the default tile (same wrong-tile lesson as EXP-G/J).
GEMM-only (free quant assumed via flash_decode epilogue)."""
import torch, time, triton
from kernels.w8a8_gemm_kernel import _w8a8_gemm
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
M,N,K=128,4096,4096
W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
wi,ws=quantize_int8_per_channel(W)
xi=torch.randint(-127,127,(M,K),dtype=torch.int8,device=dev); xs=torch.rand(M,device=dev)*0.01+0.001
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
def mn(fn,reps=40,inner=100,warm=15):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
cub=min(mn(lambda:torch.nn.functional.linear(x,W)) for _ in range(2))
best=(1e9,None)
for BN in (32,64,128,256):
  for BK in (64,128,256):
    for st in (2,3,4):
      for nw in (4,8):
        y=torch.empty((M,N),dtype=torch.bfloat16,device=dev); grid=(triton.cdiv(M,128)*triton.cdiv(N,BN),)
        def run(BN=BN,BK=BK,st=st,nw=nw,y=y,grid=grid):
            _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),BLOCK_M=128,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=st,num_warps=nw)
        try:
            t=mn(run,reps=15)
            if t<best[0]: best=(t,(BN,BK,st,nw))
        except Exception: pass
print(f"o_proj: cuBLAS bf16={cub:.2f}us  best W8A8 GEMM-only={best[0]:.2f}us {best[1]}  speedup(cub/w8a8)={cub/best[0]:.3f}x")
