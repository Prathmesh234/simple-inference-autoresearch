"""Can a TUNED int8 qkv GEMM (N=6144,K=4096,M=128) beat W8A16 49.2us?
GEMM-only (pre-quantized act, assuming free quant via input-norm fusion like EXP-D).
If yes -> real win on a dominant-bucket decode GEMM."""
import torch, time, triton
from kernels.w8a8_gemm_kernel import _w8a8_gemm
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
M,N,K=128,6144,4096
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
wi,ws=quantize_int8_per_channel(W)
xi=torch.randint(-127,127,(M,K),dtype=torch.int8,device=dev)
xs=torch.rand(M,device=dev,dtype=torch.float32)*0.01+0.001
def mn(fn,reps=40,inner=100):
    for _ in range(15): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
def w8a16(): return w8a16_linear_triton(x,wi,ws)
def make(BM,BN,BK,st,nw,GM):
    y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    def run():
        _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),
            wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),
            BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=GM,num_stages=st,num_warps=nw)
    return run
base=min(mn(w8a16),mn(w8a16))
print(f"baseline W8A16 qkv = {base:.2f}us")
best=(1e9,None)
for BN in (64,128,256):
  for BK in (64,128,256):
    for st in (2,3,4):
      for nw in (4,8):
        try:
            t=mn(make(128,BN,BK,st,nw,8),reps=20)
            if t<best[0]: best=(t,(BN,BK,st,nw))
        except Exception: pass
print(f"best W8A8 qkv tile = {best[0]:.2f}us cfg(BN,BK,st,nw)={best[1]}  speedup vs W8A16 = {base/best[0]:.3f}x")
