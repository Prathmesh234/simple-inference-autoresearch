import torch, time, triton
from kernels.w8a8_gemm_kernel import _w8a8_gemm, _quant_per_token, w8a8_linear_triton
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
def mn(fn,reps=40,inner=100,warm=15):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
def e2e_w8a8(x,wi,ws,BM,BN,BK,st,nw,GM=8):
    M,K=x.shape; N=wi.shape[0]
    def run():
        xi=torch.empty((M,K),dtype=torch.int8,device=dev); xs=torch.empty((M,),dtype=torch.float32,device=dev)
        _quant_per_token[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=triton.next_power_of_2(K),num_warps=4)
        y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
        grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
        _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),wi.stride(0),wi.stride(1),
            y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=GM,num_stages=st,num_warps=nw)
        return y
    return run
M=128
for name,N,K in [("qkv",6144,4096),("down",4096,14336)]:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wi,ws=quantize_int8_per_channel(W)
    base=min(mn(lambda:w8a16_linear_triton(x,wi,ws)) for _ in range(2))
    # find best tile (GEMM-only via prequant) then report e2e for that tile
    xi=torch.randint(-127,127,(M,K),dtype=torch.int8,device=dev); xs=torch.rand(M,device=dev)*0.01+0.001
    bestg=(1e9,None)
    for BN in (64,128,256):
      for BK in (64,128,256):
        for st in (2,3,4):
          for nw in (4,8):
            y=torch.empty((M,N),dtype=torch.bfloat16,device=dev); grid=(triton.cdiv(M,BN==0 and 1 or 128)*triton.cdiv(N,BN),)
            grid=(triton.cdiv(M,128)*triton.cdiv(N,BN),)
            def run(BN=BN,BK=BK,st=st,nw=nw,y=y,grid=grid):
                _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),BLOCK_M=128,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=st,num_warps=nw)
            try:
                t=mn(run,reps=15)
                if t<bestg[0]: bestg=(t,(BN,BK,st,nw))
            except Exception: pass
    BN,BK,st,nw=bestg[1]
    e2e=min(mn(e2e_w8a8(x,wi,ws,128,BN,BK,st,nw),reps=20) for _ in range(2))
    print(f"{name}: W8A16={base:.2f}us  W8A8 GEMM-only={bestg[0]:.2f}us {bestg[1]}  W8A8 e2e(self-quant)={e2e:.2f}us  e2e-speedup={base/e2e:.3f}x")
