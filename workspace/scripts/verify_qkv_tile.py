import torch, time, triton
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel
torch.manual_seed(1); dev="cuda"
M,N,K=128,6144,4096
w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
wi,ws=quantize_int8_per_channel(w)
x=(torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.5).contiguous()
def run(BM,BN,BK,ns,nw):
    y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a16_gemm[grid](x,wi,ws,y,M,N,K,x.stride(0),x.stride(1),wi.stride(0),wi.stride(1),
        y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
    return y
def bench(c,iters=600):
    for _ in range(50): run(*c)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): run(*c)
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6
for trial in range(3):
    cur=bench((32,128,128,3,8)); new=bench((128,64,128,3,4))
    print(f"trial{trial}: current={cur:.1f}us  new(128,64,128,3,4)={new:.1f}us  speedup={cur/new:.3f}x")
