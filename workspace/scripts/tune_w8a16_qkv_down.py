import torch, time, triton
from kernels.w8a16_gemm_kernel import _w8a16_gemm, quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"

def run(x2d, w_int8, scale, M,N,K, BM,BN,BK,ns,nw,GM=8):
    y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a16_gemm[grid](x2d,w_int8,scale,y,M,N,K,x2d.stride(0),x2d.stride(1),
        w_int8.stride(0),w_int8.stride(1),y.stride(0),y.stride(1),
        BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=GM,num_stages=ns,num_warps=nw)
    return y

def bench(fn,iters=400):
    for _ in range(40): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/iters*1e6

for name,(M,N,K) in {"qkv":(128,6144,4096),"down":(128,4096,14336)}.items():
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wi,ws=quantize_int8_per_channel(w)
    x=(torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.5).contiguous()
    cur = (32,128,128,3,8)
    tcur = bench(lambda:run(x,wi,ws,M,N,K,*cur))
    best=(tcur,cur)
    results=[]
    for BM in (16,32,64,128):
        for BN in (64,128,256):
            for BK in (64,128,256):
                for ns in (2,3,4):
                    for nw in (4,8):
                        try:
                            t=bench(lambda:run(x,wi,ws,M,N,K,BM,BN,BK,ns,nw))
                        except Exception:
                            continue
                        results.append((t,(BM,BN,BK,ns,nw)))
                        if t<best[0]: best=(t,(BM,BN,BK,ns,nw))
    results.sort()
    print(f"=== {name} (M={M},N={N},K={K}) current{cur}={tcur:.1f}us ===")
    for t,c in results[:5]:
        print(f"   {t:.1f}us  {c}  {'<-BEST' if c==best[1] else ''}")
    print(f"   speedup best/current = {tcur/best[0]:.3f}x")
