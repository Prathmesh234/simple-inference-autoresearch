import torch, triton, time
from kernels.w8a16_gemm_kernel import _w8a16_gemm
torch.manual_seed(0); dev='cuda'
M,N,K=128,4096,14336
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
w=torch.randint(-127,127,(N,K),device=dev,dtype=torch.int8)
s=torch.rand(N,device=dev,dtype=torch.float32)*0.01
y=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
def run(BM,BN,BK,st,nw):
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    def f():
        _w8a16_gemm[grid](x,w,s,y,M,N,K,x.stride(0),x.stride(1),w.stride(0),w.stride(1),
            y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=st,num_warps=nw)
    for _ in range(5): f()
    torch.cuda.synchronize(); best=1e9
    for _ in range(50):
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(10): f()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t0)/10*1e6)
    return best
configs=[(32,128,256,3,8),(32,128,256,4,8),(32,128,256,3,4),(64,128,256,3,8),
         (16,128,256,4,8),(32,64,256,3,8),(32,256,256,3,8),(32,128,128,3,8)]
res=[]
for c in configs:
    try: res.append((run(*c),c))
    except Exception: res.append((9e9,c))
res.sort()
for t,c in res: print(f"{t:8.2f}us  BM={c[0]} BN={c[1]} BK={c[2]} st={c[3]} nw={c[4]}")
