import torch, time, triton
from kernels.w8a8_gemm_kernel import _w8a8_swiglu_fwd, _w8a8_gemm, _quant_per_token
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"

def minbench(fn, blocks=25, iters=150):
    for _ in range(40): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(blocks):
        t=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/iters*1e6)
    return best

# ---- gate_up swiglu: M=128, K=4096, I=14336 (N=2I) ----
M,K,I = 128,4096,14336
w=torch.randint(-127,128,(2*I,K),dtype=torch.int8,device=dev)
ws=(torch.rand(2*I,device=dev)*0.001+1e-4).float()
xi=torch.randint(-127,128,(M,K),dtype=torch.int8,device=dev)
xs=(torch.rand(M,device=dev)*0.01+1e-3).float()
def run_sg(BM,BN,BK,ns,nw):
    y=torch.empty((M,I),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(I,BN),)
    _w8a8_swiglu_fwd[grid](xi,w,xs,ws,y,M,I,K,xi.stride(0),xi.stride(1),w.stride(0),w.stride(1),
        y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
cur=(128,64,128,2,4)
tcur=minbench(lambda:run_sg(*cur))
best=(tcur,cur); res=[]
for BM in (64,128):
    for BK in (64,128,256):
        for ns in (2,3,4):
            for nw in (4,8):
                c=(BM,64,BK,ns,nw)
                try: t=minbench(lambda:run_sg(*c),blocks=15,iters=120)
                except Exception: continue
                res.append((t,c)); best=min(best,(t,c),key=lambda z:z[0])
res.sort()
print(f"=== gate_up swiglu cur{cur}={tcur:.1f}us ===")
for t,c in res[:5]: print(f"   {t:.1f}us {c}")
print(f"   best={best[1]} {best[0]:.1f}us speedup={tcur/best[0]:.3f}x")

# ---- lm_head W8A8: M=128, N=128256, K=4096 ----
M,N,K=128,128256,4096
wl=torch.randint(-127,128,(N,K),dtype=torch.int8,device=dev)
wls=(torch.rand(N,device=dev)*0.001+1e-4).float()
xil=torch.randint(-127,128,(M,K),dtype=torch.int8,device=dev)
xsl=(torch.rand(M,device=dev)*0.01+1e-3).float()
def run_lm(BM,BN,BK,ns,nw):
    y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a8_gemm[grid](xil,wl,xsl,wls,y,M,N,K,xil.stride(0),xil.stride(1),wl.stride(0),wl.stride(1),
        y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
cur=(128,128,128,2,4)
tcur=minbench(lambda:run_lm(*cur))
best=(tcur,cur); res=[]
for BM in (64,128):
    for BN in (64,128,256):
        for BK in (64,128,256):
            for ns in (2,3,4):
                for nw in (4,8):
                    c=(BM,BN,BK,ns,nw)
                    try: t=minbench(lambda:run_lm(*c),blocks=12,iters=80)
                    except Exception: continue
                    res.append((t,c)); best=min(best,(t,c),key=lambda z:z[0])
res.sort()
print(f"=== lm_head W8A8 cur{cur}={tcur:.1f}us ===")
for t,c in res[:6]: print(f"   {t:.1f}us {c}")
print(f"   best={best[1]} {best[0]:.1f}us speedup={tcur/best[0]:.3f}x")
