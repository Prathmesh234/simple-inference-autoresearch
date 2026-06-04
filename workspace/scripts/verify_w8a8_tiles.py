import torch, time, triton
from kernels.w8a8_gemm_kernel import _w8a8_swiglu_fwd, _w8a8_gemm
torch.manual_seed(3); dev="cuda"
def minbench(fn, blocks=40, iters=150):
    for _ in range(50): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(blocks):
        t=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/iters*1e6)
    return best
M,K,I=128,4096,14336
w=torch.randint(-127,128,(2*I,K),dtype=torch.int8,device=dev); ws=(torch.rand(2*I,device=dev)*0.001+1e-4).float()
xi=torch.randint(-127,128,(M,K),dtype=torch.int8,device=dev); xs=(torch.rand(M,device=dev)*0.01+1e-3).float()
def sg(c):
    BM,BN,BK,ns,nw=c; y=torch.empty((M,I),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(I,BN),)
    _w8a8_swiglu_fwd[grid](xi,w,xs,ws,y,M,I,K,xi.stride(0),xi.stride(1),w.stride(0),w.stride(1),y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
print("gate_up swiglu:")
for n,c in [("cur",(128,64,128,2,4)),("new",(128,64,256,2,8))]:
    print(f"  {n}{c}: {minbench(lambda:sg(c)):.2f}us")
N=128256
wl=torch.randint(-127,128,(N,K),dtype=torch.int8,device=dev); wls=(torch.rand(N,device=dev)*0.001+1e-4).float()
xil=torch.randint(-127,128,(M,K),dtype=torch.int8,device=dev); xsl=(torch.rand(M,device=dev)*0.01+1e-3).float()
def lm(c):
    BM,BN,BK,ns,nw=c; y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a8_gemm[grid](xil,wl,xsl,wls,y,M,N,K,xil.stride(0),xil.stride(1),wl.stride(0),wl.stride(1),y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
print("lm_head W8A8:")
for n,c in [("cur",(128,128,128,2,4)),("new",(128,256,256,2,8))]:
    print(f"  {n}{c}: {minbench(lambda:lm(c),blocks=30,iters=80):.2f}us")
