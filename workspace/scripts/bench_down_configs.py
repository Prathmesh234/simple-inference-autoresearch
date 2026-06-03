import sys, os; sys.path.insert(0, os.getcwd())
import torch, triton
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel, _w8a16_gemm
torch.manual_seed(0); dev='cuda'
N,K=4096,14336  # down
M=128
w=(torch.randn(N,K,device=dev)*0.02).bfloat16(); wi,sc=quantize_int8_per_channel(w)
x=(torch.randn(M,K,device=dev)*0.5).bfloat16()
def run(BM,BN,BK,st,wp,GM=8):
    y=torch.empty((M,N),dtype=x.dtype,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    def f():
        _w8a16_gemm[grid](x,wi,sc,y,M,N,K,x.stride(0),x.stride(1),wi.stride(0),wi.stride(1),
            y.stride(0),y.stride(1),BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=GM,num_stages=st,num_warps=wp)
    progs=grid[0]
    ms=triton.testing.do_bench(f, warmup=50, rep=200)
    return ms,progs
cfgs=[(32,128,128,3,8),(128,128,128,3,8),(128,64,64,3,8),(128,32,128,4,8),(128,64,128,3,8),
      (64,64,128,3,8),(128,32,64,4,8),(64,128,128,3,8),(128,128,64,4,8)]
print(f"down M={M} N={N} K={K} (weight {N*K/1e6:.0f}MB int8)")
for c in cfgs:
    try:
        ms,progs=run(*c); print(f"  BM={c[0]:3} BN={c[1]:3} BK={c[2]:3} st{c[3]} wp{c[4]}: {ms*1000:6.1f}us  progs={progs} weightloads={M//c[0] if M>=c[0] else 1}x")
    except Exception as e: print(f"  {c}: ERR {str(e)[:50]}")
