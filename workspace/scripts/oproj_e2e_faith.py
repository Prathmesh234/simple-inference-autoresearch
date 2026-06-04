import sys, torch, time, triton, torch.nn.functional as F
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
from kernels.w8a8_gemm_kernel import _w8a8_gemm, _quant_per_token
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
M,N,K=128,4096,4096
W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
wi,ws=quantize_int8_per_channel(W)
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
# preallocated (in-graph representative)
xi=torch.empty((M,K),dtype=torch.int8,device=dev); xs=torch.empty((M,),dtype=torch.float32,device=dev)
y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
BN,BK,st,nw=32,256,3,4; grid=(triton.cdiv(M,128)*triton.cdiv(N,BN),)
def quant():
    _quant_per_token[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=triton.next_power_of_2(K),num_warps=4)
def gemm():
    _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),BLOCK_M=128,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=st,num_warps=nw)
def e2e():
    quant(); gemm()
def mn(fn,reps=50,inner=200,warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
cub=min(mn(lambda:F.linear(x,W)) for _ in range(2))
tq=min(mn(quant) for _ in range(2))
te=min(mn(e2e) for _ in range(2))
print(f"o_proj: cuBLAS={cub:.2f}us  quant-only={tq:.2f}us  W8A8 e2e(quant+gemm)={te:.2f}us  e2e-speedup={cub/te:.3f}x")
# faithfulness on realistic bounded attention-output-like activation
quant(); gemm()
ref=F.linear(x,W).float()
rel=((y.float()-ref).abs()/(ref.abs()+1.0)).mean().item()
cos=F.cosine_similarity(y.float().flatten(),ref.flatten(),dim=0).item()
print(f"o_proj W8A8 vs bf16: rel_err={rel:.4f} cos={cos:.5f}")
