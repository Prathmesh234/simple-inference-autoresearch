"""W8A8 with a FUSED Triton per-token quant kernel (one launch). Honest timing."""
import torch, triton, triton.language as tl
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
dev="cuda"; torch.manual_seed(0)

@triton.jit
def _quant_per_token(x_ptr, xi_ptr, xs_ptr, M, K, sxm, sxk,
                     BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    # one program per row; BLOCK_K >= K (K=4096 -> BLOCK_K=4096)
    mask = offs < K
    xp = x_ptr + row*sxm + offs*sxk
    x = tl.load(xp, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-8)
    scale = amax / 127.0
    xi = tl.extra.cuda.libdevice.round(x / scale)
    xi = tl.minimum(tl.maximum(xi, -127.0), 127.0).to(tl.int8)
    tl.store(xi_ptr + row*K + offs, xi, mask=mask)
    tl.store(xs_ptr + row, scale)

@triton.jit
def _w8a8_gemm(x_ptr,w_ptr,xs_ptr,ws_ptr,y_ptr,M,N,K,sxm,sxk,swn,swk,sym,syn,
               BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,GROUP_M:tl.constexpr):
    pid=tl.program_id(0)
    npm=tl.cdiv(M,BLOCK_M);npn=tl.cdiv(N,BLOCK_N);nig=GROUP_M*npn
    gid=pid//nig;fpm=gid*GROUP_M;gsm=min(npm-fpm,GROUP_M)
    pid_m=fpm+((pid%nig)%gsm);pid_n=(pid%nig)//gsm
    offs_m=(pid_m*BLOCK_M+tl.arange(0,BLOCK_M))%M
    offs_n=(pid_n*BLOCK_N+tl.arange(0,BLOCK_N))%N
    offs_k=tl.arange(0,BLOCK_K)
    xp=x_ptr+offs_m[:,None]*sxm+offs_k[None,:]*sxk
    wp=w_ptr+offs_n[:,None]*swn+offs_k[None,:]*swk
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.int32)
    for k in range(0,tl.cdiv(K,BLOCK_K)):
        a=tl.load(xp);b=tl.load(wp)
        acc+=tl.dot(a,b.T,out_dtype=tl.int32)
        xp+=BLOCK_K*sxk;wp+=BLOCK_K*swk
    xs=tl.load(xs_ptr+offs_m)[:,None].to(tl.float32)
    ws=tl.load(ws_ptr+offs_n)[None,:].to(tl.float32)
    y=acc.to(tl.float32)*xs*ws
    oym=pid_m*BLOCK_M+tl.arange(0,BLOCK_M);oyn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    yp=y_ptr+oym[:,None]*sym+oyn[None,:]*syn
    tl.store(yp,y.to(tl.bfloat16),mask=(oym[:,None]<M)&(oyn[None,:]<N))

def w8a8_linear(x,wi,ws,BM,BN,BK,ns,nw):
    M,K=x.shape;N=wi.shape[0]
    xi=torch.empty((M,K),dtype=torch.int8,device=dev)
    xs=torch.empty((M,),dtype=torch.float32,device=dev)
    BK_q=triton.next_power_of_2(K)
    _quant_per_token[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=BK_q,num_warps=8)
    y=torch.empty((M,N),dtype=torch.bfloat16,device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),
        wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),
        BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
    return y, xi, xs

def bench(fn,iters=300,warmup=80):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000

# accuracy check vs reference
N,K=28672,4096
w=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02)
wi,ws=quantize_int8_per_channel(w);wi,ws=wi.to(dev),ws.to(dev)
x=torch.randn(128,K,device=dev,dtype=torch.bfloat16)
y,xi,xs=w8a8_linear(x,wi,ws,128,128,128,2,4)
ref=x.float()@(wi.float()*ws.float()[:,None]).t()
print(f"fused-quant W8A8 rel_err={((y.float()-ref).abs().mean()/ref.abs().mean()).item():.4f}")

SHAPES={"gate_up":(28672,4096),"qkv":(6144,4096)}
CFG={1:(16,128,128,2,4),128:(128,128,128,2,4),256:(128,128,128,2,4)}
for name,(N,K) in SHAPES.items():
    w=(torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02)
    wi,ws=quantize_int8_per_channel(w);wi,ws=wi.to(dev),ws.to(dev)
    print(f"\n=== {name} N={N} K={K} ===")
    for M in [1,128,256]:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        t16=sorted(bench(lambda: w8a16_linear_triton(x,wi,ws)) for _ in range(7))[3]
        cfg=CFG[M]
        t8=sorted(bench(lambda: w8a8_linear(x,wi,ws,*cfg)) for _ in range(7))[3]
        def qonly():
            xi=torch.empty((M,K),dtype=torch.int8,device=dev)
            xs=torch.empty((M,),dtype=torch.float32,device=dev)
            _quant_per_token[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),
                BLOCK_K=triton.next_power_of_2(K),num_warps=8)
        tq=sorted(bench(qonly) for _ in range(7))[3]
        print(f"  M={M:>4}  W8A16={t16:7.1f}us  W8A8={t8:7.1f}us  "
              f"fused_quant={tq:5.1f}us  speedup={t16/t8:.2f}x")
