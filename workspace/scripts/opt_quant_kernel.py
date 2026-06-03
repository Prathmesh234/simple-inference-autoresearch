"""Speed up per-token int8 quant of (M=128, K=4096) bf16. Current ~36us is slow."""
import torch, triton, triton.language as tl
dev="cuda"; torch.manual_seed(0)
M,K=128,4096
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)

@triton.jit
def q_rowblock(x_ptr, xi_ptr, xs_ptr, M, K, sxm, sxk, BLOCK_K: tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0,BLOCK_K); mask=offs<K
    xv=tl.load(x_ptr+row*sxm+offs*sxk,mask=mask,other=0.0).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(xv)),1e-8); scale=amax/127.0
    xi=tl.extra.cuda.libdevice.round(xv/scale)
    xi=tl.minimum(tl.maximum(xi,-127.0),127.0).to(tl.int8)
    tl.store(xi_ptr+row*K+offs,xi,mask=mask); tl.store(xs_ptr+row,scale)

# variant: no libdevice round, use floor(x+0.5*sign) approximation via int cast on rounded
@triton.jit
def q_fast(x_ptr, xi_ptr, xs_ptr, M, K, sxm, sxk, BLOCK_K: tl.constexpr):
    row=tl.program_id(0); offs=tl.arange(0,BLOCK_K); mask=offs<K
    xv=tl.load(x_ptr+row*sxm+offs*sxk,mask=mask,other=0.0).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(xv)),1e-8); inv=127.0/amax
    xs=xv*inv
    xi=tl.where(xs>=0, tl.floor(xs+0.5), tl.ceil(xs-0.5))
    xi=tl.minimum(tl.maximum(xi,-127.0),127.0).to(tl.int8)
    tl.store(xi_ptr+row*K+offs,xi,mask=mask); tl.store(xs_ptr+row,amax/127.0)

xi=torch.empty((M,K),dtype=torch.int8,device=dev); xs=torch.empty((M,),dtype=torch.float32,device=dev)
def bench(fn,iters=500,warmup=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000
BKq=triton.next_power_of_2(K)
for nw in [4,8,16]:
    t=sorted(bench(lambda nw=nw: q_rowblock[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=BKq,num_warps=nw)) for _ in range(7))[3]
    print(f"q_rowblock libdevice num_warps={nw}: {t:.1f}us")
for nw in [4,8,16]:
    t=sorted(bench(lambda nw=nw: q_fast[(M,)](x,xi,xs,M,K,x.stride(0),x.stride(1),BLOCK_K=BKq,num_warps=nw)) for _ in range(7))[3]
    print(f"q_fast      no-libdev num_warps={nw}: {t:.1f}us")
# pure torch baseline for reference
def tref():
    amax=x.abs().amax(-1,keepdim=True).clamp_min(1e-8); s=amax/127.0
    return (x/s).round().clamp(-127,127).to(torch.int8)
print(f"torch chain: {sorted(bench(tref) for _ in range(7))[3]:.1f}us")
