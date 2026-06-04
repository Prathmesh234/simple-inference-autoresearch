"""HONEST W8A8 bench: per-token activation quant INSIDE the timed loop, for
gate_up and qkv shapes at M=1/128/256. Compare to current W8A16."""
import torch, triton, triton.language as tl
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

dev = "cuda"; torch.manual_seed(0)

@triton.jit
def _w8a8_gemm(x_ptr, w_ptr, xs_ptr, ws_ptr, y_ptr, M, N, K,
               sxm, sxk, swn, swk, sym, syn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m*BLOCK_M + tl.arange(0,BLOCK_M)) % M
    offs_n = (pid_n*BLOCK_N + tl.arange(0,BLOCK_N)) % N
    offs_k = tl.arange(0,BLOCK_K)
    x_ptrs = x_ptr + offs_m[:,None]*sxm + offs_k[None,:]*sxk
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
    acc = tl.zeros((BLOCK_M,BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K,BLOCK_K)):
        a = tl.load(x_ptrs); b = tl.load(w_ptrs)
        acc += tl.dot(a, b.T, out_dtype=tl.int32)
        x_ptrs += BLOCK_K*sxk; w_ptrs += BLOCK_K*swk
    xs = tl.load(xs_ptr + offs_m)[:,None].to(tl.float32)
    ws = tl.load(ws_ptr + offs_n)[None,:].to(tl.float32)
    y = acc.to(tl.float32) * xs * ws
    offs_ym = pid_m*BLOCK_M + tl.arange(0,BLOCK_M)
    offs_yn = pid_n*BLOCK_N + tl.arange(0,BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:,None]*sym + offs_yn[None,:]*syn
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=(offs_ym[:,None]<M)&(offs_yn[None,:]<N))

def w8a8_linear(x, wi, ws, BM, BN, BK, ns, nw):
    M, K = x.shape; N = wi.shape[0]
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    xs = (amax/127.0)
    xi = (x/xs).round().clamp(-127,127).to(torch.int8)
    xs = xs.squeeze(1).contiguous()
    y = torch.empty((M,N), dtype=torch.bfloat16, device=dev)
    grid=(triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _w8a8_gemm[grid](xi,wi,xs,ws,y,M,N,K,xi.stride(0),xi.stride(1),
        wi.stride(0),wi.stride(1),y.stride(0),y.stride(1),
        BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,num_stages=ns,num_warps=nw)
    return y

def bench(fn,iters=300,warmup=80):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000

SHAPES={"gate_up":(28672,4096),"qkv":(6144,4096)}
# pick a good tile per M from earlier: M=128 -> (128,128,128,2,4)
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
        # quant-only cost
        def qonly():
            amax=x.abs().amax(-1,keepdim=True).clamp_min(1e-8)
            xi=(x/(amax/127.0)).round().clamp(-127,127).to(torch.int8)
            return xi
        tq=sorted(bench(qonly) for _ in range(7))[3]
        print(f"  M={M:>4}  W8A16={t16:7.1f}us  W8A8(+quant)={t8:7.1f}us  "
              f"quant_only={tq:6.1f}us  speedup={t16/t8:.2f}x")
