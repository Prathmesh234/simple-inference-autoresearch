import torch, time
from torch.profiler import profile, ProfilerActivity
import kernels.rope_kernel as RK
torch.manual_seed(0); dev="cuda"
B,Hq,Hkv,D = 128,32,8,128
qkv = torch.randn(B,1,Hq*D+2*Hkv*D, device=dev, dtype=torch.bfloat16)
q = qkv[..., :Hq*D].reshape(B,1,Hq,D)
k = qkv[..., Hq*D:Hq*D+Hkv*D].reshape(B,1,Hkv,D)
cos = torch.randn(1,D,device=dev,dtype=torch.bfloat16); sin=torch.randn(1,D,device=dev,dtype=torch.bfloat16)

# confirm path taken
T=1
def rs(x):
    if x.stride(-1)==1 and x.stride(2)==D and x.stride(0)==T*x.stride(1): return x.stride(1)
    return None
print("q_rs:", rs(q), "k_rs:", rs(k), "(None=fallback-to-contiguous)")

def bench(fn, n=200):
    for _ in range(30): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e6

t_strided = bench(lambda: RK.rope_triton(q,k,cos,sin))
t_contig  = bench(lambda: RK.rope_triton(q.contiguous(), k.contiguous(), cos, sin))
print(f"strided-input rope_triton : {t_strided:.2f} us/call")
print(f"contiguous-first rope_triton: {t_contig:.2f} us/call (includes the 2 copies)")

# full kernel names
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(50): RK.rope_triton(q,k,cos,sin)
    torch.cuda.synchronize()
for e in p.key_averages():
    print(f"  {e.key[:70]:<70} calls={e.count} cuda={e.self_cuda_time_total/ e.count:.2f}us")
