"""Count aten copy/elementwise kernels in the decode rope path with the CURRENT
(strided) rope vs a forced-contiguous variant, to see if the .contiguous() copies
were really eliminated and what the 4096-call elementwise op actually is."""
import torch
from torch.profiler import profile, ProfilerActivity
from kernels.rope_kernel import rope_triton
torch.manual_seed(0); dev="cuda"
B,Hq,Hkv,D = 128,32,8,128
qkv = torch.randn(B,1,Hq*D+2*Hkv*D, device=dev, dtype=torch.bfloat16)
q = qkv[..., :Hq*D].reshape(B,1,Hq,D)
k = qkv[..., Hq*D:Hq*D+Hkv*D].reshape(B,1,Hkv,D)
cos = torch.randn(1,D,device=dev,dtype=torch.bfloat16); sin=torch.randn(1,D,device=dev,dtype=torch.bfloat16)
print("q contiguous?", q.is_contiguous())
for _ in range(10): rope_triton(q,k,cos,sin)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(50): rope_triton(q,k,cos,sin)
    torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
