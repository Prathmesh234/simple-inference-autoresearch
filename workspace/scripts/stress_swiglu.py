import sys, os; sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F
from kernels.swiglu_kernel import swiglu_triton
torch.manual_seed(0); dev='cuda'; ncol=14336
def ref(g,u): return (F.silu(g.float())*u.float()).to(g.dtype)
for M in [1,128,100736]:
    comb=(torch.randn(M,2*ncol,device=dev)*0.5).bfloat16()
    g,u=comb.chunk(2,dim=-1)  # non-contiguous views
    y=swiglu_triton(g,u); torch.cuda.synchronize()
    for r in [0,M//2,M-1]:
        rr=ref(g[r:r+1],u[r:r+1])
        rel=(y[r:r+1].float()-rr.float()).norm()/rr.float().norm().clamp_min(1e-9)
        print(f"M={M:7} row={r:7} rel={rel:.2e}")
    del comb,g,u,y; torch.cuda.empty_cache()
print("OK")
