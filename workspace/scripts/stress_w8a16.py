import sys, os; sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel, w8a16_linear_triton
torch.manual_seed(0)
dev='cuda'
shapes={'gate_up':(28672,4096),'down':(4096,14336)}
Ms=[1,128,2048,65536,100736,131072]
for name,(N,K) in shapes.items():
    w=(torch.randn(N,K,device=dev)*0.02).bfloat16()
    wi,sc=quantize_int8_per_channel(w)
    for M in Ms:
        x=(torch.randn(M,K,device=dev)*0.5).bfloat16()
        try:
            y=w8a16_linear_triton(x,wi,sc)
            torch.cuda.synchronize()
            ref=F.linear(x.float(),(wi.float()*sc[:,None])).to(torch.bfloat16)
            rel=(y.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-9)
            print(f"{name:8} M={M:6} OK rel={rel:.2e}")
        except Exception as e:
            print(f"{name:8} M={M:6} CRASH {type(e).__name__}: {str(e)[:80]}")
            raise
