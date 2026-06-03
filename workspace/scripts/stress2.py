import sys, os; sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel, w8a16_linear_triton
torch.manual_seed(0); dev='cuda'
N,K=28672,4096  # gate_up
w=(torch.randn(N,K,device=dev)*0.02).bfloat16()
wi,sc=quantize_int8_per_channel(w)
for M in [100736,131072]:
    x=(torch.randn(M,K,device=dev)*0.5).bfloat16()
    y=w8a16_linear_triton(x,wi,sc); torch.cuda.synchronize()
    # check last rows (where int32 overflow would corrupt)
    for r in [0, M//2, M-1]:
        ref=F.linear(x[r:r+1].float(), wi.float()*sc[:,None]).bfloat16()
        rel=(y[r:r+1].float()-ref.float()).norm()/ref.float().norm()
        print(f"gate_up M={M} row={r} rel={rel:.2e}")
    del x,y; torch.cuda.empty_cache()
print("OK no illegal access")
