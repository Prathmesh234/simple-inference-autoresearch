"""b128 decode is compute-bound. qkv(N=6144)/down(N=4096) use Triton W8A16 which
upcasts int8->bf16 then bf16-MMA (upcast = pure overhead at M=128). Does native
cuBLAS bf16 (no upcast, like o_proj) beat Triton W8A16 at the decode shape?
MIN-of-many interleaved (clock-drift robust)."""
import torch, time
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel
torch.manual_seed(0); dev="cuda"
def mn(fn,reps=50,inner=100):
    for _ in range(20): fn()
    torch.cuda.synchronize(); best=1e9
    for _ in range(reps):
        t=time.perf_counter()
        for _ in range(inner): fn()
        torch.cuda.synchronize(); best=min(best,(time.perf_counter()-t)/inner*1e6)
    return best
M=128
for name,N,K in [("qkv",6144,4096),("down",4096,14336),("o_proj",4096,4096)]:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    W=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.02
    wi,ws=quantize_int8_per_channel(W)
    # interleave the two measurements to share thermal window
    t_w8=mn(lambda: w8a16_linear_triton(x,wi,ws))
    t_bf=mn(lambda: torch.nn.functional.linear(x,W))
    t_w8b=mn(lambda: w8a16_linear_triton(x,wi,ws))
    t_bfb=mn(lambda: torch.nn.functional.linear(x,W))
    w8=min(t_w8,t_w8b); bf=min(t_bf,t_bfb)
    print(f"{name:7s} N={N:6d} K={K:5d}: W8A16-triton={w8:7.2f}us  cuBLAS-bf16={bf:7.2f}us  speedup(bf/w8)={w8/bf:.3f}x")
