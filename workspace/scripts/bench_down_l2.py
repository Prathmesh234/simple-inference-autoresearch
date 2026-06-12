"""Diagnose the down_proj in-graph penalty: warm-L2 vs cold-L2 (evicted) timing.

down decode shape: M=128, K=14336, N=4096. Weight int8 = 4096*14336 = 58.7 MB,
larger than the 96MB L2 only when combined with other layers' traffic. If forcing
an L2 eviction (touch a >96MB buffer) between calls reproduces the ~399us in-graph
time, the penalty is cold-weight DRAM re-reads (split-K won't help: it doesn't cut
weight bytes). If cold-L2 stays near the warm ~95-165us, the in-graph penalty is
scheduling/occupancy and split-K (more programs, shorter K-loops) could help.
"""
import torch
from kernels.w8a16_gemm_kernel import w8a16_linear_triton, quantize_int8_per_channel

torch.manual_seed(0)
dev = "cuda"
M, K, N = 128, 14336, 4096
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
w_int8, scale = quantize_int8_per_channel(w)
print(f"down weight int8 = {w_int8.numel()/1e6:.1f} MB")

# L2 on RTX 6000 Ada = 96MB; evict with a 192MB scratch.
evict = torch.empty(192 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)


def time_warm(iters=300, reps=8):
    w8a16_linear_triton(x, w_int8, scale); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            w8a16_linear_triton(x, w_int8, scale)
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000


def time_cold(iters=200, reps=8):
    w8a16_linear_triton(x, w_int8, scale); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            evict.zero_()  # trash L2 between GEMMs
            w8a16_linear_triton(x, w_int8, scale)
        e.record(); torch.cuda.synchronize()
        # subtract the measured zero_ cost
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000


def time_evict_only(iters=200, reps=8):
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            evict.zero_()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1000


warm = time_warm()
ev = time_evict_only()
cold = time_cold() - ev
print(f"warm-L2 down GEMM:        {warm:7.1f} us")
print(f"cold-L2 down GEMM (net):  {cold:7.1f} us   (evict_only={ev:.1f}us subtracted)")
print(f"cold/warm ratio:          {cold/warm:7.2f}x")
