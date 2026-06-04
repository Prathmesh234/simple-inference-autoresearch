import sys, os; sys.path.insert(0, os.getcwd())
import torch, triton
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel, _w8a16_gemm, w8a16_linear_triton
from kernels.w8a16_gemm_splitk_kernel import w8a16_linear_splitk

torch.manual_seed(0); dev = 'cuda'
N, K = 4096, 14336  # down proj
M = 128
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
wi, sc = quantize_int8_per_channel(w)
x = (torch.randn(M, K, device=dev) * 0.5).bfloat16()
y_ref = torch.nn.functional.linear(x.float(), w.float())

def relerr(y):
    return (y.float() - y_ref).norm().item() / y_ref.norm().item()

# big eviction buffer to force cold L2 before a timed region
evict = torch.empty(int(300e6), dtype=torch.int8, device=dev)

def cold_time(fn, iters=100):
    # flush L2 before EACH call via a large memset, time with CUDA events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(20):
        evict.zero_(); fn()
    torch.cuda.synchronize()
    for i in range(iters):
        evict.zero_()
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(starts[i].elapsed_time(ends[i]) for i in range(iters))
    return ts[len(ts)//2] * 1000  # median us

# --- current production kernel (BM=32 path) ---
def cur():
    return w8a16_linear_triton(x, wi, sc)

def mk_splitk(BM, BN, BK, SK, st, wp):
    def f():
        return w8a16_linear_splitk(x, wi, sc, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                   SPLIT_K=SK, num_stages=st, num_warps=wp)
    return f

print(f"down M={M} N={N} K={K} weight={N*K/1e6:.0f}MB int8  (median of 100, L2 flushed each call)")
print(f"  rel_err current : {relerr(cur()):.2e}")
print(f"  {'current BM=32':<26}: hot {triton.testing.do_bench(cur)*1000:6.1f}us  cold {cold_time(cur):6.1f}us")

variants = [
    ("splitk BM=128 SK=4", 128, 128, 128, 4, 3, 8),
    ("splitk BM=128 SK=2", 128, 128, 128, 2, 3, 8),
    ("splitk BM=128 SK=8", 128, 128, 128, 8, 3, 8),
    ("splitk BM=64  SK=2", 64, 128, 128, 2, 3, 8),
    ("splitk BM=64  SK=4", 64, 128, 128, 4, 3, 8),
    ("splitk BM=32  SK=4 (ctrl)", 32, 128, 128, 4, 3, 8),
]
for name, BM, BN, BK, SK, st, wp in variants:
    try:
        f = mk_splitk(BM, BN, BK, SK, st, wp)
        re = relerr(f())
        hot = triton.testing.do_bench(f) * 1000
        cold = cold_time(f)
        print(f"  {name:<26}: hot {hot:6.1f}us  cold {cold:6.1f}us  rel_err {re:.2e}")
    except Exception as e:
        print(f"  {name:<26}: ERR {str(e)[:60]}")
