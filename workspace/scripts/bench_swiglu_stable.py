import torch, time
import workspace.scripts.bench_swiglu_fuse as B
for trial in range(3):
    tc = B.bench(B.current, B.xi, B.w, B.xs, B.ws)
    tf = B.bench(lambda: B.fused(B.xi, B.w, B.xs, B.ws, BLOCK_N=64, num_warps=4))
    print(f"trial {trial}: current={tc:.1f}us  fused64nw4={tf:.1f}us  speedup={tc/tf:.3f}x")
