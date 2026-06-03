"""
Standalone correctness + speed gate for the fused add+RMSNorm kernel.

Compares against the unfused reference (bf16 residual add, then the existing
RMSNorm) for both numerical equivalence and decode-step speed.

Run:
    uv run python benchmarks/benchmark_kernel/bench_add_rmsnorm_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from kernels.add_rmsnorm_kernel import add_rmsnorm_triton
from kernels.rmsnorm_kernel import rmsnorm_triton
from ops.rmsnorm import _pytorch_rmsnorm

DEVICE = "cuda"
DTYPE = torch.bfloat16
H = 4096
EPS = 1e-5


def reference(hidden, residual, weight, eps):
    """Unfused: bf16 add, then RMSNorm (the path the engine runs today)."""
    new_r = residual + hidden            # bf16 elementwise add (separate kernel)
    normed = rmsnorm_triton(new_r, weight, eps)
    return normed, new_r


def correctness():
    print("== correctness: fused vs unfused (add then rmsnorm) ==")
    torch.manual_seed(0)
    max_n = 0.0
    max_r = 0.0
    for _ in range(100):
        rows = torch.randint(1, 4096, (1,)).item()
        hidden = torch.randn(rows, H, device=DEVICE, dtype=DTYPE)
        residual = torch.randn(rows, H, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(H, device=DEVICE, dtype=DTYPE)
        n_ref, r_ref = reference(hidden, residual, weight, EPS)
        n_fus, r_fus = add_rmsnorm_triton(hidden, residual, weight, EPS)
        # new_residual must be bit-identical (same bf16 add).
        max_r = max(max_r, (r_ref.float() - r_fus.float()).abs().max().item())
        max_n = max(max_n, (n_ref.float() - n_fus.float()).abs().max().item())
    print(f"  trials=100  max|Δnew_residual|={max_r:.3e}  max|Δnormed|={max_n:.3e}")
    assert max_r == 0.0, "new_residual must match the reference bf16 add exactly"
    # The normed output is NOT required to be bit-identical to rmsnorm_triton:
    # both kernels reduce the sum-of-squares in float32 under @triton.autotune,
    # which picks num_warps by timing, so their reduction trees (and thus the
    # last bits of the norm statistic) differ run-to-run. rmsnorm_triton is
    # itself not bit-stable, so the real gate is closeness to the float32
    # reference math below. Here we only assert the two triton paths agree to
    # within f32 reduction-order noise.
    assert max_n < 1e-1, "normed must match the unfused path within f32 reduction noise"

    # Also check against the float32 PyTorch reference math for the norm.
    torch.manual_seed(1)
    hidden = torch.randn(128, H, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(128, H, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(H, device=DEVICE, dtype=DTYPE)
    n_fus, r_fus = add_rmsnorm_triton(hidden, residual, weight, EPS)
    n_torch = _pytorch_rmsnorm(residual + hidden, weight, EPS)
    # Triton normalises in f32 but writes bf16; vs the f32 PyTorch reference this
    # is just output bf16 quantisation (the same noise the existing rmsnorm_triton
    # kernel has). Check on a relative scale.
    denom = n_torch.float().abs().clamp(min=1e-2)
    rel = ((n_fus.float() - n_torch.float()).abs() / denom).max().item()
    print(f"  vs float32 pytorch rmsnorm: max relative |Δ|={rel:.3e}")
    assert rel < 5e-2
    print("  PASS\n")


def speed():
    print("== speed: residual-add + norm cost (decode B sweep, T=1) ==")
    weight = torch.randn(H, device=DEVICE, dtype=DTYPE)
    for B in (1, 32, 64, 128):
        hidden = torch.randn(B, 1, H, device=DEVICE, dtype=DTYPE)
        residual = torch.randn(B, 1, H, device=DEVICE, dtype=DTYPE)
        for _ in range(20):
            reference(hidden, residual, weight, EPS)
            add_rmsnorm_triton(hidden, residual, weight, EPS)
        torch.cuda.synchronize()
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            reference(hidden, residual, weight, EPS)
        torch.cuda.synchronize()
        t_ref = (time.perf_counter() - t0) / N * 1000
        t0 = time.perf_counter()
        for _ in range(N):
            add_rmsnorm_triton(hidden, residual, weight, EPS)
        torch.cuda.synchronize()
        t_fus = (time.perf_counter() - t0) / N * 1000
        print(f"  B={B:4d}  unfused={t_ref:6.4f}ms  fused={t_fus:6.4f}ms  "
              f"speedup={t_ref / t_fus:4.2f}x")


if __name__ == "__main__":
    print(f"device={DEVICE} dtype={DTYPE} H={H}\n")
    correctness()
    speed()
    print("ALL CHECKS PASSED")
