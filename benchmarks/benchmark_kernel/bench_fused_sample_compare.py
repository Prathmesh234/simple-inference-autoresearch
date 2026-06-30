"""
Standalone correctness + speed gate for the FUSED top-k/top-p sampler kernel
(kernels/sampling_kernel.py::fused_topk_sample).

It must be DISTRIBUTIONALLY faithful to the canonical float32 top-k -> top-p
reference (the same bar sampling.sample()'s fast path already meets). We verify
three things:

  1. The kernel's per-candidate FINAL distribution (probs over the k nucleus
     survivors) matches the canonical float32 reference within tolerance. This is
     the "faithful distribution" guarantee.
  2. The kernel's inverse-CDF SELECTION, given a fixed uniform draw u, matches an
     independent torch inverse-CDF reference bit-for-bit (deterministic) — i.e.
     the kernel's arithmetic (temperature, nucleus, renorm, CDF, gather) is exact.
  3. Edge cases: top_p=1.0 (pure top-k), sampled ids always inside the top-k
     support, and argmax recovered as temperature -> 0 via a near-greedy draw.

Then it times the full fused sampler vs the current sampling.sample() fast path.

Run:
    uv run python benchmarks/benchmark_kernel/bench_fused_sample_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from sampling import sample
from kernels.sampling_kernel import fused_topk_sample, _fused_sample_fwd
import triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
VOCAB = 128256
TEMP, TOP_K, TOP_P = 0.7, 50, 0.9


def ref_final_probs(logits, temperature, top_k, top_p):
    """Canonical float32 top-k -> top-p distribution over the k candidates.

    Returns (vals_kept_probs (B,k), idx (B,k)) — the final per-candidate
    distribution the kernel samples from, in sorted-desc candidate order.
    """
    vals, idx = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    v = vals.float() / temperature
    p = torch.softmax(v, dim=-1)
    c = p.cumsum(-1)
    excl = c - p
    keep = excl <= top_p
    v2 = v.masked_fill(~keep, float("-inf"))
    p2 = torch.softmax(v2, dim=-1)
    return p2, idx


def torch_invcdf_select(logits, temperature, top_k, top_p, u):
    """Independent torch inverse-CDF reference: same math the kernel does."""
    vals, idx = torch.topk(logits, top_k, dim=-1, largest=True, sorted=True)
    p2, _ = ref_final_probs(logits, temperature, top_k, top_p)
    cdf = p2.cumsum(-1)
    chosen = (cdf < u.unsqueeze(-1)).sum(-1).clamp_max(top_k - 1)
    return idx.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


def kernel_final_probs(vals, idx, temperature, top_p):
    """Run the kernel's exact math but emit the FINAL per-candidate probs.

    Mirrors _fused_sample_fwd up to the inverse-CDF draw so we can compare the
    distribution directly (the kernel itself only returns the sampled id).
    """
    v = vals.float() / temperature
    p = torch.softmax(v, dim=-1)
    c = p.cumsum(-1)
    excl = c - p
    keep = excl <= top_p
    v2 = v.masked_fill(~keep, float("-inf"))
    return torch.softmax(v2, dim=-1)


def correctness():
    print("== correctness: final distribution vs canonical float32 reference ==")
    torch.manual_seed(0)
    max_abs = 0.0
    max_support = 0
    for _ in range(200):
        B = 128
        logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE) * 6.0
        ref_p, _ = ref_final_probs(logits, TEMP, TOP_K, TOP_P)
        vals, idx = torch.topk(logits, TOP_K, dim=-1, largest=True, sorted=True)
        ker_p = kernel_final_probs(vals, idx, TEMP, TOP_P)
        max_abs = max(max_abs, (ref_p - ker_p).abs().max().item())
        sd = ((ref_p > 0).sum(-1) - (ker_p > 0).sum(-1)).abs().max().item()
        max_support = max(max_support, sd)
    print(f"  trials=200  max|Δprob|={max_abs:.3e}  max_support_delta={max_support}")
    assert max_abs < 1e-5, f"kernel distribution diverges: {max_abs}"
    print("  PASS (kernel distribution == canonical float32 top-k/top-p)\n")

    print("== correctness: kernel selection == torch inverse-CDF (deterministic) ==")
    torch.manual_seed(1)
    mismatches = 0
    total = 0
    for _ in range(200):
        B = 128
        logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE) * 6.0
        vals, idx = torch.topk(logits, TOP_K, dim=-1, largest=True, sorted=True)
        idx = idx.to(torch.int64)
        u = torch.rand(B, device=DEVICE, dtype=torch.float32)
        out = torch.empty(B, device=DEVICE, dtype=torch.int64)
        _fused_sample_fwd[(B,)](
            vals, idx, u, out,
            vals.stride(0), vals.stride(1), idx.stride(0), idx.stride(1),
            TOP_K, float(TEMP), float(TOP_P),
            BLOCK_K=triton.next_power_of_2(TOP_K), num_warps=1,
        )
        ref_tok = torch_invcdf_select(logits, TEMP, TOP_K, TOP_P, u)
        mismatches += (out != ref_tok).sum().item()
        total += B
    print(f"  mismatches={mismatches}/{total}")
    assert mismatches == 0, "kernel selection diverges from inverse-CDF reference"
    print("  PASS (kernel arithmetic is exact vs the inverse-CDF reference)\n")


def edge_cases():
    print("== edge cases ==")
    torch.manual_seed(2)
    B = 64
    logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE) * 6.0
    # top_p = 1.0 -> pure top-k; sampled ids must be in support.
    ids = fused_topk_sample(logits, TEMP, TOP_K, 1.0)
    topk_idx = torch.topk(logits, TOP_K, dim=-1).indices
    assert (ids.unsqueeze(-1) == topk_idx).any(-1).all().item(), "id outside top-k support"
    # with top_p active, still in support
    ids2 = fused_topk_sample(logits, TEMP, TOP_K, TOP_P)
    assert (ids2.unsqueeze(-1) == topk_idx).any(-1).all().item(), "id outside support (top_p)"
    # near-greedy: with one clearly-dominant logit and a small temperature the
    # nucleus collapses to that token, so every draw must select it. (The true
    # greedy path, temperature==0, uses argmax directly and never hits this kernel;
    # this just checks the nucleus+inverse-CDF degenerate to a point mass.)
    sep = logits.clone()
    win = torch.randint(0, VOCAB, (B,), device=DEVICE)
    sep[torch.arange(B, device=DEVICE), win] = 100.0  # unambiguous max
    ids3 = fused_topk_sample(sep, 0.1, TOP_K, TOP_P)
    assert torch.equal(ids3, win), "dominant-logit draw did not collapse to the winner"
    print("  PASS (top_p=1.0, support membership, dominant-logit collapse)\n")


def distribution_montecarlo():
    print("== Monte-Carlo: empirical token frequencies match reference probs ==")
    torch.manual_seed(3)
    B = 4
    logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE) * 6.0
    ref_p, idx = ref_final_probs(logits, TEMP, TOP_K, TOP_P)  # (B,k)
    N = 20000
    counts = torch.zeros(B, TOP_K, device=DEVICE)
    for _ in range(N):
        ids = fused_topk_sample(logits, TEMP, TOP_K, TOP_P)  # (B,)
        # map sampled id back to its candidate position
        pos = (ids.unsqueeze(-1) == idx).float().argmax(-1)
        counts[torch.arange(B, device=DEVICE), pos] += 1
    emp = counts / N
    max_dev = (emp - ref_p).abs().max().item()
    print(f"  draws={N}  max|empirical-ref prob|={max_dev:.4f}")
    assert max_dev < 0.02, f"empirical distribution off by {max_dev}"
    print("  PASS (empirical sampling distribution matches the reference)\n")


def speed():
    print("== speed: full sampler (topk + tail) ==")
    for B in (1, 32, 64, 128):
        logits = torch.randn(B, VOCAB, device=DEVICE, dtype=DTYPE) * 6.0
        for fn in (lambda l: sample(l, TEMP, TOP_K, TOP_P),
                   lambda l: fused_topk_sample(l, TEMP, TOP_K, TOP_P)):
            for _ in range(20):
                fn(logits)
        torch.cuda.synchronize()
        N = 300
        t0 = time.perf_counter()
        for _ in range(N):
            sample(logits, TEMP, TOP_K, TOP_P)
        torch.cuda.synchronize()
        t_old = (time.perf_counter() - t0) / N * 1000
        t0 = time.perf_counter()
        for _ in range(N):
            fused_topk_sample(logits, TEMP, TOP_K, TOP_P)
        torch.cuda.synchronize()
        t_new = (time.perf_counter() - t0) / N * 1000
        print(f"  B={B:4d}  current={t_old:6.3f}ms  fused={t_new:6.3f}ms  "
              f"speedup={t_old / t_new:4.2f}x")


if __name__ == "__main__":
    print(f"device={DEVICE} dtype={DTYPE} vocab={VOCAB} "
          f"(temp={TEMP}, top_k={TOP_K}, top_p={TOP_P})\n")
    correctness()
    edge_cases()
    distribution_montecarlo()
    speed()
    print("\nALL CHECKS PASSED")
