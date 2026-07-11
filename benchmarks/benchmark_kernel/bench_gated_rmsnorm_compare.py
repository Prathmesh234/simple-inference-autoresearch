"""Correctness and speed gate for Qwen DeltaNet gated RMSNorm."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
)

from benchmarks.bench_utils import bench_fn
from kernels.gated_rmsnorm_kernel import gated_rmsnorm


def main() -> None:
    torch.manual_seed(7)
    rows, width = 32, 128
    module = Qwen3_5RMSNormGated(width).cuda().to(torch.bfloat16)
    module.weight.data.normal_()
    hidden = torch.randn(
        rows, width, device="cuda", dtype=torch.bfloat16
    )
    gate = torch.randn_like(hidden)

    ref_out = module(hidden, gate)
    actual_out = gated_rmsnorm(
        hidden, gate, module.weight, module.variance_epsilon
    )
    torch.cuda.synchronize()
    output_error = (actual_out - ref_out).abs().max().item()
    if output_error != 0:
        raise AssertionError(f"parity failed: output_error={output_error}")

    ref_ms = bench_fn(
        lambda: module(hidden, gate), warmup=25, rep=100
    )
    triton_ms = bench_fn(
        lambda: gated_rmsnorm(
            hidden, gate, module.weight, module.variance_epsilon
        ),
        warmup=25,
        rep=100,
    )
    print(f"max output error: {output_error:.6f}")
    print(f"torch:  {ref_ms * 1000:.1f} us")
    print(f"triton: {triton_ms * 1000:.1f} us")
    print(f"speedup: {ref_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    main()
