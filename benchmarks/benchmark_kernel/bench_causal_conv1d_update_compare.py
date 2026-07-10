"""Correctness and speed gate for Qwen single-token causal convolution."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_causal_conv1d_update,
)

from benchmarks.bench_utils import bench_fn
from kernels.causal_conv1d_update_kernel import causal_conv1d_update


def main() -> None:
    torch.manual_seed(7)
    batch, channels, state_len = 1, 8192, 4
    hidden = torch.randn(
        batch, channels, 1, device="cuda", dtype=torch.bfloat16
    )
    state = torch.randn(
        batch, channels, state_len, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        channels, state_len, device="cuda", dtype=torch.bfloat16
    )

    ref_state = state.clone()
    ref_out = torch_causal_conv1d_update(
        hidden, ref_state, weight, activation="silu"
    )
    actual_state = state.clone()
    actual_out = causal_conv1d_update(
        hidden, actual_state, weight, activation="silu"
    )
    torch.cuda.synchronize()

    output_error = (actual_out - ref_out).abs().max().item()
    state_error = (actual_state - ref_state).abs().max().item()
    if output_error > 0.04 or state_error != 0:
        raise AssertionError(
            f"parity failed: output_error={output_error}, "
            f"state_error={state_error}"
        )

    ref_bench_state = state.clone()
    ref_ms = bench_fn(
        lambda: torch_causal_conv1d_update(
            hidden, ref_bench_state, weight, activation="silu"
        ),
        warmup=25,
        rep=100,
    )
    kernel_state = state.clone()
    triton_ms = bench_fn(
        lambda: causal_conv1d_update(
            hidden, kernel_state, weight, activation="silu"
        ),
        warmup=25,
        rep=100,
    )
    print(f"max output error: {output_error:.6f}")
    print(f"max state error:  {state_error:.8f}")
    print(f"torch:  {ref_ms * 1000:.1f} us")
    print(f"triton: {triton_ms * 1000:.1f} us")
    print(f"speedup: {ref_ms / triton_ms:.2f}x")


if __name__ == "__main__":
    main()
