"""Correctness and speed gate for single-token Gated DeltaNet recurrence."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

from benchmarks.bench_utils import bench_fn
from kernels.gated_delta_recurrent_kernel import gated_delta_recurrent


def main() -> None:
    torch.manual_seed(7)
    B, T, H, D = 1, 1, 32, 128
    query = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    g = -torch.rand(B, T, H, device="cuda", dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32))
    state = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32) * 0.01

    ref_out, ref_state = torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        state.clone(),
        True,
        True,
    )
    actual_state = state.clone()
    actual_out, actual_state = gated_delta_recurrent(
        query,
        key,
        value,
        g,
        beta,
        actual_state,
        True,
        True,
    )
    torch.cuda.synchronize()

    output_error = (actual_out - ref_out).abs().max().item()
    state_error = (actual_state - ref_state).abs().max().item()
    if output_error > 0.02 or state_error > 0.003:
        raise AssertionError(
            f"parity failed: output_error={output_error}, state_error={state_error}"
        )

    ref_ms = bench_fn(
        lambda: torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, state.clone(), True, True
        ),
        warmup=25,
        rep=100,
    )
    kernel_state = state.clone()
    triton_ms = bench_fn(
        lambda: gated_delta_recurrent(
            query, key, value, g, beta, kernel_state, True, True
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
