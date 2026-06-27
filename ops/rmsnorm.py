"""
RMSNorm — Root Mean Square Layer Normalization.

Why RMSNorm instead of LayerNorm?
----------------------------------
LayerNorm subtracts the mean then divides by std:
    y = (x - mean(x)) / std(x) * weight + bias

RMSNorm skips the mean subtraction and the bias:
    y = x / rms(x) * weight
    rms(x) = sqrt(mean(x²) + eps)

Llama uses RMSNorm because:
  - Cheaper: one fewer pass over the data (no mean computation)
  - Empirically matches LayerNorm quality on language modelling
  - No bias parameter to store or load

Where it appears in the model:
  - Before every attention block  (input_layernorm)
  - Before every MLP block        (post_attention_layernorm)
  - After the final layer         (model.norm)
  That is 2×28 + 1 = 57 RMSNorm calls per forward pass.

Shape contract:
  input:  (batch, seq_len, hidden_size)   any batch/seq dims work
  weight: (hidden_size,)                  learned scale, loaded from checkpoint
  output: (batch, seq_len, hidden_size)   same shape as input
"""

import os
import torch
import torch.nn as nn

# Triton kernel dispatch (Section 14a).
#   Controlled by the USE_TRITON env var (set in .env or shell).
#   Defaults to True — Triton is the production path. Set USE_TRITON=false
#   to force the pure-PyTorch reference (useful for debugging / parity checks).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


def _pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return x_normed.to(input_dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if USE_TRITON and x.is_cuda:
            from kernels.rmsnorm_kernel import rmsnorm_triton
            return rmsnorm_triton(x, self.weight, self.eps)
        return _pytorch_rmsnorm(x, self.weight, self.eps)

    def add_norm(
        self, hidden: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual-add + RMSNorm.

        Computes ``new_residual = residual + hidden`` and ``normed =
        rmsnorm(new_residual)`` in a single kernel, folding the residual add
        (an otherwise separate elementwise launch) into this norm. Returns
        ``(normed, new_residual)``. Bit-identical to ``rmsnorm(residual +
        hidden)`` with a bf16 add — see kernels/add_rmsnorm_kernel.py.
        """
        if USE_TRITON and hidden.is_cuda:
            from kernels.add_rmsnorm_kernel import add_rmsnorm_triton
            return add_rmsnorm_triton(hidden, residual, self.weight, self.eps)
        new_residual = residual + hidden
        return _pytorch_rmsnorm(new_residual, self.weight, self.eps), new_residual

    def load_weight(self, weight: torch.Tensor):
        """Copy a tensor from the checkpoint into this module's parameter."""
        with torch.no_grad():
            self.weight.copy_(weight)

    def add_norm_quant(
        self, hidden: torch.Tensor, residual: torch.Tensor, emit_bf16: bool = True
    ):
        """Fused residual-add + RMSNorm that ALSO emits a per-token int8
        quantization of the normed output (EXP-D).

        Used for the pre-MLP norm, whose output feeds only the W8A8 gate_up
        GEMM. Folding the int8 quant into this already-memory-bound kernel
        removes the standalone per-token quant launch. Returns
        ``(normed_bf16, new_residual, normed_int8, act_scale)``. The CPU/non-
        Triton fallback returns ``(normed, new_residual, None, None)`` so the
        MLP transparently uses its bf16/W8A16 path.
        """
        if USE_TRITON and hidden.is_cuda:
            from kernels.add_rmsnorm_kernel import add_rmsnorm_quant_triton
            return add_rmsnorm_quant_triton(
                hidden, residual, self.weight, self.eps, emit_bf16=emit_bf16
            )
        new_residual = residual + hidden
        return _pytorch_rmsnorm(new_residual, self.weight, self.eps), new_residual, None, None
