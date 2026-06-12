"""
SwiGLU MLP — the feed-forward block inside every Llama transformer layer.

Why SwiGLU instead of a plain two-layer MLP?
----------------------------------------------
A standard two-layer MLP is:
    y = activation(x @ W_up.T) @ W_down.T

SwiGLU replaces that with a gated variant using THREE weight matrices:
    gate = silu(x @ W_gate.T)      # values in (0, 1) after sigmoid-weighted linear
    up   = x @ W_up.T              # unconstrained projection
    y    = (gate * up) @ W_down.T  # gate selects which up-projected values pass through

The intuition: the gate learns to suppress irrelevant features element-wise.
This "soft selection" is empirically stronger than plain GELU (no gating) and
comes at the cost of one extra matrix (W_gate) — a worthwhile trade-off for
model quality.

SiLU (Sigmoid Linear Unit):
    silu(x) = x * sigmoid(x)
    - Smooth, non-monotonic activation
    - Unlike ReLU it doesn't hard-zero negative values — gentler gradient flow
    - Used here because the sigmoid part forms the "gate"

Expand-then-contract pattern
------------------------------
    hidden_size      = 4096
    intermediate_size = 14336

    x        : (B, T, 4096)   ← input from residual stream
    gate, up : (B, T, 14336)  ← expand ~3.5×
    y        : (B, T, 4096)   ← contract back down

The expansion lets the model learn richer feature interactions before
contracting back to the residual stream dimension.

Weight shapes for Llama-3.1-8B
---------------------------------
    W_gate : (intermediate_size, hidden_size) = (14336, 4096)
    W_up   : (intermediate_size, hidden_size) = (14336, 4096)
    W_down : (hidden_size, intermediate_size) = (4096, 14336)

No biases — consistent with the rest of the Llama architecture.

MLP is usually the most compute-heavy op per layer (bigger matrices than attn)
and is COMPUTE-bound at prefill and MEMORY-bound at decode.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton kernel dispatch (Section 14b — SwiGLU fusion).
#   Same env-var toggle as ops/rmsnorm.py. Controls whether the silu(gate)*up
#   elementwise step is run via the fused Triton kernel or two separate
#   PyTorch ops (silu followed by mul).
USE_TRITON = os.environ.get("USE_TRITON", "true").lower() in ("1", "true", "yes", "on")


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        Args:
            hidden_size       : width of the residual stream (4096)
            intermediate_size : width of the expanded hidden dim (14336)

        Weight layout (Section 14b Level-2 fusion)
        ------------------------------------------
        W_gate and W_up share the same input x and produce parallel outputs of
        shape (B, T, I). We store them stacked as a single (2*I, H) parameter
        so the forward pass does ONE matmul instead of two:

            combined = x @ W_gate_up.T     # (B, T, 2I)
            gate, up = combined.chunk(2, dim=-1)

        This is the same trick vLLM calls MergedColumnParallelLinear.
        One larger GEMM beats two smaller GEMMs because of better tile
        utilization and a single set of kernel-launch / setup overheads.
        """
        super().__init__()
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size

        # Weight-only int8 (W8A16): decode is weight-bandwidth bound and the MLP
        # weights are ~70% of the bytes streamed per token. We store them as int8
        # (1 byte) with a per-output-channel fp32 scale instead of bf16 (2 bytes),
        # and dequantize inside the matmul (kernels/w8a16_gemm_kernel.py). The
        # standalone gate shows 1.6-1.7x at the M=128 decode shapes for both MLP
        # matrices; attention/lm_head stay bf16 (they lose at decode M). With the
        # EXP10 CUDA-graph decode the extra Triton launches replay for free, so
        # the bandwidth saving surfaces in the profiled headline.
        #
        # These are buffers, not Parameters: `module.to(dtype)` skips integer
        # tensors so the int8 storage survives the model's bf16 cast (the float
        # scale buffers are cast to bf16, which is fine — int8 |v|<=127 is exact
        # and per-output-channel scaling keeps coherence through 32 layers).
        # Combined gate+up: rows [0:I) are gate, rows [I:2I) are up.
        self.register_buffer(
            "w_gate_up_int8",
            torch.empty(2 * intermediate_size, hidden_size, dtype=torch.int8),
        )
        self.register_buffer(
            "w_gate_up_scale", torch.empty(2 * intermediate_size, dtype=torch.float32)
        )
        self.register_buffer(
            "w_down_int8",
            torch.empty(hidden_size, intermediate_size, dtype=torch.int8),
        )
        self.register_buffer(
            "w_down_scale", torch.empty(hidden_size, dtype=torch.float32)
        )

    def load_weights(
        self,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        """
        Quantize checkpoint tensors to int8 and store them.

        We keep the (w_gate, w_up, w_down) external API so the loader code
        doesn't change. The concat (gate stacked on up) and the per-output-channel
        int8 quantization both happen here, once at load time.
        """
        from kernels.w8a16_gemm_kernel import quantize_int8_per_channel
        with torch.no_grad():
            gate_up = torch.cat([w_gate, w_up], dim=0)
            gu_int8, gu_scale = quantize_int8_per_channel(gate_up)
            d_int8, d_scale = quantize_int8_per_channel(w_down)
            self.w_gate_up_int8.copy_(gu_int8)
            self.w_gate_up_scale.copy_(gu_scale)
            self.w_down_int8.copy_(d_int8)
            self.w_down_scale.copy_(d_scale)

    def forward(
        self,
        x: torch.Tensor,
        x_int8: torch.Tensor | None = None,
        x_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x       : (B, T, hidden_size) bf16 normed input.
            x_int8  : optional (M, hidden_size) int8 — the per-token quantization
                      of `x` produced by the fused add_rmsnorm (EXP-D). When
                      present and the batch is in the W8A8 decode bucket, the
                      gate_up GEMM consumes it directly, skipping a standalone
                      activation-quant launch.
            x_scale : optional (M,) fp32 per-row scale paired with `x_int8`.

        Returns:
            (B, T, hidden_size)
        """
        if USE_TRITON and x.is_cuda:
            from kernels.w8a16_gemm_kernel import w8a16_linear_triton
            from kernels.w8a8_gemm_kernel import (
                w8a8_linear_triton, w8a8_swiglu_prequant, w8a8_down_linear,
            )
            # --- 1+2. Fused gate+up (W8A8 int8 tensor cores) + SwiGLU ---
            # gate_up is the single dominant decode GEMM (N=28672). Its input is
            # the post-rmsnorm residual stream, whose activation outliers are
            # absorbed by per-token dynamic int8 scaling (end-to-end rel_err
            # matches the accepted weight-only int8). The W8A8 path covers the
            # b128 decode bucket; b1 (M<=16) and prefill (M>256) fall back to
            # W8A16. When the preceding add_rmsnorm already produced the int8
            # activation (EXP-D), a single fused kernel does the gate_up GEMM
            # AND the silu(gate)*up epilogue (EXP-E), skipping both the standalone
            # activation-quant launch and the separate swiglu launch and never
            # materialising the (M, 2*intermediate) tensor.
            M = x.reshape(-1, x.shape[-1]).shape[0]
            if x_int8 is not None and 16 < M <= 256:
                fused = w8a8_swiglu_prequant(
                    x_int8, x_scale, self.w_gate_up_int8, self.w_gate_up_scale,
                    x.shape,
                )
            else:
                combined = w8a8_linear_triton(
                    x, self.w_gate_up_int8, self.w_gate_up_scale
                )                                          # (B, T, 2*intermediate)
                gate, up = combined.chunk(2, dim=-1)
                from kernels.swiglu_kernel import swiglu_triton
                fused = swiglu_triton(gate, up)

            # --- 3. Down projection ---
            # down is the dominant decode GEMM. Its 58.7 MB int8 weight fits Ada's
            # 96 MB L2, so it is compute/L2-bound and the int8 tensor cores (~2x
            # bf16) pay off: W8A8 runs the GEMM ~1.9x faster than the W8A16
            # (weight-int8, bf16-MMA) path. The SwiGLU output has no free per-row
            # int8 quant (it is not produced by a norm), so the W8A8 path pays a
            # standalone per-token quant; net it still wins in the decode bucket.
            # b1 (M<=16) and prefill (M>256) keep W8A16 (no int8-MMA win there).
            # The standalone per-token quant (K=14336) only amortises once M is
            # large enough to fill the SMs: it wins at M>=128 (the b128 headline)
            # but loses at M=32/64, so the down W8A8 bucket starts above 64.
            if 64 < M <= 256:
                return w8a8_down_linear(fused, self.w_down_int8, self.w_down_scale)
            return w8a16_linear_triton(fused, self.w_down_int8, self.w_down_scale)

        # Reference / CPU fallback: dequantize weights to x.dtype, then dense MLP.
        w_gate_up = self.w_gate_up_int8.to(x.dtype) * self.w_gate_up_scale.to(x.dtype)[:, None]
        w_down    = self.w_down_int8.to(x.dtype) * self.w_down_scale.to(x.dtype)[:, None]
        combined = F.linear(x, w_gate_up)
        gate, up = combined.chunk(2, dim=-1)
        fused = F.silu(gate) * up
        return F.linear(fused, w_down)
