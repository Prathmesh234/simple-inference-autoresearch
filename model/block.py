"""
Transformer Block — one layer of the Llama model.

Architecture recap
------------------
Each of the 32 Llama layers follows this structure:

    x → RMSNorm → Attention → + (residual) → x'
    x' → RMSNorm → MLP → + (residual) → output

This is called "pre-norm" because the norm happens *before* the op,
not after. Why?
  - Post-norm (norm after residual): gradients can explode as depth increases
  - Pre-norm: each op sees normalized input → more stable training

The residual connection is critical:
  - Without it, gradients vanish through 32 layers
  - With it, there's always a "straight path" for gradients to flow backward

Weight manifest for one block (layer 0 as example)
---------------------------------------------------
From HuggingFace checkpoint, each layer has 9 weight tensors:

  layers.0.input_layernorm.weight            (4096,)         ← attn_norm
  layers.0.self_attn.q_proj.weight           (4096, 4096)    ← wq
  layers.0.self_attn.k_proj.weight           (1024, 4096)    ← wk (GQA: 4× smaller)
  layers.0.self_attn.v_proj.weight           (1024, 4096)    ← wv
  layers.0.self_attn.o_proj.weight           (4096, 4096)    ← wo
  layers.0.post_attention_layernorm.weight   (4096,)         ← mlp_norm
  layers.0.mlp.gate_proj.weight              (14336, 4096)   ← w_gate
  layers.0.mlp.up_proj.weight                (14336, 4096)   ← w_up
  layers.0.mlp.down_proj.weight              (4096, 14336)   ← w_down

Total per layer: 2 RMSNorms + 4 attention matrices + 3 MLP matrices.

Shape flow for one forward pass
--------------------------------
  x                      (B, T, 4096)  ← from previous layer or embedding
  → attn_norm            (B, T, 4096)  ← normalize
  → attention            (B, T, 4096)  ← Q/K/V projection, rope, sdpa, output proj
  → + x (residual)       (B, T, 4096)
  → mlp_norm             (B, T, 4096)  ← normalize again
  → mlp                  (B, T, 4096)  ← gate/up expand to 14336, down to 4096
  → + x' (residual)      (B, T, 4096)  ← output
"""

import torch
import torch.nn as nn

from ops.rmsnorm import RMSNorm
from ops.attention import GroupedQueryAttention
from ops.mlp import SwiGLUMLP
from ops.rope import RopeFrequencies


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        rope_freqs: RopeFrequencies,
        norm_eps: float = 1e-5,
        layer_idx: int = 0,
    ):
        """
        Args:
            hidden_size:       dimension of residual stream (4096)
            intermediate_size: expanded dimension in MLP (14336)
            num_heads_q:       number of query heads (32)
            num_heads_kv:      number of key/value heads (8, GQA)
            head_dim:          dimension per head (128)
            rope_freqs:        precomputed RoPE cos/sin tables
            norm_eps:          epsilon for RMSNorm (1e-5 or 1e-6)
            layer_idx:         which transformer layer this is (for KV cache routing)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx   = layer_idx

        # Pre-attention norm
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)

        # Attention
        self.attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            rope_freqs=rope_freqs,
            layer_idx=layer_idx,
        )

        # Pre-MLP norm
        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps)

        # MLP
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def load_weights(
        self,
        attn_norm_weight: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        mlp_norm_weight: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        """Load all 9 weights for this block from the checkpoint."""
        self.attn_norm.load_weight(attn_norm_weight)
        self.attn.load_weights(wq, wk, wv, wo)
        self.mlp_norm.load_weight(mlp_norm_weight)
        self.mlp.load_weights(w_gate, w_up, w_down)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache=None,
        decode_ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        vLLM-style residual threading: the residual-add of each sublayer is
        folded into the *next* RMSNorm (see RMSNorm.add_norm), so no explicit
        ``x = x + h`` elementwise kernel is launched.

        Args:
            x:         incoming hidden states (B, T, hidden_size). For the first
                       block this is the embedding; for later blocks it is the
                       previous block's sublayer output (the pending residual add
                       is resolved here by add_norm).
            residual:  the running residual stream, or None for the first block
                       (then x itself seeds the residual).
            start_pos: position offset for RoPE.
            kv_cache:  optional KVCache.

        Returns:
            (hidden, residual) — hidden is the MLP output (its residual add is
            still pending, to be folded into the next norm); residual is the
            stream after the attention add.
        """
        # In the W8A8 decode bucket (16 < M <= 256, the b128 headline) the qkv and
        # gate_up GEMMs consume only the int8 activation + scale that the fused norm
        # emits, so the norm's bf16 normed output is dead — tell add_norm_quant to
        # skip that 1 MB/row store. Outside the bucket (b1 / prefill) the W8A16
        # fallback reads the bf16 normed, so it must still be written.
        M = x.shape[0] * x.shape[1]
        emit_bf16 = not (16 < M <= 256)

        # --- 1. Attention block ---
        # First block seeds the residual from the embedding (no pending add); later
        # blocks fold residual += x into the norm. Both go through add_norm_quant so
        # they emit the per-token int8 quant of the normed output for free (EXP-D),
        # letting EVERY layer's qkv run W8A8 (the first layer was W8A16 before — its
        # old residual=None path emitted no int8). add_residual=False just skips the
        # residual read for the first block.
        if residual is None:
            h, residual, h_int8, h_scale = self.attn_norm.add_norm_quant(
                x, None, emit_bf16=emit_bf16, add_residual=False)
        else:
            h, residual, h_int8, h_scale = self.attn_norm.add_norm_quant(
                x, residual, emit_bf16=emit_bf16)
        h = self.attn(h, start_pos=start_pos, kv_cache=kv_cache,
                      decode_ctx=decode_ctx, x_int8=h_int8, x_scale=h_scale)

        # --- 2. MLP block ---
        # Fold the attention residual add into the pre-MLP norm, and (EXP-D)
        # the per-token int8 activation quant for the W8A8 gate_up GEMM, so no
        # standalone quant kernel runs on the decode hot path.
        h, residual, h_int8, h_scale = self.mlp_norm.add_norm_quant(
            h, residual, emit_bf16=emit_bf16)
        h = self.mlp(h, x_int8=h_int8, x_scale=h_scale)

        # h's residual add stays pending — resolved by the next block's
        # attn_norm.add_norm (or the model's final norm).
        return h, residual
