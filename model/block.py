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
        # --- 1. Attention block ---
        # First block: no pending add, just normalize and seed the residual.
        # Later blocks: residual += x (previous mlp out), then normalize.
        if residual is None:
            residual = x
            h = self.attn_norm(x)
        else:
            h, residual = self.attn_norm.add_norm(x, residual)
        h = self.attn(h, start_pos=start_pos, kv_cache=kv_cache, decode_ctx=decode_ctx)

        # --- 2. MLP block ---
        # Fold the attention residual add into the pre-MLP norm.
        h, residual = self.mlp_norm.add_norm(h, residual)
        h = self.mlp(h)

        # h's residual add stays pending — resolved by the next block's
        # attn_norm.add_norm (or the model's final norm).
        return h, residual
