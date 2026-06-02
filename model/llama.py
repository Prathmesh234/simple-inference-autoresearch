"""
LlamaModel — the full Llama-3.1-8B prefill forward pass.

Architecture recap
------------------
token_ids
  → TokenEmbedding          (vocab_size, hidden_size) lookup
  → 32 × TransformerBlock   each = RMSNorm + Attention + RMSNorm + MLP
  → RMSNorm                 final layer norm
  → OutputProjection        (hidden_size → vocab_size) logits, untied weights

The residual stream flows through all 32 blocks unchanged in shape:
  (B, T, 4096)  the entire time.

Weight manifest
---------------
  embed_tokens.weight              (128256, 4096)   ← TokenEmbedding
  layers.{0..31}.*                 9 tensors each   ← TransformerBlock
  norm.weight                      (4096,)          ← final RMSNorm
  lm_head.weight                   (128256, 4096)   ← separate lm_head (untied)

Parameter count sanity check
-----------------------------
  Embed + head (untied): 2 × 128256 × 4096          = 1050.6M
  32 × attention:        32 × (16.78+4.19+4.19+16.78)M = 32 × 41.94M = 1342.1M
  32 × MLP:              32 × (58.72+58.72+58.72)M   = 32 × 176.16M = 5637.1M
  32 × 2 norms:          32 × 2 × 4096               = negligible
  Final norm:            4096                         = negligible
  Total:                 ~8.03B parameters
"""

import torch
import torch.nn as nn

from config import ModelConfig
from loader import WeightLoader
from ops.embedding import TokenEmbedding, OutputProjection
from ops.rmsnorm import RMSNorm
from ops.rope import RopeFrequencies
from model.block import TransformerBlock


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, device: torch.device):
        """
        Args:
            cfg:    ModelConfig with all hyperparameters
            device: target device (cuda)
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Precompute RoPE tables once — shared across all blocks
        rope_cfg = cfg.rope_scaling
        self.rope_freqs = RopeFrequencies(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            rope_type=rope_cfg.rope_type,
            factor=rope_cfg.factor,
            low_freq_factor=rope_cfg.low_freq_factor,
            high_freq_factor=rope_cfg.high_freq_factor,
            original_max_seq_len=rope_cfg.original_max_position_embeddings,
            device=device,
        )

        # Token embedding table (owns the weight; OutputProjection will share it)
        self.embed = TokenEmbedding(cfg.vocab_size, cfg.hidden_size)

        # 32 transformer blocks — all share the same RopeFrequencies instance
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size, 
                num_heads_q=cfg.num_attention_heads,
                num_heads_kv=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                rope_freqs=self.rope_freqs,
                norm_eps=cfg.rms_norm_eps,
                layer_idx=i,
            )
            for i in range(cfg.num_hidden_layers)
        ])
        # Final layer norm (applied after all 32 blocks)
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Output projection. When the checkpoint ties embeddings (tied checkpoints)
        # this shares embed's weight; otherwise (Llama-3.1-8B) it owns a separate
        # lm_head weight loaded in load_weights().
        self.head = OutputProjection(self.embed, tied=cfg.tie_word_embeddings)

    def load_weights(self, loader: WeightLoader):
        """Load all weights from the checkpoint into this model."""
        dtype = self.cfg.model_dtype()

        # Embedding table
        self.embed.load_weight(loader.get("embed_tokens").to(dtype))

        # All 32 blocks
        for i, block in enumerate(self.layers):
            block.load_weights(
                attn_norm_weight=loader.get(f"layers.{i}.attn_norm").to(dtype),
                wq=loader.get(f"layers.{i}.attn.wq").to(dtype),
                wk=loader.get(f"layers.{i}.attn.wk").to(dtype),
                wv=loader.get(f"layers.{i}.attn.wv").to(dtype),
                wo=loader.get(f"layers.{i}.attn.wo").to(dtype),
                mlp_norm_weight=loader.get(f"layers.{i}.mlp_norm").to(dtype),
                w_gate=loader.get(f"layers.{i}.mlp.w_gate").to(dtype),
                w_up=loader.get(f"layers.{i}.mlp.w_up").to(dtype),
                w_down=loader.get(f"layers.{i}.mlp.w_down").to(dtype),
            )

        # Final norm
        self.norm.load_weight(loader.get("norm").to(dtype))

        # lm_head: tied embeddings (tied checkpoints) reuse embed.weight, so the
        # OutputProjection already points at it — nothing to load. Untied models
        # (Llama-3.1-8B) carry a separate lm_head.weight we load explicitly.
        if not self.cfg.tie_word_embeddings:
            self.head.load_weight(loader.get("lm_head").to(dtype))

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache=None,
    ) -> torch.Tensor:
        """
        Full prefill forward pass.

        Args:
            token_ids: (B, T) integer token IDs
            start_pos: position offset for RoPE (0 for prefill, >0 for decode)
            kv_cache:  optional KVCache (Section 11) — None for now

        Returns:
            logits: (B, T, vocab_size)
        """
        # 1. Embed tokens → residual stream
        x = self.embed(token_ids)                      # (B, T, hidden_size)

        # 2. Pass through all 32 transformer blocks
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, kv_cache=kv_cache)

        # 3. Final layer norm
        x = self.norm(x)                               # (B, T, hidden_size)

        # 4. Project to vocabulary logits
        logits = self.head(x)                          # (B, T, vocab_size)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: torch.device | str = "cuda",
    ) -> "LlamaModel":
        """Convenience constructor: build + load weights in one call."""
        from config import ModelConfig
        from loader import WeightLoader

        device = torch.device(device)
        loader = WeightLoader.from_pretrained(model_id)
        # Read the real architecture from the downloaded config.json so this works
        # for any Llama variant (tied checkpoints, Llama-3.1-8B untied, ...).
        cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")

        model = cls(cfg, device)
        model.load_weights(loader)
        model.to(device)
        model.eval()
        return model
