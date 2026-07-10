"""Qwen3.5 text-model plugin.

Qwen3.5-9B is a hybrid architecture: 24 Gated DeltaNet layers and eight
full-attention layers. The plugin initially uses Transformers' faithful text
backbone while exposing the same engine contract as the custom Llama path.
That gives DeltaNet kernels a stable replacement boundary without coupling
callers to Transformers or to Qwen-specific cache state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
)

from loader import WeightLoader
from model.registry import ModelPlugin, register_model_plugin


USE_QWEN_GDN_KERNEL = os.environ.get("USE_QWEN_GDN_KERNEL", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
USE_QWEN_INPLACE_CACHE = os.environ.get(
    "USE_QWEN_INPLACE_CACHE", "true"
).lower() in ("1", "true", "yes", "on")


def load_qwen35_config(path: str | Path) -> Qwen3_5TextConfig:
    with open(path) as f:
        raw = json.load(f)
    text_config = raw.get("text_config", raw)
    return Qwen3_5TextConfig.from_dict(text_config)


def qwen35_checkpoint_name(state_name: str) -> str:
    """Translate a text-only CausalLM state key to the multimodal checkpoint."""
    if state_name.startswith("model."):
        return f"model.language_model.{state_name[len('model.'):]}"
    return state_name


class Qwen35DynamicCache(DynamicCache):
    """Skip copies when a custom kernel has already updated state in place."""

    def update_recurrent_state(
        self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs
    ) -> torch.Tensor:
        layer = self.layers[layer_idx]
        cached = getattr(layer, "recurrent_states", None)
        if (
            USE_QWEN_INPLACE_CACHE
            and cached is not None
            and recurrent_states.data_ptr() == cached.data_ptr()
        ):
            return recurrent_states
        return super().update_recurrent_state(
            recurrent_states, layer_idx, **kwargs
        )


class Qwen35Cache:
    """Hybrid full-attention KV plus DeltaNet recurrent-state cache."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        max_batch: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ):
        self.config = config
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device)
        self.cache = Qwen35DynamicCache(config=config)

    def reset(self) -> None:
        self.cache.reset()

    def __repr__(self) -> str:
        linear_layers = self.config.layer_types.count("linear_attention")
        full_layers = self.config.layer_types.count("full_attention")
        return (
            f"Qwen35Cache(linear_layers={linear_layers}, full_layers={full_layers}, "
            f"max_batch={self.max_batch}, max_seq_len={self.max_seq_len})"
        )


class Qwen35Model(nn.Module):
    """Engine-compatible wrapper around the Qwen3.5 text backbone."""

    def __init__(self, config: Qwen3_5TextConfig, device: torch.device):
        super().__init__()
        self.cfg = config
        self.device = device
        # Build metadata only. load_weights assigns checkpoint tensors directly,
        # avoiding a second 9B-parameter allocation during model construction.
        with torch.device("meta"):
            self.backbone = Qwen3_5ForCausalLM(config)

    def load_weights(self, loader: WeightLoader) -> None:
        state = {
            name: loader.get_hf(qwen35_checkpoint_name(name), device=str(self.device))
            for name in self.backbone.state_dict()
        }
        self.backbone.load_state_dict(state, strict=True, assign=True)
        # Non-persistent buffers are absent from state_dict and therefore remain
        # on meta after direct parameter assignment. Rebuild the tiny RoPE module
        # on the target device before registry finalization calls Module.to().
        self.backbone.model.rotary_emb = Qwen3_5TextRotaryEmbedding(
            self.cfg, device=self.device
        )
        if USE_QWEN_GDN_KERNEL:
            from kernels.gated_delta_recurrent_kernel import gated_delta_recurrent

            for layer in self.backbone.model.layers:
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is not None:
                    linear_attn.recurrent_gated_delta_rule = gated_delta_recurrent

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Qwen35Cache | None = None,
    ) -> torch.Tensor:
        if kv_cache is None:
            cache = None
        else:
            if token_ids.shape[0] > kv_cache.max_batch:
                raise ValueError(
                    f"batch {token_ids.shape[0]} exceeds cache capacity "
                    f"{kv_cache.max_batch}"
                )
            if start_pos + token_ids.shape[1] > kv_cache.max_seq_len:
                raise ValueError(
                    f"sequence length {start_pos + token_ids.shape[1]} exceeds "
                    f"cache capacity {kv_cache.max_seq_len}"
                )
            cache = kv_cache.cache

        outputs = self.backbone(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=cache is not None,
            logits_to_keep=1,
            return_dict=True,
        )
        return outputs.logits


def _build_model(
    config: Qwen3_5TextConfig, device: torch.device
) -> Qwen35Model:
    return Qwen35Model(config, device)


def _load_weights(model: Qwen35Model, loader: WeightLoader) -> None:
    model.load_weights(loader)


def _build_cache(
    config: Qwen3_5TextConfig,
    *,
    max_batch: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Qwen35Cache:
    return Qwen35Cache(
        config,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )


QWEN35_PLUGIN = ModelPlugin(
    family="qwen3_5",
    model_types=("qwen3_5", "qwen3_5_text"),
    config_loader=load_qwen35_config,
    model_builder=_build_model,
    weight_loader=_load_weights,
    cache_builder=_build_cache,
)

register_model_plugin(QWEN35_PLUGIN)
