"""Llama model-family plugin."""

from __future__ import annotations

import torch

from config import ModelConfig
from loader import WeightLoader
from model.kv_cache import KVCache
from model.llama import LlamaModel
from model.registry import ModelPlugin, register_model_plugin


def _build_model(config: ModelConfig, device: torch.device) -> LlamaModel:
    return LlamaModel(config, device)


def _load_weights(model: LlamaModel, loader: WeightLoader) -> None:
    model.load_weights(loader)


def _build_cache(
    config: ModelConfig,
    *,
    max_batch: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> KVCache:
    return KVCache(
        n_layers=config.num_hidden_layers,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        n_heads_kv=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )


LLAMA_PLUGIN = ModelPlugin(
    family="llama",
    model_types=("llama",),
    config_loader=ModelConfig.from_hf_config,
    model_builder=_build_model,
    weight_loader=_load_weights,
    cache_builder=_build_cache,
)

register_model_plugin(LLAMA_PLUGIN)
