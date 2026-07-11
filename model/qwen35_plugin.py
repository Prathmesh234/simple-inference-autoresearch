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
import torch.nn.functional as F
from transformers import DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.cache_utils import DynamicLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
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
USE_QWEN_RMSNORM_KERNEL = os.environ.get(
    "USE_QWEN_RMSNORM_KERNEL", "true"
).lower() in ("1", "true", "yes", "on")
USE_QWEN_GATED_RMSNORM_KERNEL = os.environ.get(
    "USE_QWEN_GATED_RMSNORM_KERNEL", "true"
).lower() in ("1", "true", "yes", "on")
USE_QWEN_COMBINED_GATE_UP = os.environ.get(
    "USE_QWEN_COMBINED_GATE_UP", "true"
).lower() in ("1", "true", "yes", "on")
USE_QWEN_COMBINED_DELTANET_PROJ = os.environ.get(
    "USE_QWEN_COMBINED_DELTANET_PROJ", "true"
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


def _qwen35_rmsnorm_forward(
    module: Qwen3_5RMSNorm, x: torch.Tensor
) -> torch.Tensor:
    from kernels.rmsnorm_kernel import rmsnorm_triton

    return rmsnorm_triton(
        x, module.weight, module.eps, weight_offset=1.0
    )


def _qwen35_gated_rmsnorm_forward(
    module: Qwen3_5RMSNormGated,
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    from kernels.gated_rmsnorm_kernel import gated_rmsnorm

    return gated_rmsnorm(
        hidden_states, gate, module.weight, module.variance_epsilon
    )


def _qwen35_combined_deltanet_forward(
    module: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = (
        cache_params is not None
        and cache_params.has_previous_state(module.layer_idx)
    )
    if use_precomputed_states:
        cache_layer = cache_params.layers[module.layer_idx]
        conv_state = cache_layer.conv_states
        recurrent_state = cache_layer.recurrent_states

    mixed_qkv, z, b, a = module.in_proj_all(hidden_states).split(
        module.projection_sizes, dim=-1
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = z.reshape(batch_size, seq_len, -1, module.head_v_dim)

    if use_precomputed_states and seq_len == 1:
        mixed_qkv = module.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            module.conv1d.weight.squeeze(1),
            module.conv1d.bias,
            module.activation,
        )
    else:
        if use_precomputed_states:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            new_conv_state = F.pad(
                mixed_qkv,
                (module.conv_kernel_size - mixed_qkv.shape[-1], 0),
            )
            cache_params.update_conv_state(
                new_conv_state, module.layer_idx
            )
        if module.causal_conv1d_fn is not None:
            mixed_qkv = module.causal_conv1d_fn(
                x=mixed_qkv,
                weight=module.conv1d.weight.squeeze(1),
                bias=module.conv1d.bias,
                activation=module.activation,
                seq_idx=kwargs.get("seq_idx"),
            )
        else:
            mixed_qkv = F.silu(
                module.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]]
            )
        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    query = query.reshape(
        batch_size, seq_len, -1, module.head_k_dim
    )
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(
        batch_size, seq_len, -1, module.head_v_dim
    )

    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(
        a.float() + module.dt_bias
    )
    if module.num_v_heads // module.num_k_heads > 1:
        repeats = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    if use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = (
            module.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        )
    else:
        core_attn_out, last_recurrent_state = (
            module.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=(
                    recurrent_state if use_precomputed_states else None
                ),
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(
            last_recurrent_state, module.layer_idx
        )
    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return module.out_proj(core_attn_out)


class Qwen35DynamicCache(DynamicCache):
    """Skip copies when a custom kernel has already updated state in place."""

    def reset(self) -> None:
        super().reset()
        for layer in self.layers:
            if isinstance(layer, DynamicLayer) and layer.is_initialized:
                layer.keys = layer.keys.new_empty(0)
                layer.values = layer.values.new_empty(0)

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


class Qwen35CombinedMLP(nn.Module):
    """Qwen MLP with one combined gate/up projection."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.gate_up_weight = nn.Parameter(
            torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0),
            requires_grad=False,
        )
        self.intermediate_size = mlp.gate_proj.out_features
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up_weight)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen35CombinedLinear(nn.Module):
    """One linear projection backed by concatenated output-channel weights."""

    def __init__(self, projections: tuple[nn.Linear, ...]):
        super().__init__()
        self.weight = nn.Parameter(
            torch.cat(tuple(proj.weight for proj in projections), dim=0),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


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
        if USE_QWEN_RMSNORM_KERNEL:
            for module in self.backbone.modules():
                if isinstance(module, Qwen3_5RMSNorm):
                    module.forward = _qwen35_rmsnorm_forward.__get__(
                        module, Qwen3_5RMSNorm
                    )
        if USE_QWEN_GATED_RMSNORM_KERNEL:
            for module in self.backbone.modules():
                if isinstance(module, Qwen3_5RMSNormGated):
                    module.forward = _qwen35_gated_rmsnorm_forward.__get__(
                        module, Qwen3_5RMSNormGated
                    )
        if USE_QWEN_COMBINED_GATE_UP:
            for layer in self.backbone.model.layers:
                layer.mlp = Qwen35CombinedMLP(layer.mlp)
        if USE_QWEN_COMBINED_DELTANET_PROJ:
            for layer in self.backbone.model.layers:
                linear_attn = getattr(layer, "linear_attn", None)
                if linear_attn is None:
                    continue
                projections = (
                    linear_attn.in_proj_qkv,
                    linear_attn.in_proj_z,
                    linear_attn.in_proj_b,
                    linear_attn.in_proj_a,
                )
                linear_attn.projection_sizes = tuple(
                    proj.out_features for proj in projections
                )
                linear_attn.in_proj_all = Qwen35CombinedLinear(projections)
                linear_attn.in_proj_qkv = None
                linear_attn.in_proj_z = None
                linear_attn.in_proj_b = None
                linear_attn.in_proj_a = None
                linear_attn.forward = (
                    _qwen35_combined_deltanet_forward.__get__(
                        linear_attn, Qwen3_5GatedDeltaNet
                    )
                )

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
