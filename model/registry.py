"""Model-family plugin registry.

The engine selects a plugin from the checkpoint's config.json. Model-specific
config parsing, module construction, weight loading, and cache construction stay
behind that plugin so entry points do not need architecture conditionals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from loader import WeightLoader


ConfigLoader = Callable[[Path], Any]
ModelBuilder = Callable[[Any, torch.device], nn.Module]
WeightLoaderFn = Callable[[nn.Module, WeightLoader], None]
CacheBuilder = Callable[..., Any]


@dataclass(frozen=True)
class ModelPlugin:
    family: str
    model_types: tuple[str, ...]
    config_loader: ConfigLoader
    model_builder: ModelBuilder
    weight_loader: WeightLoaderFn
    cache_builder: CacheBuilder


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    config: Any
    plugin: ModelPlugin
    weight_loader: WeightLoader


_PLUGINS: dict[str, ModelPlugin] = {}
_BUILTINS_LOADED = False


def register_model_plugin(plugin: ModelPlugin) -> None:
    """Register one model family under every HF model type it supports."""
    for model_type in plugin.model_types:
        existing = _PLUGINS.get(model_type)
        if existing is not None and existing is not plugin:
            raise ValueError(
                f"model type {model_type!r} is already registered by "
                f"{existing.family!r}"
            )
        _PLUGINS[model_type] = plugin


def checkpoint_model_type(config_path: str | Path) -> str:
    """Return the text-backbone model type from an HF config.json."""
    with open(config_path) as f:
        raw = json.load(f)

    text_config = raw.get("text_config")
    if isinstance(text_config, dict) and text_config.get("model_type"):
        return str(text_config["model_type"])
    if raw.get("model_type"):
        return str(raw["model_type"])
    raise ValueError(f"checkpoint config has no model_type: {config_path}")


def _load_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from model import llama_plugin  # noqa: F401

    _BUILTINS_LOADED = True


def get_model_plugin(config_path: str | Path) -> ModelPlugin:
    _load_builtin_plugins()
    model_type = checkpoint_model_type(config_path)
    try:
        return _PLUGINS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_PLUGINS))
        raise ValueError(
            f"unsupported model_type {model_type!r}; registered types: {supported}"
        ) from exc


def load_model(
    model_id: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Download a checkpoint and construct it through its registered plugin."""
    device = torch.device(device)
    loader = WeightLoader.from_pretrained(model_id)
    config_path = loader.model_dir / "config.json"
    plugin = get_model_plugin(config_path)
    config = plugin.config_loader(config_path)
    model = plugin.model_builder(config, device)
    plugin.weight_loader(model, loader)
    model.to(device=device, **({"dtype": dtype} if dtype is not None else {}))
    model.eval()
    return LoadedModel(model, config, plugin, loader)


def build_cache(
    loaded: LoadedModel,
    *,
    max_batch: int,
    max_seq_len: int,
    dtype: torch.dtype,
    device: torch.device | str,
):
    """Construct the architecture's decode cache through its plugin."""
    return loaded.plugin.cache_builder(
        loaded.config,
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
