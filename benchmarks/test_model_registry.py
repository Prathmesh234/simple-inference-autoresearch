from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from transformers import Qwen3_5TextConfig

from model.registry import checkpoint_model_type, get_model_plugin
from model.qwen35_plugin import (
    Qwen35DynamicCache,
    Qwen35CombinedLinear,
    load_qwen35_config,
    qwen35_checkpoint_name,
)


class ModelRegistryTest(unittest.TestCase):
    def _config(self, data: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps(data))
        return path

    def test_resolves_llama_plugin(self):
        path = self._config({"model_type": "llama"})
        self.assertEqual(checkpoint_model_type(path), "llama")
        self.assertEqual(get_model_plugin(path).family, "llama")

    def test_prefers_nested_text_model_type(self):
        path = self._config(
            {
                "model_type": "multimodal_wrapper",
                "text_config": {"model_type": "llama"},
            }
        )
        self.assertEqual(checkpoint_model_type(path), "llama")
        self.assertEqual(get_model_plugin(path).family, "llama")

    def test_rejects_unknown_model_type(self):
        path = self._config({"model_type": "unknown_architecture"})
        with self.assertRaisesRegex(ValueError, "unsupported model_type"):
            get_model_plugin(path)

    def test_resolves_nested_qwen_text_plugin(self):
        path = self._config(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 4096,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            }
        )
        plugin = get_model_plugin(path)
        config = load_qwen35_config(path)
        self.assertEqual(plugin.family, "qwen3_5")
        self.assertEqual(config.layer_types[-1], "full_attention")

    def test_maps_qwen_text_checkpoint_prefix(self):
        self.assertEqual(
            qwen35_checkpoint_name("model.layers.3.self_attn.q_proj.weight"),
            "model.language_model.layers.3.self_attn.q_proj.weight",
        )
        self.assertEqual(qwen35_checkpoint_name("lm_head.weight"), "lm_head.weight")

    def test_qwen_cache_reset_clears_dynamic_sequence_length(self):
        config = Qwen3_5TextConfig(
            num_hidden_layers=1,
            layer_types=["full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=16,
        )
        cache = Qwen35DynamicCache(config=config)
        keys = torch.randn(1, 1, 3, 8)
        values = torch.randn_like(keys)
        cache.update(keys, values, layer_idx=0)
        self.assertEqual(cache.get_seq_length(), 3)

        cache.reset()

        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(cache.layers[0].keys.numel(), 0)
        self.assertEqual(cache.layers[0].values.numel(), 0)

    def test_combined_linear_preserves_projection_outputs(self):
        torch.manual_seed(7)
        first = nn.Linear(8, 5, bias=False)
        second = nn.Linear(8, 3, bias=False)
        hidden = torch.randn(2, 4, 8)
        expected = torch.cat((first(hidden), second(hidden)), dim=-1)

        actual = Qwen35CombinedLinear((first, second))(hidden)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
