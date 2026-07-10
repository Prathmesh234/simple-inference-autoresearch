from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model.registry import checkpoint_model_type, get_model_plugin
from model.qwen35_plugin import load_qwen35_config, qwen35_checkpoint_name


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


if __name__ == "__main__":
    unittest.main()
