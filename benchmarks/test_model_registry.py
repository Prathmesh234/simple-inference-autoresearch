from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model.registry import checkpoint_model_type, get_model_plugin


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


if __name__ == "__main__":
    unittest.main()
