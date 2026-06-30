"""Coherence check for the fused sampler (EXP-21) in the REAL engine.

Generates with the actual decode sampler (temp=0.7, top_k=50, top_p=0.9) so the
fused Triton top-k/top-p/sample kernel is exercised end-to-end, and prints the
text for an eyeball coherence check (program.md guardrail). Run with
USE_FUSED_SAMPLE=false to compare against the torch tail.
"""
import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, torch
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample, USE_FUSED_SAMPLE

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
print(f"USE_FUSED_SAMPLE={USE_FUSED_SAMPLE}")

prompts = [
    "The capital of France is",
    "Q: What is 2+2? A:",
    "Water boils at a temperature of",
    "The first three prime numbers are",
]
torch.manual_seed(0)
for prompt in prompts:
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    B, P = ids.shape; maxlen = P + 48
    kv = KVCache(cfg.num_hidden_layers, B, maxlen, cfg.num_key_value_heads,
                 cfg.head_dim, DT, DEV)
    with torch.no_grad():
        logits = model(ids, start_pos=0, kv_cache=kv)
        nxt = sample(logits[:, -1, :], temperature=0.7, top_k=50, top_p=0.9).unsqueeze(-1)
        out = []
        for i in range(48):
            out.append(nxt.item())
            logits = model(nxt, start_pos=P + i, kv_cache=kv)
            nxt = sample(logits[:, -1, :], temperature=0.7, top_k=50, top_p=0.9).unsqueeze(-1)
    print(repr(prompt + " ||| " + tok.decode(out)))
