"""Coherence sanity-check for the down W8A8 decode path (program.md guardrail:
"output stays coherent -- sample from a fixed prompt set and sanity-check").

Runs the REAL engine (CUDA graph on) at batch 128 so the decode bucket M=128
exercises the W8A8 down path, greedily generates from generic prompts (NOT the
held-out benchmarks/prompts.py), and prints the continuations. The text must read
as coherent English. Quantitative faithfulness vs bf16 is covered separately by
probe_down_w8a8_e2e.py (teacher-forced next-token agreement).
"""
import sys, os
sys.path.insert(0, os.getcwd())
import env_loader  # noqa
import torch
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEV, DT = "cuda", torch.bfloat16
MODEL = "meta-llama/Llama-3.1-8B"
B, NEW = 128, 48

PROMPTS = [
    "The capital of France is",
    "In a complete sentence, explain why the sky appears blue:",
    "Here is a short recipe for chocolate chip cookies. First,",
    "The three laws of motion, formulated by Isaac Newton, state that",
]

loader = WeightLoader.from_pretrained(MODEL)
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV))
model.load_weights(loader)
model.to(DEV, DT)
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

texts = [PROMPTS[i % len(PROMPTS)] for i in range(B)]
tok.padding_side = "left"
ids = tok(texts, return_tensors="pt", padding=True).input_ids.to(DEV)
T = ids.shape[1]

kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B, max_seq_len=T + NEW,
             n_heads_kv=cfg.num_key_value_heads, head_dim=cfg.head_dim,
             dtype=DT, device=DEV)

gen = [[] for _ in range(B)]
with torch.no_grad():
    logits = model(ids, start_pos=0, kv_cache=kv)
    nxt = logits[:, -1, :].argmax(-1, keepdim=True)
    pos = T
    for _ in range(NEW):
        for b in range(B):
            gen[b].append(nxt[b, 0].item())
        logits = model(nxt, start_pos=pos, kv_cache=kv)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        pos += 1

print(f"=== down W8A8 greedy continuations (B={B}, M={B} -> W8A8 down) ===")
for i in range(len(PROMPTS)):
    cont = tok.decode(gen[i], skip_special_tokens=True)
    print(f"\n[{i}] {PROMPTS[i]!r}\n    -> {cont!r}")
