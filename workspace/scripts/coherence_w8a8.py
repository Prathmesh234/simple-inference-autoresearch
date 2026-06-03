"""Greedy batched continuations (B=24 -> W8A8 path active) W8A8 vs W8A16."""
import sys, torch
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import kernels.w8a8_gemm_kernel as w8
from kernels.w8a16_gemm_kernel import w8a16_linear_triton

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token = tok.eos_token

B = 24
prompts = [
 "The theory of relativity states that",
 "In a surprising turn of events, scientists discovered that",
 "The best way to learn programming is to",
 "Once upon a time in a distant galaxy,",
]
prompts = (prompts * 6)[:B]
enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)
ids0 = enc["input_ids"].to(DEV)

def gen(force_w8a16, nnew=30):
    orig = w8.w8a8_linear_triton
    if force_w8a16:
        w8.w8a8_linear_triton = lambda x, wi, s: w8a16_linear_triton(x, wi, s)
    kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B,
                 max_seq_len=ids0.shape[1]+nnew+1, n_heads_kv=cfg.num_key_value_heads,
                 head_dim=cfg.head_dim, dtype=DT, device=DEV)
    out = []
    with torch.no_grad():
        logits = model(ids0, start_pos=0, kv_cache=kv)
        nt = logits[:, -1, :].argmax(-1, keepdim=True); out.append(nt)
        pos = ids0.shape[1]
        for _ in range(nnew-1):
            logits = model(nt, start_pos=pos, kv_cache=kv)
            nt = logits[:, -1, :].argmax(-1, keepdim=True); out.append(nt); pos += 1
    w8.w8a8_linear_triton = orig
    return torch.cat(out, dim=1)

g8 = gen(False); g16 = gen(True)
for i in [0,1,2,3]:
    print(f"\n--- prompt: {prompts[i]!r}")
    print(f"  W8A8 : {tok.decode(g8[i], skip_special_tokens=True)!r}")
    print(f"  W8A16: {tok.decode(g16[i], skip_special_tokens=True)!r}")
match = (g8 == g16).float().mean().item()
print(f"\ntoken match W8A8 vs W8A16 (greedy): {match:.3f}")
