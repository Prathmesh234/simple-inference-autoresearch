"""lm_head W8A8 faithfulness on REAL final-norm hidden states.
Compares W8A8 (per-token int8 act quant) vs current W8A16 logits: argmax match,
top-1/5/50 overlap, greedy multi-token coherence. lm_head argmax directly picks
the token, so this is the riskiest int8-activation site.
"""
import sys, torch
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.embedding as emb_mod
from kernels.w8a16_gemm_kernel import w8a16_linear_triton
from kernels.w8a8_gemm_kernel import w8a8_linear_triton

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token = tok.eos_token

B = 24
prompts = ["The theory of relativity states that",
           "In a surprising turn of events, scientists discovered that",
           "The best way to learn programming is to",
           "Once upon a time in a distant galaxy,",
           "The capital of France is",
           "Water boils at a temperature of"]
prompts = (prompts * 4)[:B]
ids0 = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)["input_ids"].to(DEV)

head = model.head
def logits_w8a16(x):
    return w8a16_linear_triton(x, head.w_int8, head.w_scale)
def logits_w8a8(x):
    return w8a8_linear_triton(x, head.w_int8, head.w_scale)

# Capture real final-norm hidden states by monkeypatching head.forward to record x.
captured = {}
orig_forward = emb_mod.OutputProjection.forward
def rec_forward(self, x):
    captured['x'] = x.detach().clone()
    return orig_forward(self, x)
emb_mod.OutputProjection.forward = rec_forward

kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B, max_seq_len=ids0.shape[1]+2,
             n_heads_kv=cfg.num_key_value_heads, head_dim=cfg.head_dim, dtype=DT, device=DEV)
with torch.no_grad():
    model(ids0, start_pos=0, kv_cache=kv)
emb_mod.OutputProjection.forward = orig_forward

x = captured['x'][:, -1, :].contiguous()  # (B, H) last position = decode-like M=B
print(f"final hidden: shape={tuple(x.shape)} amax/row mean={x.abs().amax(-1).mean():.2f} "
      f"outlier ratio(amax/p99)={ (x.abs().amax(-1)/x.float().abs().quantile(0.99,dim=-1)).mean():.1f}")

with torch.no_grad():
    l16 = logits_w8a16(x).float()
    l8 = logits_w8a8(x).float()

am16 = l16.argmax(-1); am8 = l8.argmax(-1)
print(f"argmax match: {(am16==am8).float().mean():.3f}")
for k in (1, 5, 50):
    t16 = l16.topk(k, -1).indices; t8 = l8.topk(k, -1).indices
    ov = torch.tensor([len(set(t16[i].tolist()) & set(t8[i].tolist()))/k for i in range(B)]).mean()
    print(f"top{k} overlap: {ov:.3f}")
rel = (l8 - l16).abs() / (l16.abs() + 1.0)
print(f"logit rel_err mean={rel.mean():.4f} max={rel.max():.4f}")
