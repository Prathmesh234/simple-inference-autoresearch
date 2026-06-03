"""End-to-end faithfulness: full-model logit parity W8A8(gate_up) vs W8A16 at a
batch=32 decode step (exercises the int8-MMA path), plus greedy coherence."""
import os, sys, torch
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from benchmarks.prompts import load_prompts
from transformers import AutoTokenizer
import ops.mlp as mlp_mod

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token = tok.eos_token

B = 32
prompts = load_prompts("instruct", n=B, max_chars=1500)
ids = tok(prompts[:B], return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"].to(DEV)

def run_step(force_w8a16):
    # toggle: patch w8a8 wrapper to delegate to w8a16 when force
    import kernels.w8a8_gemm_kernel as w8
    orig = w8.w8a8_linear_triton
    if force_w8a16:
        from kernels.w8a16_gemm_kernel import w8a16_linear_triton
        w8.w8a8_linear_triton = lambda x, wi, s: w8a16_linear_triton(x, wi, s)
        # mlp imports the symbol inside forward, so patch the module attr too
    kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B,
                 max_seq_len=ids.shape[1]+4, n_heads_kv=cfg.num_key_value_heads,
                 head_dim=cfg.head_dim, dtype=DT, device=DEV)
    with torch.no_grad():
        logits = model(ids, start_pos=0, kv_cache=kv)
        nt = logits[:, -1, :].argmax(-1, keepdim=True)
        logits2 = model(nt, start_pos=ids.shape[1], kv_cache=kv)  # decode step, M=B=32
    w8.w8a8_linear_triton = orig
    return logits2[:, -1, :].float()

# mlp.forward does `from kernels.w8a8_gemm_kernel import w8a8_linear_triton`
# each call, so patching the module attribute is picked up.
lg_w8a8 = run_step(force_w8a16=False)
lg_ref  = run_step(force_w8a16=True)

rel = (lg_w8a8 - lg_ref).abs().mean() / lg_ref.abs().mean()
# top-k agreement (sampler uses top_k=50)
k = 50
t8 = lg_w8a8.topk(k, dim=-1).indices
tr = lg_ref.topk(k, dim=-1).indices
overlap = sum(len(set(t8[i].tolist()) & set(tr[i].tolist())) for i in range(B)) / (B*k)
arg_match = (lg_w8a8.argmax(-1) == lg_ref.argmax(-1)).float().mean().item()
print(f"batch={B} decode-step logits  rel_err={rel.item():.4f}  "
      f"top{k}_overlap={overlap:.3f}  argmax_match={arg_match:.3f}")

# greedy coherence from one prompt (batch path forced by tiling to >16 not needed
# for eyeball; just show it produces sensible text via the real engine b1 fallback)
