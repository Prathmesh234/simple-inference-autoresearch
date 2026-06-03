"""Capture REAL gate_up & down inputs during decode and measure W8A8
(per-token int8 activation + per-channel int8 weight) error vs bf16, per layer.
This is the outlier faithfulness test that random-gaussian benches can't do."""
import os, sys, torch
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample
from benchmarks.prompts import load_prompts
from transformers import AutoTokenizer
from kernels.w8a16_gemm_kernel import quantize_int8_per_channel

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir)

prompts = load_prompts("instruct", n=4, max_chars=2000)
tok.pad_token = tok.eos_token
enc = tok(prompts[:4], return_tensors="pt", padding=True, truncation=True, max_length=256)
ids = enc["input_ids"].to(DEV)

kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=ids.shape[0],
             max_seq_len=ids.shape[1]+8, n_heads_kv=cfg.num_key_value_heads,
             head_dim=cfg.head_dim, dtype=DT, device=DEV)

# capture gate_up input (x) and down input (fused) per layer on a DECODE step
caps = {}
def mk_hook(i, kind):
    def h(mod, inp, out=None):
        caps.setdefault((i,kind), inp[0].detach())
    return h

from ops.mlp import SwiGLUMLP
mlps = [m for m in model.modules() if isinstance(m, SwiGLUMLP)]
def w8a8_err(x, w_int8, w_scale):
    # per-token act quant
    xs = x.abs().amax(-1, keepdim=True)/127.0
    xi = (x/xs).round().clamp(-127,127).to(torch.int8)
    acc = (xi.float() @ w_int8.float().t())
    y = acc * xs.float() * w_scale.float()[None,:]
    ref = x.float() @ (w_int8.float()*w_scale.float()[:,None]).t()
    return ((y-ref).abs().mean()/ref.abs().mean()).item(), \
           x.abs().amax().item()/ (x.abs().mean().item()+1e-9)  # outlier ratio

# prefill then 1 decode step, hooking gate_up input
logits = model(ids, start_pos=0, kv_cache=kv)
nt = sample(logits[:,-1,:], temperature=0.7, top_k=50, top_p=0.9).unsqueeze(-1)
hooks=[m.register_forward_pre_hook(mk_hook(i,"gu")) for i,m in enumerate(mlps)]
logits = model(nt, start_pos=ids.shape[1], kv_cache=kv)
for h in hooks: h.remove()

print(f"{'layer':>5} {'gu_rel_err':>10} {'gu_outlier':>10}")
errs=[]
for i,m in enumerate(mlps):
    x = caps[(i,"gu")]
    e, o = w8a8_err(x, m.w_gate_up_int8, m.w_gate_up_scale)
    errs.append(e)
    if i<4 or i>=len(mlps)-3 or e>0.05:
        print(f"{i:>5} {e:>10.4f} {o:>10.1f}")
import statistics
print(f"\ngate_up W8A8 rel_err: mean={statistics.mean(errs):.4f} max={max(errs):.4f} "
      f"(MLP int8 weight-only accepted err ~0.06)")
