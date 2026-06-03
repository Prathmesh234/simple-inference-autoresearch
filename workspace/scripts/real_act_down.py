import os, sys, torch, statistics
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample
from benchmarks.prompts import load_prompts
from transformers import AutoTokenizer
from kernels.swiglu_kernel import swiglu_triton

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token = tok.eos_token
prompts = load_prompts("instruct", n=4, max_chars=2000)
ids = tok(prompts[:4], return_tensors="pt", padding=True, truncation=True, max_length=256)["input_ids"].to(DEV)
kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=ids.shape[0],
             max_seq_len=ids.shape[1]+8, n_heads_kv=cfg.num_key_value_heads,
             head_dim=cfg.head_dim, dtype=DT, device=DEV)
from ops.mlp import SwiGLUMLP
mlps = [m for m in model.modules() if isinstance(m, SwiGLUMLP)]
caps={}
def mk(i):
    def h(mod, inp): caps.setdefault(i, inp[0].detach())
    return h
logits = model(ids, start_pos=0, kv_cache=kv)
nt = sample(logits[:,-1,:], temperature=0.7, top_k=50, top_p=0.9).unsqueeze(-1)
hk=[m.register_forward_pre_hook(mk(i)) for i,m in enumerate(mlps)]
model(nt, start_pos=ids.shape[1], kv_cache=kv)
for h in hk: h.remove()

def w8a8_err(x, wi, ws):
    xs = x.abs().amax(-1, keepdim=True)/127.0
    xi = (x/xs).round().clamp(-127,127).to(torch.int8)
    y = (xi.float() @ wi.float().t()) * xs.float() * ws.float()[None,:]
    ref = x.float() @ (wi.float()*ws.float()[:,None]).t()
    return ((y-ref).abs().mean()/ref.abs().mean()).item()

errs=[]
for i,m in enumerate(mlps):
    x = caps[i]
    combined = x.float() @ (m.w_gate_up_int8.float()*m.w_gate_up_scale.float()[:,None]).t()
    g,u = combined.chunk(2,-1)
    fused = (torch.nn.functional.silu(g)*u).to(DT)
    errs.append(w8a8_err(fused, m.w_down_int8, m.w_down_scale))
print(f"down W8A8 rel_err: mean={statistics.mean(errs):.4f} max={max(errs):.4f}")
