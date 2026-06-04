"""EXP-D faithfulness: full current code (fused-quant in add_rmsnorm) vs a
forced-W8A16 reference (MLP ignores int8, uses bf16 W8A16). B=24 greedy."""
import sys, torch
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.mlp as mlp_mod
from kernels.w8a16_gemm_kernel import w8a16_linear_triton

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
           "Once upon a time in a distant galaxy,"]
prompts = (prompts * 6)[:B]
ids0 = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)["input_ids"].to(DEV)

orig_forward = mlp_mod.SwiGLUMLP.forward
def forced_w8a16(self, x, x_int8=None, x_scale=None):
    return orig_forward(self, x, None, None)  # drop int8 -> w8a8_linear_triton path

# but w8a8_linear_triton at M=24 still does W8A8 (just self-quantizes). To get a
# TRUE bf16 W8A16 reference, also patch w8a8 wrapper to delegate to w8a16:
import kernels.w8a8_gemm_kernel as w8

def gen(mode, nnew=30):
    if mode == "ref":
        mlp_mod.SwiGLUMLP.forward = forced_w8a16
        w8.w8a8_linear_triton = lambda x, wi, s: w8a16_linear_triton(x, wi, s)
    else:
        mlp_mod.SwiGLUMLP.forward = orig_forward
    kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B,
                 max_seq_len=ids0.shape[1]+nnew+1, n_heads_kv=cfg.num_key_value_heads,
                 head_dim=cfg.head_dim, dtype=DT, device=DEV)
    out=[]
    with torch.no_grad():
        logits = model(ids0, start_pos=0, kv_cache=kv)
        nt = logits[:,-1,:].argmax(-1,keepdim=True); out.append(nt); pos=ids0.shape[1]
        for _ in range(nnew-1):
            logits = model(nt, start_pos=pos, kv_cache=kv)
            nt = logits[:,-1,:].argmax(-1,keepdim=True); out.append(nt); pos+=1
    return torch.cat(out,1)

w8_orig = w8.w8a8_linear_triton
gd = gen("expd"); 
w8.w8a8_linear_triton = w8_orig
gr = gen("ref")
mlp_mod.SwiGLUMLP.forward = orig_forward; w8.w8a8_linear_triton = w8_orig
for i in [0,1,3]:
    print(f"\n--- {prompts[i]!r}")
    print(f"  EXP-D: {tok.decode(gd[i], skip_special_tokens=True)!r}")
    print(f"  W8A16: {tok.decode(gr[i], skip_special_tokens=True)!r}")
print(f"\nEXP-D vs W8A16 greedy token match: {(gd==gr).float().mean().item():.3f}")
