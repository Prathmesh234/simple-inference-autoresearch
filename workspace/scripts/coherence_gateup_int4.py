"""Multi-token greedy coherence gate for int4-grouped gate_up (g=64).

faith_gateup_int4.py showed single-token final-position argmax=1.000 at g=64 but
logit_rel~0.25 and top5~0.70 -- borderline. The real program.md gate is whether
GREEDY generation stays coherent over many tokens (errors compound). This monkeypatches
SwiGLUMLP.forward to dequantize gate_up through a grouped-int4 weight and greedily
generates from generic prompts; the text must read as coherent English. If it does,
a W4A8 fused-swiglu kernel is worth building (gate_up is the biggest, HBM-bound decode
op; int4 weight halves its 117MB->58MB HBM traffic).
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.mlp as mlp_mod

DEV, DT = "cuda", torch.bfloat16
GROUP = 64
NEW = 48
model_id = "meta-llama/Llama-3.1-8B"
loader = WeightLoader.from_pretrained(model_id)
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader); model.to(DEV, DT); model.eval()
tok = AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token = tok.eos_token
# force the eager (non-graph) decode path so the monkeypatch is actually used
import model.llama as llama_mod
llama_mod.USE_CUDA_GRAPH = False

PROMPTS = [
    "The capital of France is",
    "In a complete sentence, explain why the sky appears blue:",
    "Here is a short recipe for chocolate chip cookies. First,",
    "The three laws of motion, formulated by Isaac Newton, state that",
]

def grouped_int4_deq(W, group):
    N, K = W.shape; Wg = W.view(N, K // group, group)
    amax = Wg.abs().amax(-1, keepdim=True); scale = amax / 7.0
    q = torch.clamp(torch.round(Wg / scale), -8, 7)
    return (q * scale).view(N, K).to(W.dtype)

MODE = {"int4": True}
orig_fwd = mlp_mod.SwiGLUMLP.forward
def patched(self, x, x_int8=None, x_scale=None):
    if not MODE["int4"]:
        return orig_fwd(self, x, x_int8, x_scale)
    W = (self.w_gate_up_int8.float() * self.w_gate_up_scale.float()[:, None]).to(x.dtype)
    Wd = grouped_int4_deq(W, GROUP)
    gu = F.linear(x, Wd); I = gu.shape[-1] // 2
    g, u = gu[..., :I], gu[..., I:]
    h = F.silu(g) * u
    Wd2 = (self.w_down_int8.float() * self.w_down_scale.float()[:, None]).to(x.dtype)
    return F.linear(h, Wd2)
mlp_mod.SwiGLUMLP.forward = patched

tok.padding_side = "left"
ids = tok(PROMPTS, return_tensors="pt", padding=True).input_ids.to(DEV)
B, T = ids.shape
kv = KVCache(n_layers=cfg.num_hidden_layers, max_batch=B, max_seq_len=T + NEW,
             n_heads_kv=cfg.num_key_value_heads, head_dim=cfg.head_dim, dtype=DT, device=DEV)
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

print(f"=== int4 gate_up g={GROUP} greedy continuations (B={B}) ===")
for i in range(B):
    print(f"\n[{i}] {PROMPTS[i]!r}\n    -> {tok.decode(gen[i], skip_special_tokens=True)!r}")
