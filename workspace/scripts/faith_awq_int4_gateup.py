"""jun30: does AWQ (activation-aware weight scaling) make int4 g=64 gate_up FAITHFUL?

Prior int4 gate_up (EXP-I/O/14) died on TWO grounds:
  (1) kernel: g=64 forces K=64 dots -> GEMM-inefficient at M=128 (b128 decode). BUT at
      M=1 (b1 decode) the GEMM is a bandwidth-bound GEMV: K=64 inefficiency is hidden and
      halving the 117MB->58.7MB weight read directly speeds it. So b1 int4 gate_up is
      UNEXPLORED (prior work was all M=128).
  (2) faithfulness: naive int4 g=64 gate_up -> top5 overlap only 0.6-0.7 (EXP-O). Too lossy
      for the sampler. THIS probe asks: does AWQ fix the faithfulness?

AWQ: scale each gate_up INPUT channel c by s_c = (mean|x_c|)^alpha (salient channels get
larger weights -> quantize more accurately). The identity (x/s) @ (W*s).T == x @ W.T means
the activation scaling folds into the preceding mlp_norm weight (FREE), and we int4-quantize
(W*s). Here we MEASURE final-logit faithfulness of int4 g=64 gate_up: naive vs AWQ, vs the
bf16(dequant-int8) engine reference.

Cheap gate: if AWQ int4 reaches argmax~1.0 AND top5>~0.9 AND top50>~0.9, b1 int4 gate_up is
worth a kernel. Else int4 gate_up is conclusively dead (faithfulness) at all batch sizes.
"""
import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, torch
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader); model.to(DEV, DT); model.eval()
H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
G = 64  # int4 group size along the input (K=H) dim


def int4_g(w, gsize):
    """Symmetric int4 (15 levels, +-7) grouped along the input dim (dim=1). Returns dequant."""
    N, K = w.shape
    wg = w.float().reshape(N, K // gsize, gsize)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 7.0
    wi = torch.round(wg / scale).clamp(-7, 7)
    return (wi * scale).reshape(N, K).to(w.dtype)


# Dequant the engine's int8 gate_up to bf16 (the engine's own reference weights).
gu_bf16 = []
for blk in model.layers:
    w = (blk.mlp.w_gate_up_int8.float() * blk.mlp.w_gate_up_scale.float()[:, None]).to(DT)
    gu_bf16.append(w)  # (2I, H)

# ---- Calib: collect per-input-channel activation magnitude for each layer's gate_up input.
calib = [
    "The history of the Roman empire spans many centuries of conquest and culture.",
    "Quantum mechanics describes the behavior of particles at very small scales.",
    "A balanced diet includes proteins, carbohydrates, fats, vitamins and minerals.",
    "The stock market reacted sharply to the central bank's interest rate decision.",
]
act_sum = [torch.zeros(H, device=DEV) for _ in range(L)]
act_cnt = [0 for _ in range(L)]
hooks = []
def mk_hook(i):
    def h(mod, inp, out):
        x = inp[0].detach()
        if x.shape[1] >= 1:
            act_sum[i] += x.float().abs().reshape(-1, H).mean(0) * x.reshape(-1, H).shape[0]
            act_cnt[i] += x.reshape(-1, H).shape[0]
    return h
for i, blk in enumerate(model.layers):
    hooks.append(blk.mlp.register_forward_hook(lambda m, inp, out: None))  # placeholder
# Use mlp_norm output as gate_up input: hook the mlp's forward (its x arg). Simpler: hook
# the SwiGLUMLP forward pre-hook to grab x.
for hk in hooks:
    hk.remove()
hooks = []
def mk_pre(i):
    def h(mod, args, kwargs):
        x = args[0].detach().reshape(-1, H)
        act_sum[i] += x.float().abs().mean(0) * x.shape[0]
        act_cnt[i] += x.shape[0]
    return h
for i, blk in enumerate(model.layers):
    hooks.append(blk.mlp.register_forward_pre_hook(mk_pre(i), with_kwargs=True))

with torch.no_grad():
    for p in calib:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        kv = KVCache(L, 1, ids.shape[1] + 1, cfg.num_key_value_heads, cfg.head_dim, DT, DEV)
        model(ids, start_pos=0, kv_cache=kv)
for hk in hooks:
    hk.remove()
act_mag = [(act_sum[i] / max(act_cnt[i], 1)).clamp_min(1e-6) for i in range(L)]  # (H,) per layer

# ---- Build AWQ-scaled int4 g=64 gate_up per layer; measure final-logit faithfulness.
def quant_variants(alpha):
    naive, awq = [], []
    for i in range(L):
        w = gu_bf16[i]
        naive.append(int4_g(w, G))
        s = act_mag[i].pow(alpha)              # (H,) AWQ per-input-channel scale
        s = s / s.mean()                       # normalize to keep magnitudes sane
        w_awq = w * s[None, :]                 # fold s into weight columns
        wq = int4_g(w_awq, G)
        awq.append(wq / s[None, :])            # un-fold (== quantized x@W.T identity)
    return naive, awq

test = "In a shocking turn of events, scientists announced today that"
ids = tok(test, return_tensors="pt").input_ids.to(DEV)

@torch.no_grad()
def final_logits(gu_weights):
    """Run the model with gate_up replaced by the given dequant bf16 weights (per layer)."""
    orig = []
    for i, blk in enumerate(model.layers):
        orig.append((blk.mlp.w_gate_up_int8, blk.mlp.w_gate_up_scale))
        # monkeypatch forward to use a plain bf16 matmul with the provided weight
        blk.mlp._awq_w = gu_weights[i]
    import torch.nn.functional as F
    import types
    def patched_forward(self, x, x_int8=None, x_scale=None):
        comb = F.linear(x, self._awq_w)
        gate, up = comb.chunk(2, dim=-1)
        fused = F.silu(gate) * up
        wd = (self.w_down_int8.float() * self.w_down_scale.float()[:, None]).to(x.dtype)
        return F.linear(fused, wd)
    for blk in model.layers:
        blk.mlp._orig_forward = blk.mlp.forward
        blk.mlp.forward = types.MethodType(patched_forward, blk.mlp)
    kv = KVCache(L, 1, ids.shape[1] + 1, cfg.num_key_value_heads, cfg.head_dim, DT, DEV)
    lg = model(ids, start_pos=0, kv_cache=kv)[:, -1, :].float()
    for blk in model.layers:
        blk.mlp.forward = blk.mlp._orig_forward
    return lg

ref = final_logits(gu_bf16)  # int8-dequant bf16 gate_up = engine reference
def faith(lg, ref):
    am = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
    rt = ref.topk(50, -1).indices[0]; lt = lg.topk(50, -1).indices[0]
    top5 = len(set(rt[:5].tolist()) & set(lt[:5].tolist())) / 5
    top50 = len(set(rt.tolist()) & set(lt.tolist())) / 50
    rel = ((lg - ref).abs() / (ref.abs() + 1e-3)).mean().item()
    return am, top5, top50, rel

print(f"{'variant':<22} {'argmax':>7} {'top5':>6} {'top50':>7} {'logit_rel':>10}")
for alpha in (0.0, 0.5, 1.0):
    naive, awq = quant_variants(alpha)
    if alpha == 0.0:
        a, t5, t50, r = faith(final_logits(naive), ref)
        print(f"{'int4 g64 naive':<22} {a:>7.3f} {t5:>6.2f} {t50:>7.2f} {r:>10.4f}")
    a, t5, t50, r = faith(final_logits(awq), ref)
    print(f"{'int4 g64 AWQ a='+str(alpha):<22} {a:>7.3f} {t5:>6.2f} {t50:>7.2f} {r:>10.4f}")
print("\n(reference = engine's int8-dequant bf16 gate_up; faithful bar ~ top5>=0.9 top50>=0.9)")
