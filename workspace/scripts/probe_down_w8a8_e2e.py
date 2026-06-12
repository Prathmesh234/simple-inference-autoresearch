"""End-to-end faithfulness gate for down_proj W8A8.

The single-GEMM probe (probe_down_smoothquant.py) showed per-token W8A8 down has
rel_err ~0.07 naive / ~0.038 SmoothQuant a=0.7 -- comparable to the *accepted*
gate_up W8A8. But single-GEMM rel_err can hide error that compounds across 32
layers. This probe measures the REAL gate: next-token argmax agreement over a
full forward when EVERY layer's down switches W8A16 -> W8A8.

Method: monkeypatch SwiGLUMLP.forward to a torch (cuBLAS) implementation whose
down step is selectable via a module-global MODE. gate_up is held at bf16 for all
modes so the ONLY variable is down -> we measure down's marginal degradation.
Reference = MODE 'bf16' (dense bf16, ground truth). For each candidate mode we
report: next-token argmax agreement vs ref (all prefill positions), and mean
top-1 logit cosine. Bar: W8A8 should ~match W8A16 (the already-accepted band).
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.llama import LlamaModel

dev = "cuda"
MODEL = "meta-llama/Llama-3.1-8B"
ALPHA = 0.7

CALIB = [
    "The history of mathematics spans thousands of years and many distinct cultures, "
    "from Babylonian arithmetic and Egyptian geometry to the algebra of medieval scholars "
    "and the calculus developed independently by Newton and Leibniz in the seventeenth century.",
    "Photosynthesis is the biochemical process by which green plants, algae, and some bacteria "
    "convert sunlight, water, and atmospheric carbon dioxide into glucose and oxygen, forming the "
    "foundation of nearly every food chain on the planet and shaping the composition of the atmosphere.",
    "In modern software engineering, modular design reduces coupling between components, so that "
    "each module can be developed, tested, and replaced independently; this discipline becomes more "
    "important as systems grow and as teams collaborate across many time zones and codebases.",
    "Quarterly revenue grew steadily throughout the year as the company expanded into new international "
    "markets, launched two flagship products, and invested heavily in customer support, although rising "
    "logistics costs and currency fluctuations weighed noticeably on the reported operating margin.",
    "Quantum computers exploit the principles of superposition and entanglement to represent and process "
    "information in ways that classical machines cannot easily replicate, promising dramatic speedups for "
    "certain problems in cryptography, materials simulation, and optimization, though many engineering "
    "obstacles remain.",
    "The treaty was finally signed after many months of difficult and often contentious diplomatic "
    "negotiation, during which envoys from a dozen nations argued over borders, trade tariffs, and the "
    "precise wording of clauses that each delegation feared might later be interpreted against its own "
    "national interest.",
]

print("loading model...")
model = LlamaModel.from_pretrained(MODEL, device=dev)
tok = AutoTokenizer.from_pretrained(MODEL)

import ops.mlp as _mlp_mod
import ops.attention as _attn_mod
_mlp_mod.USE_TRITON = False
_attn_mod.USE_TRITON = False

for i, layer in enumerate(model.layers):
    layer.mlp._layer_id = i

# per-layer SmoothQuant scale s (I,), filled by calibration pass
SQ_S: dict[int, torch.Tensor] = {}
MODE = "bf16"
GU_W8A8 = False  # when True, gate_up also runs W8A8 (mirrors real engine)


def _qpt(x):  # per-token (per-row) int8 round-trip
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(x / s).clamp_(-127, 127) * s


def _qpc(w):  # per-output-channel int8 round-trip
    s = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    return torch.round(w / s).clamp_(-127, 127) * s


def patched_forward(self, x, x_int8=None, x_scale=None):
    w_gate_up = self.w_gate_up_int8.to(x.dtype) * self.w_gate_up_scale.to(x.dtype)[:, None]
    w_down = self.w_down_int8.to(x.dtype) * self.w_down_scale.to(x.dtype)[:, None]
    if GU_W8A8:
        xi = _qpt(x.float())
        combined = F.linear(xi, _qpc(w_gate_up.float()).to(xi.dtype)).to(x.dtype)
    else:
        combined = F.linear(x, w_gate_up)
    gate, up = combined.chunk(2, dim=-1)
    fused = F.silu(gate) * up  # (B, T, I) down input

    if MODE == "bf16":
        return F.linear(fused, w_down)
    if MODE == "w8a16":
        return F.linear(fused, _qpc(w_down.float()).to(x.dtype))
    if MODE == "w8a8_naive":
        f = _qpt(fused.float())
        return F.linear(f, _qpc(w_down.float()).to(f.dtype)).to(x.dtype)
    if MODE == "w8a8_sq":
        s = SQ_S[self._layer_id].to(fused.dtype)  # (I,)
        fs = _qpt((fused / s).float())
        ws = _qpc((w_down * s).float())
        return F.linear(fs, ws.to(fs.dtype)).to(x.dtype)
    raise ValueError(MODE)


_orig_forward = type(model.layers[0].mlp).forward
type(model.layers[0].mlp).forward = patched_forward


def run_all(mode):
    global MODE
    MODE = mode
    outs = []
    with torch.no_grad():
        for text in CALIB:
            ids = tok(text, return_tensors="pt").input_ids.to(dev)
            logits = model.forward(ids, start_pos=0, kv_cache=None)  # (1, T, V)
            outs.append(logits[0].float().cpu())
    return outs


# --- calibration: per-layer per-channel act_max of fused -> SmoothQuant s ---
print("calibrating SmoothQuant scales (a=%.2f)..." % ALPHA)
act_max = {i: None for i in range(len(model.layers))}


def calib_hook(i):
    def hook(mod, args):
        x = args[0]
        w_gate_up = mod.w_gate_up_int8.to(x.dtype) * mod.w_gate_up_scale.to(x.dtype)[:, None]
        gate, up = F.linear(x, w_gate_up).chunk(2, dim=-1)
        fused = (F.silu(gate) * up).reshape(-1, mod.intermediate_size).float()
        m = fused.abs().amax(dim=0)  # (I,)
        act_max[i] = m if act_max[i] is None else torch.maximum(act_max[i], m)
    return hook


handles = [l.mlp.register_forward_pre_hook(calib_hook(i)) for i, l in enumerate(model.layers)]
MODE = "bf16"
with torch.no_grad():
    for text in CALIB:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        model.forward(ids, start_pos=0, kv_cache=None)
for h in handles:
    h.remove()
for i, l in enumerate(model.layers):
    w = (l.mlp.w_down_int8.float() * l.mlp.w_down_scale.float()[:, None]).to(dev)
    w_max = w.abs().amax(dim=0).clamp_min(1e-8)  # (I,)
    am = act_max[i].to(dev).clamp_min(1e-8)
    SQ_S[i] = ((am ** ALPHA) / (w_max ** (1 - ALPHA))).clamp_min(1e-8)

# --- evaluate ---
ref = run_all("bf16")
ref_arg = [r.argmax(dim=-1) for r in ref]


def report(mode):
    outs = run_all(mode)
    tot = match = 0
    cos_sum = 0.0
    cos_n = 0
    for r, o, ra in zip(ref, outs, ref_arg):
        oa = o.argmax(dim=-1)
        match += (oa == ra).sum().item()
        tot += ra.numel()
        cos_sum += F.cosine_similarity(r, o, dim=-1).mean().item()
        cos_n += 1
    print(f"  {mode:>12}: argmax_agree={100*match/tot:6.2f}%  logit_cos={cos_sum/cos_n:.5f}")


npos = sum(r.shape[0] for r in ref)
print(f"\n[gate_up bf16] next-token argmax agreement vs bf16 ref ({npos} positions):")
report("w8a16")
report("w8a8_naive")
report("w8a8_sq")

print(f"\n[gate_up W8A8 -- mirrors real engine] vs bf16 ref:")
GU_W8A8 = True
report("bf16")        # gate_up W8A8 only (accepted precedent)
report("w8a16")       # current engine: gate_up W8A8 + down W8A16
report("w8a8_sq")     # proposed engine: gate_up W8A8 + down W8A8 SQ

type(model.layers[0].mlp).forward = _orig_forward
