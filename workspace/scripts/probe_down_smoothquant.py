"""Faithfulness gate for down_proj W8A8 via SmoothQuant.

The down projection is the dominant decode op (~48%) but is stuck on W8A16
(bf16 activations) because naive per-token int8 quant of its input (the SwiGLU
output) has large per-channel outliers -> rel_err ~26 (PROGRESS: down W8A8 RTN
DEAD). SmoothQuant migrates the activation outliers into the weight via a
per-input-channel smoothing factor s:  y = (X/s) @ (W*s).T  (mathematically
unchanged), making X/s quantizable.

This probe captures REAL down inputs on GENERIC calibration text (NOT the
held-out benchmark prompts) and reports, per layer, the int8 reconstruction
rel_err for: (a) current W8A16 (weight int8, act bf16), (b) naive W8A8,
(c) SmoothQuant W8A8 over a few alpha values. Decision gate: does SmoothQuant
bring W8A8 down to the accepted ~0.06 band that gate_up/qkv W8A8 already use?
"""
import torch
from transformers import AutoTokenizer
from model.llama import LlamaModel

dev = "cuda"
MODEL = "meta-llama/Llama-3.1-8B"

# Generic calibration text (NOT from benchmarks/prompts.py). Each is long enough
# (>16 tokens) that prefill M=T lands in the 16<M<=256 W8A8 bucket, avoiding the
# M<=16 W8A16 qkv tile that overflows Ada's 101KB shared-memory limit.
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
    "The chef explained that the recipe calls for two cups of flour, a generous pinch of salt, three "
    "fresh eggs, and a tablespoon of melted butter, all whisked together slowly until the batter is "
    "smooth and free of lumps before it is poured onto the hot, lightly oiled griddle.",
    "Quarterly revenue grew steadily throughout the year as the company expanded into new international "
    "markets, launched two flagship products, and invested heavily in customer support, although rising "
    "logistics costs and currency fluctuations weighed noticeably on the reported operating margin.",
    "She walked along the deserted shore at dawn, listening to the waves break gently against the smooth "
    "gray stones, watching the pale light spread slowly across the water, and thinking about all the "
    "choices that had quietly led her, year after year, to this particular and unremarkable morning.",
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

# Force the cuBLAS reference path for attn/mlp GEMMs: we only need realistic
# down-input activations, and the Triton W8A16 qkv tile overflows Ada's 101KB
# shared-memory limit in this standalone (non-graph) context. env_loader (loaded
# via config/loader at import) sets USE_TRITON=true in os.environ, so override the
# already-cached module globals directly.
import ops.mlp as _mlp_mod
import ops.attention as _attn_mod
_mlp_mod.USE_TRITON = False
_attn_mod.USE_TRITON = False

captured = {}

def mk_hook(i):
    def hook(mod, args):
        captured[i] = args[0].detach()  # (B, T, hidden) normed mlp input
    return hook

handles = [layer.mlp.register_forward_pre_hook(mk_hook(i)) for i, layer in enumerate(model.layers)]

print("running calibration prefill...")
with torch.no_grad():
    for text in CALIB:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        model.forward(ids, start_pos=0, kv_cache=None)
        # accumulate down-inputs across prompts per layer
        for i, layer in enumerate(model.layers):
            x = captured[i].reshape(-1, captured[i].shape[-1]).float()  # (M, hidden)
            mlp = layer.mlp
            wgu = (mlp.w_gate_up_int8.float() * mlp.w_gate_up_scale.float()[:, None])
            comb = x @ wgu.T
            I = mlp.intermediate_size
            gate, up = comb[:, :I], comb[:, I:]
            fused = (gate * torch.sigmoid(gate)) * up  # (M, I) down input
            store = mlp.__dict__.setdefault("_calib", [])
            store.append(fused.cpu())

for h in handles:
    h.remove()


def per_token_int8(x):
    s = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    xi = torch.round(x / s).clamp_(-127, 127)
    return xi * s


def per_channel_int8_w(w):  # w: (N_out, K) -> per output-channel
    s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    wi = torch.round(w / s).clamp_(-127, 127)
    return wi * s


def relerr(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


print(f"\n{'layer':>5} {'W8A16':>9} {'W8A8naive':>10} {'SQ a.5':>9} {'SQ a.7':>9} {'SQ a.8':>9} {'SQ a.9':>9}")
import statistics
agg = {k: [] for k in ("w16", "naive", "a5", "a7", "a8", "a9")}
for i, layer in enumerate(model.layers):
    mlp = layer.mlp
    fused = torch.cat(mlp._calib, dim=0).to(dev)            # (Mtot, I)
    w = (mlp.w_down_int8.float() * mlp.w_down_scale.float()[:, None]).to(dev)  # (hidden, I)
    ref = fused @ w.T
    # (a) W8A16: weight int8, act bf16 (current path)
    y16 = fused @ per_channel_int8_w(w).T
    # (b) naive W8A8
    y8 = per_token_int8(fused) @ per_channel_int8_w(w).T
    row = [relerr(y16, ref), relerr(y8, ref)]
    agg["w16"].append(row[0]); agg["naive"].append(row[1])
    act_max = fused.abs().amax(dim=0).clamp_min(1e-8)       # (I,)
    w_max = w.abs().amax(dim=0).clamp_min(1e-8)             # (I,)
    for a, key in ((0.5, "a5"), (0.7, "a7"), (0.8, "a8"), (0.9, "a9")):
        s = (act_max ** a) / (w_max ** (1 - a))
        s = s.clamp_min(1e-8)
        fs = fused / s[None, :]
        ws = w * s[None, :]
        ysq = per_token_int8(fs) @ per_channel_int8_w(ws).T
        e = relerr(ysq, ref)
        agg[key].append(e); row.append(e)
    del mlp._calib
    print(f"{i:>5} " + " ".join(f"{v:>9.4f}" if j else f"{v:>9.4f}" for j, v in enumerate([row[0], row[1], row[2], row[3], row[4], row[5]])))

print("\nMEAN over layers:")
for k in ("w16", "naive", "a5", "a7", "a8", "a9"):
    print(f"  {k:>6}: {statistics.mean(agg[k]):.4f}  (max {max(agg[k]):.4f})")
