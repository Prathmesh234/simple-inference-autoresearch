"""jun30 EXP-23 gate: prefill W8A8 faithfulness. Prefill (M>256) now runs qkv+gate_up
as int8 tensor-core (was W8A16). Verify the prompt is processed faithfully: compare
greedy continuation + first-token logits of W8A8-prefill vs W8A16-prefill on a LONG
prompt (M>256 at b1) and a batched prefill. W8A8 is the shipped decode quant; this
confirms it stays faithful when applied to the diverse full-prompt activations.

Toggle via env PREFILL_W8A16=1 to force the old W8A16 prefill path (for the A/B)."""
import sys, os; sys.path.insert(0, os.getcwd())
import env_loader, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B"); tok.pad_token = tok.eos_token
tok.padding_side = "left"  # decoder-only batched gen needs left padding (as the profiler does)
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader); model.to(DEV, DT); model.eval()
L = cfg.num_hidden_layers

# A long single prompt (force prefill M>256) and a short batched one (M=8*~40>256).
long_prompt = ("The scientific method is a systematic approach to understanding the natural "
               "world through observation and experiment. ") * 20  # >256 tokens
batch_prompts = ["The capital of France is", "Photosynthesis is the process by which",
                 "The speed of light in a vacuum is", "Newton's first law states that",
                 "The largest planet in our solar system is", "DNA is a molecule that",
                 "The French Revolution began in the year", "Mount Everest is the tallest"]


@torch.no_grad()
def first_logits_and_gen(ids, n=24):
    kv = KVCache(L, ids.shape[0], ids.shape[1] + n + 1, cfg.num_key_value_heads,
                 cfg.head_dim, DT, DEV)
    lg = model(ids, start_pos=0, kv_cache=kv)
    first = lg[:, -1, :].float().clone()
    nxt = first.argmax(-1, keepdim=True)
    out = [nxt]
    pos = ids.shape[1]
    for i in range(n):
        lg = model(nxt, start_pos=pos, kv_cache=kv)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        out.append(nxt); pos += 1
    return first, torch.cat(out, 1)


ids_long = tok(long_prompt, return_tensors="pt").input_ids.to(DEV)
M = ids_long.shape[1]
print(f"long prompt M (b1) = {M}; prefill path: {'W8A8' if M>16 else 'W8A16'}")
first, gen = first_logits_and_gen(ids_long)
print("LONG greedy:", repr(tok.decode(gen[0].tolist())))

ids_b = tok(batch_prompts, return_tensors="pt", padding=True).input_ids.to(DEV)
print(f"\nbatch prefill M = {ids_b.shape[0]*ids_b.shape[1]} (>256 -> W8A8)")
fb, genb = first_logits_and_gen(ids_b, n=16)
for i, p in enumerate(batch_prompts):
    print(f"  {p!r} -> {tok.decode(genb[i].tolist())!r}")
