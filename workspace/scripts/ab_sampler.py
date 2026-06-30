"""Same-process A/B for the fused sampler (EXP-21): load the model ONCE, time the
b128 decode step with USE_FUSED_SAMPLE on vs off, alternating, min-of-N — both
clean and under torch.profiler(profile_memory=True) to mirror the headline metric.

Uses generic ~30-token prompts (NOT benchmarks/prompts.py) so kv_len ~= instruct.
decode_ms here is not the official number; only the fused-vs-base DELTA matters.
"""
import sys, os, time; sys.path.insert(0, os.getcwd())
import env_loader, torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
import sampling
from sampling import sample

DEV, DT = "cuda", torch.bfloat16
loader = WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B"); tok.pad_token = tok.eos_token
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader); model.to(DEV, DT); model.eval()

B = 128
base = "Tell me a short interesting fact about the history of science and technology."
ids = tok([base] * B, return_tensors="pt", padding=True).input_ids.to(DEV)
P = ids.shape[1]
print(f"B={B} prompt_len={P}")

@torch.no_grad()
def decode_ms(n_steps=40, profiled=False):
    kv = KVCache(cfg.num_hidden_layers, B, P + n_steps + 4, cfg.num_key_value_heads,
                 cfg.head_dim, DT, DEV)
    # warmup (also triggers graph capture)
    logits = model(ids, start_pos=0, kv_cache=kv)
    nxt = sample(logits[:, -1, :], 0.7, 50, 0.9).unsqueeze(-1)
    for i in range(4):
        logits = model(nxt, start_pos=P + i, kv_cache=kv)
        nxt = sample(logits[:, -1, :], 0.7, 50, 0.9).unsqueeze(-1)
    torch.cuda.synchronize()
    def run():
        ts = []
        pos = P + 4
        n = nxt.clone()
        for _ in range(n_steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lg = model(n, start_pos=pos, kv_cache=kv)
            n = sample(lg[:, -1, :], 0.7, 50, 0.9).unsqueeze(-1)
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
            pos += 1
        return sum(ts) / len(ts)
    if profiled:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True):
            return run()
    return run()

# Alternate fused/base, min-of-N, for both clean and profiled timing.
for label, profiled in (("clean", False), ("profiled", True)):
    res = {True: [], False: []}
    for rep in range(4):
        for fused in (True, False):
            sampling.USE_FUSED_SAMPLE = fused
            res[fused].append(decode_ms(profiled=profiled))
    f = min(res[True]); b = min(res[False])
    print(f"[{label:8s}] fused={f:.3f}ms  base={b:.3f}ms  "
          f"delta={(b-f):+.3f}ms ({(b/f-1)*100:+.2f}% faster fused)  "
          f"agg_fused={B*1000/f:.0f} agg_base={B*1000/b:.0f}")
