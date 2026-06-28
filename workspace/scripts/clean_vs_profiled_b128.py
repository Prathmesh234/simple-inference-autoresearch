"""jun27 EXP-8 probe: measure CLEAN b128 decode_ms (no torch.profiler) vs the
profiled headline (13.6ms). The headline cell is timed UNDER torch.profiler with
~10 eager ops/step (sample + _update_pos) still outside the CUDA graph. If clean <<
profiled, the eager-op profiler tax is recoverable by moving eager ops into the graph
(the EXP1-10 launch-count lever). If clean ~= profiled, the time is real GPU work.

Uses random tokens at the headline shape (B=128, prompt_len=29) — timing is
shape-dependent only, not content. Also breaks out model-only vs +sample.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import env_loader  # noqa
import time, torch
from torch.profiler import profile, ProfilerActivity
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from sampling import sample

DEV = "cuda"; DT = torch.bfloat16
MODEL = "meta-llama/Llama-3.1-8B"
B, PROMPT, NDEC = 128, 29, 64
TEMP, TOPK, TOPP = 0.7, 50, 0.9

loader = WeightLoader.from_pretrained(MODEL)
cfg = ModelConfig.from_hf_config(loader.model_dir / "config.json")
model = LlamaModel(cfg, torch.device(DEV)); model.load_weights(loader)
model.to(DEV, DT); model.eval()

torch.manual_seed(0)
ids = torch.randint(0, cfg.vocab_size, (B, PROMPT), device=DEV)
kv = KVCache(cfg.num_hidden_layers, B, PROMPT + NDEC, cfg.num_key_value_heads,
             cfg.head_dim, dtype=DT, device=DEV)


@torch.no_grad()
def decode_loop(do_sample=True):
    kv.reset()
    logits = model(ids, start_pos=0, kv_cache=kv)
    nt = sample(logits[:, -1, :], temperature=TEMP, top_k=TOPK, top_p=TOPP).unsqueeze(-1)
    times = []
    pos = PROMPT
    for _ in range(NDEC - 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits = model(nt, start_pos=pos, kv_cache=kv)
        if do_sample:
            nt = sample(logits[:, -1, :], temperature=TEMP, top_k=TOPK, top_p=TOPP).unsqueeze(-1)
        else:
            nt = logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
        pos += 1
    return sum(times) / len(times)


# warmup (triggers graph capture)
for _ in range(3):
    decode_loop()

clean_full = min(decode_loop() for _ in range(3))
clean_model_only = min(decode_loop(do_sample=False) for _ in range(3))

# profiled (mimic profile_engine: full trace over the decode loop)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True):
    prof_full = decode_loop()

print(f"\n  B={B} prompt={PROMPT} decode={NDEC}")
print(f"  CLEAN decode_ms (model+sample) : {clean_full:.3f}")
print(f"  CLEAN decode_ms (model only)   : {clean_model_only:.3f}  (sample={clean_full-clean_model_only:.3f})")
print(f"  PROFILED decode_ms (model+sample): {prof_full:.3f}")
print(f"  profiler tax: {prof_full-clean_full:.3f} ms ({100*(prof_full-clean_full)/clean_full:.1f}%)")
print(f"  => agg clean={B*1000/clean_full:.0f}  profiled={B*1000/prof_full:.0f} tok/s")
