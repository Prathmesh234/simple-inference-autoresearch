"""Where does the OOM come from for long-context high-batch? Measure peak VRAM of a
prefill at increasing B*T, and whether KV-cache allocation or prefill activations
dominate. Tells us if paged KV (KV-bound) or chunked prefill (activation-bound) is
the lever to open the OOM'd long-flavor cells."""
import torch, env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
DEV="cuda"; DT=torch.bfloat16; MID="meta-llama/Llama-3.1-8B"
loader=WeightLoader.from_pretrained(MID); cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
base = torch.cuda.memory_allocated()/1e9
print(f"model weights: {base:.2f} GB")
@torch.no_grad()
def trial(B, T):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        kv = KVCache(cfg.num_hidden_layers, B, T+64, cfg.num_key_value_heads, cfg.head_dim, dtype=DT, device=DEV)
        kv_gb = kv.bytes()/1e9
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
        _ = model(ids, start_pos=0, kv_cache=kv)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()/1e9
        print(f"  B={B:>3} T={T:>4} (B*T={B*T:>7}): KV={kv_gb:5.2f}GB  PEAK={peak:5.2f}GB  (act≈{peak-base-kv_gb:.2f}GB)")
        del kv, ids
    except torch.cuda.OutOfMemoryError as e:
        print(f"  B={B:>3} T={T:>4} (B*T={B*T:>7}): OOM ({str(e)[:50]})")
        torch.cuda.empty_cache()
for B,T in [(8,948),(32,948),(64,948),(128,948),(64,2000),(128,2000),(8,4000),(32,4000)]:
    trial(B,T)
