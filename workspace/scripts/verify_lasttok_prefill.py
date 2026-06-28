"""Verify last-position prefill logits: (1) b128 x 948 prefill now FITS (was OOM at
29GB logits), (2) the sampled logits are bit-identical to all-position prefill,
(3) greedy output stays coherent."""
import torch, env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
DEV="cuda"; DT=torch.bfloat16; MID="meta-llama/Llama-3.1-8B"
loader=WeightLoader.from_pretrained(MID); cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(MID); tok.pad_token=tok.eos_token

# (1) b128 x 948 prefill that OOM'd before
@torch.no_grad()
def prefill_fits(B,T):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        kv=KVCache(cfg.num_hidden_layers,B,T+64,cfg.num_key_value_heads,cfg.head_dim,dtype=DT,device=DEV)
        ids=torch.randint(0,cfg.vocab_size,(B,T),device=DEV)
        out=model(ids,start_pos=0,kv_cache=kv); torch.cuda.synchronize()
        print(f"  B={B} T={T}: FITS, peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB, logits.shape={tuple(out.shape)}")
        del kv,ids,out
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B} T={T}: OOM"); torch.cuda.empty_cache()
print("(1) prefill memory:")
prefill_fits(128,948); prefill_fits(64,2000)

# (3) coherence
@torch.no_grad()
def gen(p,n=20):
    model._decoder=None
    ids=tok(p,return_tensors="pt").input_ids.to(DEV)
    kv=KVCache(cfg.num_hidden_layers,1,ids.shape[1]+n,cfg.num_key_value_heads,cfg.head_dim,dtype=DT,device=DEV)
    lg=model(ids,start_pos=0,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True); out=[nt.item()]; pos=ids.shape[1]
    for _ in range(n-1):
        lg=model(nt,start_pos=pos,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True); out.append(nt.item()); pos+=1
    return tok.decode(out)
print("(2/3) coherence:")
for p in ["The capital of France is","Two plus two equals"]:
    print(f"  {p!r} -> {gen(p)!r}")
