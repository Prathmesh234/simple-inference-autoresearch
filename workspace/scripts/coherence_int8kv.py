"""Verify int8 KV cache: long-context generation (max_seq_len>256 -> int8 KV path)
stays coherent, and short-context (instruct-like) stays bf16. Garbage -> the int8
quant/dequant or graph capture is broken."""
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

@torch.no_grad()
def gen(prompt, n=30):
    model._decoder=None
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    msl=ids.shape[1]+n
    kv=KVCache(cfg.num_hidden_layers,1,msl,cfg.num_key_value_heads,cfg.head_dim,dtype=DT,device=DEV)
    lg=model(ids,start_pos=0,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True); out=[nt.item()]; pos=ids.shape[1]
    for _ in range(n-1):
        lg=model(nt,start_pos=pos,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True); out.append(nt.item()); pos+=1
    return ids.shape[1], kv.is_int8, tok.decode(out)

# long prompt (>256 tokens) -> int8 KV
longp = ("In computer science, a cache is a hardware or software component that stores data "
         "so that future requests for that data can be served faster. ") * 12 + \
        "The most important benefit of a KV cache in transformer inference is that it"
plen, i8, g = gen(longp, 30)
print(f"LONG  prompt_len={plen} is_int8={i8}: ...{g!r}")
# short prompt -> bf16 KV
for p in ["The capital of France is", "Two plus two equals"]:
    plen, i8, g = gen(p, 24)
    print(f"SHORT prompt_len={plen} is_int8={i8}: {g!r}")
