"""Verify EXP-11: at B in the W8A8 bucket (M>16) the norm's bf16 'normed' output is
skipped (garbage). If its data were read downstream, batched greedy gen would be
garbage. Test at B=32 (M=32 -> emit_bf16=False) AND B=1 (M=1 -> emit_bf16=True)."""
import torch
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
DEV="cuda"; DT=torch.bfloat16; MID="meta-llama/Llama-3.1-8B"
loader=WeightLoader.from_pretrained(MID); cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(MID); tok.pad_token=tok.eos_token; tok.padding_side="left"

@torch.no_grad()
def gen_batch(prompts, n=24):
    model._decoder=None
    enc=tok(prompts,return_tensors="pt",padding=True); ids=enc.input_ids.to(DEV)
    B,Tp=ids.shape
    kv=KVCache(cfg.num_hidden_layers,B,Tp+n,cfg.num_key_value_heads,cfg.head_dim,dtype=DT,device=DEV)
    logits=model(ids,start_pos=0,kv_cache=kv); nt=logits[:,-1,:].argmax(-1,keepdim=True)
    outs=[nt]; pos=Tp
    for _ in range(n-1):
        logits=model(nt,start_pos=pos,kv_cache=kv); nt=logits[:,-1,:].argmax(-1,keepdim=True)
        outs.append(nt); pos+=1
    g=torch.cat(outs,dim=1)
    return [tok.decode(g[i]) for i in range(B)]

prompts=["The capital of France is","Two plus two equals","The sun rises in the",
         "Water boils at a temperature of","The opposite of hot is"]
# B=32 (M=32 -> emit_bf16=False, the new skip path)
b32 = gen_batch(prompts*7)[:5]   # 35 prompts -> B=35 (M=35, in 16<M<=256)
print("=== B=35 (emit_bf16=FALSE, bf16 normed skipped) ===")
for p,o in zip(prompts, b32): print(f"  {p!r} -> {o!r}")
# B=1 (M=1 -> emit_bf16=True path)
b1 = gen_batch([prompts[0]])
print("=== B=1 (emit_bf16=TRUE) ===")
print(f"  {prompts[0]!r} -> {b1[0]!r}")
