"""Does int4 (per-channel / grouped RTN) lm_head weight preserve top-k ordering?
Tests on REAL final-norm hidden states. If top50 overlap stays high, a W4 lm_head
could halve the 525MB->263MB weight read on the biggest decode memory op."""
import sys, torch
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.embedding as emb_mod
DEV,DT="cuda",torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token=tok.eos_token
B=24
prompts=["The theory of relativity states that","In a surprising turn of events, scientists discovered that",
 "The best way to learn programming is to","Once upon a time in a distant galaxy,",
 "The capital of France is","Water boils at a temperature of"]
prompts=(prompts*4)[:B]
ids0=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=32)["input_ids"].to(DEV)
cap={}
of=emb_mod.OutputProjection.forward
def rf(self,x): cap['x']=x.detach().clone(); return of(self,x)
emb_mod.OutputProjection.forward=rf
kv=KVCache(n_layers=cfg.num_hidden_layers,max_batch=B,max_seq_len=ids0.shape[1]+2,
  n_heads_kv=cfg.num_key_value_heads,head_dim=cfg.head_dim,dtype=DT,device=DEV)
with torch.no_grad(): model(ids0,start_pos=0,kv_cache=kv)
emb_mod.OutputProjection.forward=of
x=cap['x'][:,-1,:].float()  # (B,H)

# reconstruct original bf16 lm_head weight from stored int8 (good enough proxy is the real weight)
W = (model.head.w_int8.float()*model.head.w_scale.float()[:,None])  # (V,H) ~ original
l_ref = (x @ W.T)

def quant_int4_grouped(W, group=128):
    V,H=W.shape
    Wg=W.view(V,H//group,group)
    amax=Wg.abs().amax(-1,keepdim=True)
    scale=amax/7.0
    q=torch.clamp(torch.round(Wg/scale),-8,7)
    deq=(q*scale).view(V,H)
    return deq
for group in (128, 64, 32):
    Wd=quant_int4_grouped(W,group)
    l4=(x @ Wd.T)
    am=(l_ref.argmax(-1)==l4.argmax(-1)).float().mean()
    ov={}
    for k in (1,5,50):
        t0=l_ref.topk(k,-1).indices; t1=l4.topk(k,-1).indices
        ov[k]=torch.tensor([len(set(t0[i].tolist())&set(t1[i].tolist()))/k for i in range(B)]).mean().item()
    rel=((l4-l_ref).abs()/(l_ref.abs()+1.0))
    print(f"int4 g={group}: argmax={am:.3f} top1={ov[1]:.3f} top5={ov[5]:.3f} top50={ov[50]:.3f} rel_err mean={rel.mean():.4f}")
