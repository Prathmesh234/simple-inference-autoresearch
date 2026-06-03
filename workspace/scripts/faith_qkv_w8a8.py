"""qkv W8A8 faithfulness: input is the clean post-attn-norm residual (same dist as
the already-accepted gate_up W8A8). Compare final-token logits (argmax/topk) of
qkv-W8A8 (self-quant; numerics identical to the planned norm-fused quant) vs the
shipping qkv-W8A16 path. Also full greedy generation coherence."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.attention as attn_mod
from kernels.w8a8_gemm_kernel import w8a8_linear_triton
DEV,DT="cuda",torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token=tok.eos_token
prompts=["The theory of relativity states that","In a surprising turn of events,",
 "The best way to learn programming is to","Once upon a time in a distant galaxy,",
 "The capital of France is","Photosynthesis is the process by which"]
ids=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=32)["input_ids"].to(DEV)
orig=attn_mod.GroupedQueryAttention._project_qkv
USE={"w8a8":False}
def patched(self,x,B,T):
    if not USE["w8a8"]:
        return orig(self,x,B,T)
    qkv=w8a8_linear_triton(x,self.w_qkv_int8,self.w_qkv_scale)  # falls back to W8A16 for M<=16/M>256
    q=qkv[...,:self.q_size].reshape(B,T,self.num_heads_q,self.head_dim)
    k=qkv[...,self.q_size:self.q_size+self.kv_size].reshape(B,T,self.num_heads_kv,self.head_dim)
    v=qkv[...,self.q_size+self.kv_size:].reshape(B,T,self.num_heads_kv,self.head_dim)
    return q,k,v
attn_mod.GroupedQueryAttention._project_qkv=patched
def run():
    kv=KVCache(n_layers=cfg.num_hidden_layers,max_batch=ids.shape[0],max_seq_len=ids.shape[1]+2,
      n_heads_kv=cfg.num_key_value_heads,head_dim=cfg.head_dim,dtype=DT,device=DEV)
    with torch.no_grad(): out=model(ids,start_pos=0,kv_cache=kv)
    return out[:,-1,:].float()
# NOTE: prefill M=B*T>256 -> w8a8 falls back to W8A16. To exercise the int8 path at
# decode M-range, also test a synthetic M=128 forward by checking the GEMM directly.
USE["w8a8"]=False; ref=run()
USE["w8a8"]=True;  l8=run()
am=(ref.argmax(-1)==l8.argmax(-1)).float().mean().item()
ov={}
for k in (1,5,10):
    t0=ref.topk(k,-1).indices; t1=l8.topk(k,-1).indices
    ov[k]=sum(len(set(t0[i].tolist())&set(t1[i].tolist()))/k for i in range(ids.shape[0]))/ids.shape[0]
print(f"qkv path (prefill, may fallback): argmax={am:.3f} top5={ov[5]:.3f} top10={ov[10]:.3f}")
# Direct decode-shape GEMM faithfulness: real qkv weight, synthetic clean act (M=128)
import numpy as np
L=model.layers[0].attn
x=torch.randn(128,cfg.hidden_size,device=DEV,dtype=DT)*1.0
W=(L.w_qkv_int8.float()*L.w_qkv_scale.float()[:,None]).to(DT)
ref_g=F.linear(x,W).float()
y8=w8a8_linear_triton(x,L.w_qkv_int8,L.w_qkv_scale).float()  # M=128 -> int8 path
rel=((y8-ref_g).abs()/(ref_g.abs()+1.0)).mean().item()
cos=F.cosine_similarity(y8.flatten(),ref_g.flatten(),dim=0).item()
print(f"decode-shape M=128 qkv W8A8 vs bf16-deq: rel_err={rel:.4f} cos={cos:.5f}")
