"""Clean long-decode faithfulness: qkv W8A8 (self-quant from the SAME bf16 normed
x, numerically == the shipped norm-fused quant) vs qkv W8A16 (shipping path), both
starting from identical bf16 activations. 256-token greedy, KV-drift tracking."""
import sys, torch, statistics as st
sys.path.insert(0,"/home/ubuntu/simple-inference-autoresearch")
import env_loader  # noqa
from config import ModelConfig
from loader import WeightLoader
from model.llama import LlamaModel
from model.kv_cache import KVCache
from transformers import AutoTokenizer
import ops.attention as attn_mod
from kernels.w8a8_gemm_kernel import w8a8_linear_triton
from kernels.w8a16_gemm_kernel import w8a16_linear_triton
DEV,DT="cuda",torch.bfloat16
loader=WeightLoader.from_pretrained("meta-llama/Llama-3.1-8B")
cfg=ModelConfig.from_hf_config(loader.model_dir/"config.json")
model=LlamaModel(cfg,torch.device(DEV)); model.load_weights(loader); model.to(DEV,DT); model.eval()
tok=AutoTokenizer.from_pretrained(loader.model_dir); tok.pad_token=tok.eos_token
B=48
base=["The theory of relativity states that","In a surprising turn of events, scientists",
 "The best way to learn programming is to","Once upon a time in a distant galaxy,",
 "The capital of France is","Photosynthesis is the process by which plants",
 "Climate change is primarily caused by","The history of the Roman Empire shows"]
prompts=(base*6)[:B]
ids0=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=24)["input_ids"].to(DEV)
orig=attn_mod.GroupedQueryAttention._project_qkv
MODE={"m":"ref"}
def patched(self,x,B,T,x_int8=None,x_scale=None):
    proj = w8a8_linear_triton if MODE["m"]=="w8a8" else (lambda x,wi,s: w8a16_linear_triton(x,wi,s))
    qkv=proj(x,self.w_qkv_int8,self.w_qkv_scale)
    q=qkv[...,:self.q_size].reshape(B,T,self.num_heads_q,self.head_dim)
    k=qkv[...,self.q_size:self.q_size+self.kv_size].reshape(B,T,self.num_heads_kv,self.head_dim)
    v=qkv[...,self.q_size+self.kv_size:].reshape(B,T,self.num_heads_kv,self.head_dim)
    return q,k,v
attn_mod.GroupedQueryAttention._project_qkv=patched
def gen(mode,nnew):
    MODE["m"]=mode
    kv=KVCache(n_layers=cfg.num_hidden_layers,max_batch=B,max_seq_len=ids0.shape[1]+nnew+1,
      n_heads_kv=cfg.num_key_value_heads,head_dim=cfg.head_dim,dtype=DT,device=DEV)
    out=[]
    with torch.no_grad():
        lg=model(ids0,start_pos=0,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True)
        out.append(nt); pos=ids0.shape[1]
        for _ in range(nnew-1):
            lg=model(nt,start_pos=pos,kv_cache=kv); nt=lg[:,-1,:].argmax(-1,keepdim=True)
            out.append(nt); pos+=1
    return torch.cat(out,1)
for nnew in (256,):
    gr=gen("ref",nnew); g8=gen("w8a8",nnew)
    m=(gr==g8).float().mean().item(); ex=(gr==g8).all(1).float().mean().item()
    print(f"nnew={nnew}: token-match={m:.3f} exact-seq={ex:.3f}")
print("REF :",repr(tok.decode(gr[0])[:160]))
print("W8A8:",repr(tok.decode(g8[0])[:160]))
print("REF :",repr(tok.decode(gr[2])[:160]))
print("W8A8:",repr(tok.decode(g8[2])[:160]))
